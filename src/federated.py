"""
federated.py — FedAvg server and client implementation for FL-IDS.

Implements the Federated Averaging algorithm (McMahan et al., 2017) as a
standalone PyTorch simulation. No external FL framework dependency.

Components:
    FedAvgClient  — represents one IIoT site; trains locally and returns weights
    FedAvgServer  — aggregates client weights via weighted averaging
    evaluate_model — computes accuracy, macro-F1, precision, recall, per-class F1
    run_federated  — orchestrates the full FL loop for R rounds

Usage:
    from federated import run_federated
    results = run_federated(
        data_path="data/processed/datasense_preprocessed.csv",
        partition_path="data/partitions/iid_K5_seed42.json",
        num_rounds=50, local_epochs=5, lr=0.01, batch_size=64,
        participation_fraction=1.0, device="cpu",
    )
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from model import (
    create_model,
    get_model_params,
    set_model_params,
    compute_model_size_bytes,
)
from preprocessing import (
    load_processed_data,
    apply_scaler,
    compute_class_weights_from_labels,
    NUM_FEATURES,
    NUM_CLASSES,
)
from partitioning import load_partition


# ======================================================================
#  EVALUATION
# ======================================================================


def evaluate_model(model, X_test, y_test, device="cpu", batch_size=512):
    """
    Evaluate a model on a test set.

    Parameters
    ----------
    model : IDSNet
    X_test : np.ndarray, shape (n, 17)   — already scaled
    y_test : np.ndarray, shape (n,)
    device : str
    batch_size : int

    Returns
    -------
    metrics : dict
        Keys: accuracy, f1_macro, precision_macro, recall_macro, per_class_f1
    """
    model.eval()
    all_preds = []

    dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)

    y_pred = np.concatenate(all_preds)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "per_class_f1": f1_score(
            y_test,
            y_pred,
            average=None,
            labels=list(range(NUM_CLASSES)),
            zero_division=0,
        ).tolist(),
    }


# ======================================================================
#  FEDAVG CLIENT
# ======================================================================


class FedAvgClient:
    """
    Represents one IIoT site in the federated system.

    A client does exactly three things each round:
    1. Receives global weights from the server
    2. Trains on its local data for E epochs
    3. Returns its updated weights to the server

    Parameters
    ----------
    client_id : int
    X_train : np.ndarray — scaled features for this client's training split
    y_train : np.ndarray — labels for this client's training split
    device : str
    """

    def __init__(self, client_id, X_train, y_train, device="cpu"):
        self.client_id = client_id
        self.n_samples = len(y_train)
        self.device = device

        # Compute per-client class weights (handles missing classes gracefully)
        class_weights = compute_class_weights_from_labels(y_train, NUM_CLASSES)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

        # Build dataset
        self.dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )

    def train_local(self, global_params, local_epochs, lr, batch_size):
        """
        Receive global weights, train locally, return updated weights.

        Parameters
        ----------
        global_params : list[np.ndarray]
            Current global model parameters from the server.
        local_epochs : int (E)
            Number of full passes over local data.
        lr : float
            Learning rate for SGD.
        batch_size : int

        Returns
        -------
        updated_params : list[np.ndarray]
            Model parameters after local training.
        n_samples : int
            Number of training samples (for weighted aggregation).
        train_loss : float
            Average loss over the last epoch.
        """
        # 1. Create a fresh local model and load global weights
        model = create_model(device=self.device)
        set_model_params(model, global_params)

        # 2. Setup training
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        # 3. Train for E local epochs
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0

        for epoch in range(local_epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_batches += 1

        avg_loss = epoch_loss / max(epoch_batches, 1)

        # 4. Extract and return updated weights
        updated_params = get_model_params(model)
        return updated_params, self.n_samples, avg_loss


# ======================================================================
#  FEDAVG SERVER
# ======================================================================


class FedAvgServer:
    """
    Central server that orchestrates Federated Averaging.

    Holds the global model and performs weighted aggregation of client updates.
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.global_model = create_model(device=device)
        self.model_size = compute_model_size_bytes(self.global_model)

    def get_global_params(self):
        """Return current global model parameters."""
        return get_model_params(self.global_model)

    def aggregate(self, client_results):
        """
        FedAvg weighted aggregation.

        w_global = sum( (n_k / n_total) * w_k )

        Parameters
        ----------
        client_results : list of (params, n_samples, loss)
            Output from each participating client's train_local().
        """
        total_samples = sum(n for _, n, _ in client_results)

        # Initialize averaged params with zeros
        avg_params = [np.zeros_like(p) for p in client_results[0][0]]

        # Weighted sum
        for params, n_k, _ in client_results:
            weight = n_k / total_samples
            for i, p in enumerate(params):
                avg_params[i] += weight * p

        # Update global model
        set_model_params(self.global_model, avg_params)


# ======================================================================
#  MAIN FL ORCHESTRATION
# ======================================================================


def run_federated(
    data_path,
    partition_path,
    num_rounds=50,
    local_epochs=5,
    lr=0.01,
    batch_size=64,
    participation_fraction=1.0,
    test_fraction=0.2,
    device="cpu",
    seed=42,
    verbose=True,
):
    """
    Run a complete FedAvg experiment.

    Parameters
    ----------
    data_path : str
        Path to datasense_preprocessed.csv.
    partition_path : str
        Path to a partition JSON file (e.g., iid_K5_seed42.json).
    num_rounds : int (R)
        Number of communication rounds.
    local_epochs : int (E)
        Local epochs per client per round.
    lr : float
        SGD learning rate.
    batch_size : int
    participation_fraction : float
        Fraction of clients selected each round (1.0 = all participate).
    test_fraction : float
        Fraction of data held out globally for evaluation.
    device : str
    seed : int
    verbose : bool

    Returns
    -------
    results : dict with keys:
        config : dict of experiment parameters
        history : list of per-round metric dicts
        final_metrics : dict of final global model metrics
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # ── 1. Load data and partition ────────────────────────────────────
    X_all, y_all, _ = load_processed_data(data_path)
    client_indices, partition_meta = load_partition(partition_path)
    num_clients = len(client_indices)

    if verbose:
        print(
            f"Loaded: {len(y_all):,} samples, "
            f"{num_clients} clients ({partition_meta['strategy']})"
        )

    # ── 2. Create global test set ─────────────────────────────────────
    # Hold out test_fraction of ALL data (stratified) for global evaluation.
    # The remaining data is partitioned among clients.
    all_indices = np.arange(len(y_all))
    train_pool, test_indices = train_test_split(
        all_indices,
        test_size=test_fraction,
        stratify=y_all,
        random_state=seed,
    )
    train_pool_set = set(train_pool.tolist())

    # Scale test set using the full train pool (not per-client)
    X_train_pool = X_all[train_pool]
    X_test_raw = X_all[test_indices]
    y_test = y_all[test_indices]

    # Fit scaler on full train pool, transform test
    _, X_test_scaled, global_scaler = apply_scaler(X_train_pool, X_test_raw)

    if verbose:
        print(f"Global test set: {len(y_test):,} samples")

    # ── 3. Create clients ─────────────────────────────────────────────
    # Each client gets its partition indices MINUS the global test set,
    # then fits its own scaler on its local data.
    clients = []
    for k in sorted(client_indices.keys()):
        # Filter out test indices from this client's data
        local_indices = [i for i in client_indices[k] if i in train_pool_set]
        if len(local_indices) == 0:
            continue

        X_local = X_all[local_indices]
        y_local = y_all[local_indices]

        # Fit scaler on this client's local data (FL-correct)
        X_local_scaled, _, _ = apply_scaler(X_local)

        client = FedAvgClient(
            client_id=k,
            X_train=X_local_scaled,
            y_train=y_local,
            device=device,
        )
        clients.append(client)

    if verbose:
        print(
            f"Created {len(clients)} clients "
            f"(samples: {[c.n_samples for c in clients]})"
        )

    # ── 4. Initialize server ──────────────────────────────────────────
    server = FedAvgServer(device=device)
    n_participating = max(1, int(len(clients) * participation_fraction))

    if verbose:
        print(
            f"Model size: {server.model_size:,} bytes "
            f"({server.model_size/1024:.1f} KB)"
        )
        print(f"Participation: {n_participating}/{len(clients)} clients/round")
        print(
            f"Starting FedAvg: R={num_rounds}, E={local_epochs}, "
            f"lr={lr}, batch={batch_size}"
        )
        print("─" * 70)

    # ── 5. FedAvg training loop ───────────────────────────────────────
    history = []

    for round_num in range(1, num_rounds + 1):
        round_start = time.time()

        # Select participating clients
        if n_participating < len(clients):
            selected = rng.choice(len(clients), size=n_participating, replace=False)
        else:
            selected = np.arange(len(clients))

        # Broadcast global params and collect client updates
        global_params = server.get_global_params()
        client_results = []

        for idx in selected:
            client = clients[idx]
            params, n_samples, loss = client.train_local(
                global_params, local_epochs, lr, batch_size
            )
            client_results.append((params, n_samples, loss))

        # Aggregate
        server.aggregate(client_results)

        # Evaluate global model on test set
        metrics = evaluate_model(server.global_model, X_test_scaled, y_test, device)

        # Communication cost for this round
        bytes_this_round = server.model_size * n_participating * 2
        round_time = time.time() - round_start

        # Per-client average loss
        avg_client_loss = np.mean([loss for _, _, loss in client_results])

        # Build round record
        round_record = {
            "round": round_num,
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "per_class_f1": metrics["per_class_f1"],
            "avg_client_loss": avg_client_loss,
            "bytes_exchanged": bytes_this_round,
            "round_time_sec": round_time,
            "n_participating": n_participating,
        }
        history.append(round_record)

        if verbose and (round_num % max(1, num_rounds // 10) == 0 or round_num == 1):
            print(
                f"  Round {round_num:>3d}/{num_rounds}  "
                f"acc={metrics['accuracy']:.4f}  "
                f"f1={metrics['f1_macro']:.4f}  "
                f"loss={avg_client_loss:.4f}  "
                f"time={round_time:.1f}s"
            )

    # ── 6. Final evaluation ───────────────────────────────────────────
    final_metrics = evaluate_model(server.global_model, X_test_scaled, y_test, device)

    total_bytes = sum(r["bytes_exchanged"] for r in history)
    total_time = sum(r["round_time_sec"] for r in history)

    if verbose:
        print("─" * 70)
        print(f"Finished {num_rounds} rounds in {total_time:.1f}s")
        print(f"Final accuracy:  {final_metrics['accuracy']:.4f}")
        print(f"Final F1 (macro): {final_metrics['f1_macro']:.4f}")
        print(f"Total comm cost: {total_bytes / (1024*1024):.2f} MB")

    # ── 7. Build results dict ─────────────────────────────────────────
    results = {
        "config": {
            "data_path": data_path,
            "partition_path": partition_path,
            "strategy": partition_meta["strategy"],
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "lr": lr,
            "batch_size": batch_size,
            "participation_fraction": participation_fraction,
            "test_fraction": test_fraction,
            "seed": seed,
        },
        "history": history,
        "final_metrics": final_metrics,
        "total_bytes": total_bytes,
        "total_time_sec": total_time,
    }

    return results
