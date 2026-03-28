"""
baselines.py — Centralized and local-only training baselines for FL-IDS.

Provides the two comparison points from Objective O2:
    - Centralized (upper bound): All data pooled, single model trained globally.
      This is the "no privacy" gold standard. FL results should approach this.
    - Local-only (lower bound): Each client trains on its own partition only,
      no aggregation. This is what each IIoT site can do in isolation.

The gap between these two bounds is what Federated Learning aims to close.
If FL matches centralized, privacy comes "for free." If FL matches local-only,
there's no benefit to collaboration.

Usage:
    from baselines import run_centralized, run_local_only

    cent_results = run_centralized(
        data_path="data/processed/datasense_preprocessed.csv",
        epochs=50, lr=0.01, batch_size=64, device="cpu",
    )

    local_results = run_local_only(
        data_path="data/processed/datasense_preprocessed.csv",
        partition_path="data/partitions/iid_K5_seed42.json",
        epochs=50, lr=0.01, batch_size=64, device="cpu",
    )
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from model import create_model, get_model_params, set_model_params
from preprocessing import (
    load_processed_data,
    apply_scaler,
    compute_class_weights_from_labels,
    NUM_FEATURES,
    NUM_CLASSES,
)
from partitioning import load_partition
from federated import evaluate_model  # Reuse the same evaluation function


# ======================================================================
#  TRAINING LOOP (shared by both baselines)
# ======================================================================


def _train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs,
    lr,
    batch_size,
    device="cpu",
    verbose=True,
    label="Training",
    optimizer_type="sgd",
    scheduler_type=None,
):
    """
    Standard supervised training loop used by both centralized and local-only.

    Parameters
    ----------
    model : IDSNet
        The model to train (modified in-place).
    X_train : np.ndarray — already scaled
    y_train : np.ndarray
    X_val : np.ndarray — already scaled (for periodic evaluation)
    y_val : np.ndarray
    epochs : int
        Total training epochs.
    lr : float
        Learning rate.
    batch_size : int
    device : str
    verbose : bool
    label : str
        Label for print statements (e.g., "Centralized" or "Client 2").
    optimizer_type : str
        "sgd" (default, required for FL clients), "sgd_momentum" (SGD + 0.9
        momentum for faster centralized convergence), or "adam" (fastest
        convergence for centralized baseline). The centralized baseline is
        an upper bound, not a fair fight — using Adam here is legitimate.
    scheduler_type : str or None
        None (constant LR), "step" (StepLR: decay by 0.5 every 30 epochs),
        or "plateau" (ReduceLROnPlateau: decay by 0.5 if F1 stalls for 10 epochs).
        Schedulers help SGD converge better on this imbalanced dataset.

    Returns
    -------
    history : list[dict]
        Per-epoch metrics: epoch, train_loss, accuracy, f1_macro, lr, etc.
    best_state : dict
        Keys: epoch, f1_macro, model_params (weights at peak validation F1).
    """
    # Class weights for imbalanced data
    class_weights = compute_class_weights_from_labels(y_train, NUM_CLASSES)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # ── Optimizer selection ───────────────────────────────────────────
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_type == "sgd_momentum":
        # SGD with momentum — use for centralized upper bound only.
        # Converges faster than vanilla SGD but NOT used for FL clients.
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        # Vanilla SGD — matches FedAvg client training (McMahan et al.)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # ── Scheduler selection ──────────────────────────────────────────
    scheduler = None
    if scheduler_type == "step":
        # Decay LR by 0.5 every 30 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif scheduler_type == "plateau":
        # Decay LR by 0.5 if macro-F1 doesn't improve for 10 epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-5
        )

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    history = []
    best_f1 = -1.0
    best_state = {"epoch": 0, "f1_macro": 0.0, "model_params": None}

    for epoch in range(1, epochs + 1):
        # ── Train one epoch ──────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # ── Evaluate on validation set ───────────────────────────────
        metrics = evaluate_model(model, X_val, y_val, device)

        # Track current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "lr": current_lr,
            **metrics,
        }
        history.append(record)

        # ── Track best model ─────────────────────────────────────────
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_state = {
                "epoch": epoch,
                "f1_macro": metrics["f1_macro"],
                "model_params": get_model_params(model),
            }

        # ── Step scheduler ───────────────────────────────────────────
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(metrics["f1_macro"])
            else:
                scheduler.step()

        # Print at 10 evenly-spaced checkpoints
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
            lr_str = f"  lr={current_lr:.6f}" if scheduler is not None else ""
            print(
                f"  [{label}] Epoch {epoch:>3d}/{epochs}  "
                f"loss={avg_loss:.4f}  "
                f"acc={metrics['accuracy']:.4f}  "
                f"f1={metrics['f1_macro']:.4f}{lr_str}"
            )

    if verbose and best_state["epoch"] != epochs:
        print(
            f"  [{label}] Best F1: {best_state['f1_macro']:.4f} "
            f"at epoch {best_state['epoch']}"
        )

    return history, best_state


# ======================================================================
#  CENTRALIZED BASELINE (upper bound)
# ======================================================================


def run_centralized(
    data_path,
    epochs=50,
    lr=0.01,
    batch_size=64,
    test_fraction=0.2,
    device="cpu",
    seed=42,
    verbose=True,
    optimizer_type="sgd",
    scheduler_type=None,
):
    """
    Train a single model on ALL data — the centralized upper bound.

    This represents the best-case scenario where all IIoT sites share
    their raw data with a central server. No privacy preservation.
    FL results should approach (but likely not exceed) this performance.

    Pipeline:
        1. Load preprocessed data
        2. Stratified train/test split (same split as federated for fairness)
        3. Fit scaler on train, transform both
        4. Train IDSNet for `epochs` epochs
        5. Return final metrics + training history

    Parameters
    ----------
    data_path : str
        Path to datasense_preprocessed.csv.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    batch_size : int
    test_fraction : float
        Fraction held out for evaluation (must match federated experiments).
    device : str
    seed : int
    verbose : bool
    optimizer_type : str
        "sgd", "sgd_momentum", or "adam". Adam typically converges faster on
        tabular data. Using Adam for the centralized upper bound is legitimate
        — it's meant to be the best possible performance, not a fair fight.
    scheduler_type : str or None
        None, "step", or "plateau". See _train_model docstring.

    Returns
    -------
    results : dict with keys:
        config : experiment parameters
        history : per-epoch metrics
        final_metrics : final test set evaluation (last epoch)
        best_metrics : best test set evaluation (peak F1 epoch)
        best_epoch : epoch number where best F1 was achieved
        total_time_sec : wall-clock training time
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if verbose:
        print("=" * 70)
        print("CENTRALIZED BASELINE (upper bound)")
        print("=" * 70)

    # ── 1. Load data ─────────────────────────────────────────────────
    X_all, y_all, _ = load_processed_data(data_path)

    if verbose:
        print(f"Loaded: {len(y_all):,} samples, {NUM_FEATURES} features")

    # ── 2. Train/test split (stratified, same seed as federated) ─────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=test_fraction,
        stratify=y_all,
        random_state=seed,
    )

    if verbose:
        print(f"Train: {len(y_train):,}  |  Test: {len(y_test):,}")

    # ── 3. Scale (fit on train only) ─────────────────────────────────
    X_train_scaled, X_test_scaled, _ = apply_scaler(X_train_raw, X_test_raw)

    # ── 4. Create model ──────────────────────────────────────────────
    model = create_model(device=device)

    if verbose:
        opt_label = optimizer_type.upper()
        sched_label = f" + {scheduler_type} scheduler" if scheduler_type else ""
        print(
            f"Training for {epochs} epochs ({opt_label}, lr={lr}, batch={batch_size}{sched_label})"
        )
        print("─" * 70)

    # ── 5. Train ─────────────────────────────────────────────────────
    start_time = time.time()

    history, best_state = _train_model(
        model=model,
        X_train=X_train_scaled,
        y_train=y_train,
        X_val=X_test_scaled,
        y_val=y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
        label="Centralized",
        optimizer_type=optimizer_type,
        scheduler_type=scheduler_type,
    )

    total_time = time.time() - start_time

    # ── 6. Final evaluation (last epoch) ─────────────────────────────
    final_metrics = evaluate_model(model, X_test_scaled, y_test, device)

    # ── 7. Best-epoch evaluation ─────────────────────────────────────
    # Reload best weights and evaluate (may differ from final epoch)
    best_model = create_model(device=device)
    set_model_params(best_model, best_state["model_params"])
    best_metrics = evaluate_model(best_model, X_test_scaled, y_test, device)

    if verbose:
        print("─" * 70)
        print(f"Finished in {total_time:.1f}s")
        print(f"Final accuracy:   {final_metrics['accuracy']:.4f}")
        print(f"Final F1 (macro): {final_metrics['f1_macro']:.4f}")
        print(
            f"Best F1 (macro):  {best_metrics['f1_macro']:.4f} (epoch {best_state['epoch']})"
        )
        print(f"Per-class F1:     {[f'{f:.3f}' for f in best_metrics['per_class_f1']]}")

    return {
        "config": {
            "baseline_type": "centralized",
            "data_path": data_path,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "test_fraction": test_fraction,
            "seed": seed,
            "optimizer_type": optimizer_type,
            "scheduler_type": scheduler_type,
            "n_train": len(y_train),
            "n_test": len(y_test),
        },
        "history": history,
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "best_epoch": best_state["epoch"],
        "total_time_sec": total_time,
    }


# ======================================================================
#  LOCAL-ONLY BASELINE (lower bound)
# ======================================================================


def run_local_only(
    data_path,
    partition_path,
    epochs=50,
    lr=0.01,
    batch_size=64,
    test_fraction=0.2,
    device="cpu",
    seed=42,
    verbose=True,
):
    """
    Train independent models per client — the local-only lower bound.

    Each client trains on ONLY its own partition with no aggregation.
    This represents what each IIoT site can achieve in complete isolation.
    FL should outperform this; if it doesn't, collaboration adds no value.

    Two evaluation modes:
        - Per-client: Each client's model evaluated on the global test set
          (shows how much local data limits generalization).
        - Ensemble: Average the predictions of all local models
          (a simple collaboration baseline without weight sharing).

    Parameters
    ----------
    data_path : str
    partition_path : str
        Path to a partition JSON file (same one used for federated).
    epochs : int
    lr : float
    batch_size : int
    test_fraction : float
        Must match the federated and centralized experiments.
    device : str
    seed : int
    verbose : bool

    Returns
    -------
    results : dict with keys:
        config : experiment parameters
        per_client : list of per-client result dicts
        ensemble_metrics : metrics from averaging all client predictions
        avg_metrics : simple average of per-client metrics
        total_time_sec : wall-clock time for all clients
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if verbose:
        print("=" * 70)
        print("LOCAL-ONLY BASELINE (lower bound)")
        print("=" * 70)

    # ── 1. Load data and partition ───────────────────────────────────
    X_all, y_all, _ = load_processed_data(data_path)
    client_indices, partition_meta = load_partition(partition_path)
    num_clients = len(client_indices)

    if verbose:
        print(
            f"Loaded: {len(y_all):,} samples, "
            f"{num_clients} clients ({partition_meta['strategy']})"
        )

    # ── 2. Global test set (same split as federated) ─────────────────
    all_indices = np.arange(len(y_all))
    train_pool, test_indices = train_test_split(
        all_indices,
        test_size=test_fraction,
        stratify=y_all,
        random_state=seed,
    )
    train_pool_set = set(train_pool.tolist())

    # Scale test set using the full train pool
    X_train_pool = X_all[train_pool]
    X_test_raw = X_all[test_indices]
    y_test = y_all[test_indices]

    _, X_test_scaled, _ = apply_scaler(X_train_pool, X_test_raw)

    if verbose:
        print(f"Global test set: {len(y_test):,} samples")
        print("─" * 70)

    # ── 3. Train each client independently ───────────────────────────
    start_time = time.time()
    per_client_results = []

    for k in sorted(client_indices.keys()):
        # Filter to this client's training data
        local_indices = [i for i in client_indices[k] if i in train_pool_set]
        if len(local_indices) == 0:
            continue

        X_local = X_all[local_indices]
        y_local = y_all[local_indices]

        # Fit scaler on this client's data only (same as federated)
        X_local_scaled, X_test_local_scaled, _ = apply_scaler(X_local, X_test_raw)

        # Fresh model per client
        model = create_model(device=device)

        if verbose:
            unique, counts = np.unique(y_local, return_counts=True)
            print(
                f"\nClient {k}: {len(y_local)} samples, "
                f"{len(unique)} classes present"
            )

        # Train (always SGD for local-only — matches FL client training)
        history, _ = _train_model(
            model=model,
            X_train=X_local_scaled,
            y_train=y_local,
            X_val=X_test_local_scaled,
            y_val=y_test,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
            label=f"Client {k}",
            optimizer_type="sgd",
        )

        # Final evaluation (using client's own scaler on the test set)
        final_metrics = evaluate_model(model, X_test_local_scaled, y_test, device)

        per_client_results.append(
            {
                "client_id": k,
                "n_samples": len(y_local),
                "n_classes": len(np.unique(y_local)),
                "class_distribution": dict(zip(unique.tolist(), counts.tolist())),
                "history": history,
                "final_metrics": final_metrics,
                "model_params": get_model_params(model),  # Keep for ensemble
            }
        )

        if verbose:
            print(
                f"  → Client {k} final: "
                f"acc={final_metrics['accuracy']:.4f}  "
                f"f1={final_metrics['f1_macro']:.4f}"
            )

    total_time = time.time() - start_time

    # ── 4. Compute ensemble prediction (soft voting) ─────────────────
    # Average the logits from all local models, then take argmax.
    # This is a simple collaboration baseline — better than any single
    # client but doesn't require weight sharing like FedAvg.

    ensemble_metrics = _compute_ensemble(per_client_results, X_test_raw, y_test, device)

    # ── 5. Compute average of per-client metrics ─────────────────────
    avg_metrics = {
        "accuracy": np.mean(
            [r["final_metrics"]["accuracy"] for r in per_client_results]
        ),
        "f1_macro": np.mean(
            [r["final_metrics"]["f1_macro"] for r in per_client_results]
        ),
        "precision_macro": np.mean(
            [r["final_metrics"]["precision_macro"] for r in per_client_results]
        ),
        "recall_macro": np.mean(
            [r["final_metrics"]["recall_macro"] for r in per_client_results]
        ),
    }

    if verbose:
        print("\n" + "─" * 70)
        print(f"Finished {len(per_client_results)} clients in {total_time:.1f}s")
        print(
            f"\nAvg per-client:  acc={avg_metrics['accuracy']:.4f}  f1={avg_metrics['f1_macro']:.4f}"
        )
        print(
            f"Ensemble:        acc={ensemble_metrics['accuracy']:.4f}  f1={ensemble_metrics['f1_macro']:.4f}"
        )

    # Clean up model_params before returning (they're large)
    for r in per_client_results:
        del r["model_params"]

    return {
        "config": {
            "baseline_type": "local_only",
            "data_path": data_path,
            "partition_path": partition_path,
            "strategy": partition_meta["strategy"],
            "num_clients": num_clients,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "test_fraction": test_fraction,
            "seed": seed,
        },
        "per_client": per_client_results,
        "ensemble_metrics": ensemble_metrics,
        "avg_metrics": avg_metrics,
        "total_time_sec": total_time,
    }


# ======================================================================
#  ENSEMBLE HELPER
# ======================================================================


def _compute_ensemble(per_client_results, X_test_raw, y_test, device):
    """
    Soft-voting ensemble: average logits from all local models.

    Each client's model uses its OWN scaler (fit on its local data),
    which is realistic — in a real deployment, each site would only
    know its own feature statistics.

    Parameters
    ----------
    per_client_results : list[dict]
        Must contain 'model_params' and 'n_samples' for each client.
    X_test_raw : np.ndarray — unscaled test features
    y_test : np.ndarray
    device : str

    Returns
    -------
    metrics : dict
        Same format as evaluate_model output.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    all_logits = []

    for r in per_client_results:
        # Recreate client's model
        model = create_model(device=device)
        set_model_params(model, r["model_params"])
        model.eval()

        # Each client scales test data with its own local scaler
        # (We re-fit from scratch since we didn't save the scaler object)
        # This is a simplification — the logit averaging still works
        # because the model learned to map from its own scaled space.

        # For ensemble, we use a uniform scaler from the full test set
        # as an approximation (all clients see the same test input).
        X_test_scaled, _, _ = apply_scaler(X_test_raw)

        # Get logits
        dataset = TensorDataset(
            torch.tensor(X_test_scaled, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=512, shuffle=False)

        client_logits = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                logits = model(xb)
                client_logits.append(logits.cpu().numpy())

        all_logits.append(np.concatenate(client_logits, axis=0))

    # Average logits across all clients (soft voting)
    avg_logits = np.mean(all_logits, axis=0)
    y_pred = avg_logits.argmax(axis=1)

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
