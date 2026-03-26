"""
partitioning.py — Federated partitioning strategies for FL-IDS.

Implements IID and Non-IID (label skew) partitioning strategies.
Saves/loads partition indices as JSON for reproducibility.

Usage:
    from partitioning import (
        partition_iid, partition_noniid_label_skew,
        save_partition, load_partition
    )
"""

import numpy as np
import json
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════
#  PARTITIONING STRATEGIES
# ══════════════════════════════════════════════════════════════════════


def partition_iid(labels, num_clients, seed=42):
    """
    IID (stratified random) partitioning.

    Each client receives an equal share of every class, so all clients
    have approximately the same class distribution as the global dataset.
    This is the "easy" FL baseline — local gradients are compatible.

    Parameters
    ----------
    labels : np.ndarray, shape (n_samples,)
        Integer-encoded class labels (0–7).
    num_clients : int
        Number of FL clients (K).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    client_indices : dict[int, list[int]]
        Mapping of client_id → list of row indices.
    """
    rng = np.random.default_rng(seed)

    # Group all sample indices by their class label
    indices_by_class = {}
    for idx, label in enumerate(labels):
        indices_by_class.setdefault(int(label), []).append(idx)

    # Deal from each class evenly across clients
    client_indices = {k: [] for k in range(num_clients)}
    for cls in sorted(indices_by_class.keys()):
        idxs = np.array(indices_by_class[cls])
        rng.shuffle(idxs)
        splits = np.array_split(idxs, num_clients)
        for k in range(num_clients):
            client_indices[k].extend(splits[k].tolist())

    return client_indices


def partition_noniid_label_skew(labels, num_clients, dominant_fraction=0.7, seed=42):
    """
    Non-IID partitioning via label skew.

    Each client is assigned a "dominant" attack category. That client
    receives `dominant_fraction` of all samples from its dominant class,
    while the remaining samples are distributed among other clients.

    This simulates realistic IIoT deployments where different industrial
    sites see different attack profiles:
      - Manufacturing plant → predominantly DDoS + DoS
      - Energy grid → predominantly Recon + MITM
      - Smart building → predominantly BruteForce + Malware

    Parameters
    ----------
    labels : np.ndarray, shape (n_samples,)
        Integer-encoded class labels (0–7).
    num_clients : int
        Number of FL clients (K). If K > num_classes, dominant classes
        cycle (e.g., client 8 gets the same dominant as client 0).
    dominant_fraction : float
        Fraction of each class's samples given to its dominant client(s).
        0.7 = moderate skew, 0.9 = severe skew.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    client_indices : dict[int, list[int]]
        Mapping of client_id → list of row indices.
    """
    rng = np.random.default_rng(seed)

    unique_classes = sorted(set(int(l) for l in labels))

    # Group all sample indices by class
    indices_by_class = {}
    for idx, label in enumerate(labels):
        indices_by_class.setdefault(int(label), []).append(idx)

    # Assign a dominant class to each client (cycling if K > num_classes)
    dominant_classes = [
        unique_classes[k % len(unique_classes)] for k in range(num_clients)
    ]

    client_indices = {k: [] for k in range(num_clients)}

    for cls in sorted(indices_by_class.keys()):
        idxs = np.array(indices_by_class[cls])
        rng.shuffle(idxs)

        # Identify which clients have this class as dominant vs. non-dominant
        dominant_clients = [k for k in range(num_clients) if dominant_classes[k] == cls]
        other_clients = [k for k in range(num_clients) if dominant_classes[k] != cls]

        # Split: dominant_fraction goes to dominant clients, rest to others.
        # If no client claims this class as dominant (happens when K < num_classes),
        # all samples go to other_clients — nothing is discarded.
        if dominant_clients:
            split_point = int(len(idxs) * dominant_fraction)
            dominant_pool = idxs[:split_point]
            other_pool = idxs[split_point:]

            for chunk, k in zip(
                np.array_split(dominant_pool, len(dominant_clients)), dominant_clients
            ):
                client_indices[k].extend(chunk.tolist())
        else:
            other_pool = idxs  # no dominant client → everything goes to others

        # Distribute remainder among other clients
        if other_clients:
            for chunk, k in zip(
                np.array_split(other_pool, len(other_clients)), other_clients
            ):
                client_indices[k].extend(chunk.tolist())

    return client_indices


# ══════════════════════════════════════════════════════════════════════
#  SAVE / LOAD UTILITIES
# ══════════════════════════════════════════════════════════════════════


def save_partition(client_indices, metadata, filepath):
    """
    Save a partition to disk as human-readable JSON.

    The file stores:
    - metadata: strategy name, seed, num_clients, parameters
    - partitions: client_id → list of integer row indices

    Parameters
    ----------
    client_indices : dict[int, list[int]]
    metadata : dict
        Must include at least: strategy, num_clients, seed, total_samples.
    filepath : str or Path
    """
    output = {
        "metadata": metadata,
        "partitions": {str(k): v for k, v in client_indices.items()},
    }
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(output, f)

    total = sum(len(v) for v in client_indices.values())
    print(
        f"  Saved: {filepath.name}  "
        f"({len(client_indices)} clients, {total:,} total samples)"
    )


def load_partition(filepath):
    """
    Load a saved partition.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    client_indices : dict[int, list[int]]
    metadata : dict
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    client_indices = {int(k): v for k, v in data["partitions"].items()}
    return client_indices, data["metadata"]


# ══════════════════════════════════════════════════════════════════════
#  DIAGNOSTIC UTILITIES
# ══════════════════════════════════════════════════════════════════════


def summarize_partition(client_indices, labels, label_names=None):
    """
    Print a summary table showing per-client sample counts and class
    distributions. Useful for verifying partition correctness.

    Parameters
    ----------
    client_indices : dict[int, list[int]]
    labels : np.ndarray
        Full label array (same one passed to the partitioning function).
    label_names : dict[int, str], optional
        Mapping of label int → class name for readable output.
    """
    n_classes = len(set(int(l) for l in labels))

    print(f"\n{'Client':<10} {'Samples':>8}  ", end="")
    for c in range(n_classes):
        name = label_names.get(c, str(c)) if label_names else str(c)
        print(f"{name[:8]:>9}", end="")
    print()
    print("─" * (20 + 9 * n_classes))

    for k in sorted(client_indices.keys()):
        idxs = client_indices[k]
        client_labels = labels[idxs]
        counts = np.bincount(client_labels, minlength=n_classes)
        total = len(idxs)

        print(f"  {k:<8} {total:>8}  ", end="")
        for c in range(n_classes):
            pct = counts[c] / total * 100 if total > 0 else 0
            print(f"{pct:>8.1f}%", end="")
        print()

    total_all = sum(len(v) for v in client_indices.values())
    print("─" * (20 + 9 * n_classes))
    print(f"  {'TOTAL':<8} {total_all:>8}")
