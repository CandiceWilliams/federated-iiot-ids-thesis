"""
preprocessing.py — Reusable preprocessing utilities for FL-IDS pipeline.

This module provides functions used by both the centralized baseline and
the federated training loop to ensure consistent data handling.

Usage:
    from preprocessing import load_processed_data, get_class_weights, apply_scaler
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler


# ── 17 selected features from DataSense Table 7 ──────────────────────
SELECTED_FEATURES = [
    "log_messages_count",  # Log Data Rate
    "log_data-ranges_avg",  # Log Data Stats
    "log_data-types_count",  # Log Data Stats
    "network_fragmented-packets",  # Fragmentation
    "network_interval-packets",  # Packet Traffic Rate
    "network_packets_all_count",  # Packet Traffic Rate
    "network_ips_dst_count",  # Address Diversity
    "network_ips_all_count",  # Address Diversity
    "network_macs_src_count",  # Address Diversity
    "network_packet-size_std_deviation",  # Size Length
    "network_ports_all_count",  # Network Multiplexing
    "network_protocols_all_count",  # Network Multiplexing
    "network_time-delta_avg",  # Timing Control
    "network_ttl_avg",  # Timing Control
    "network_window-size_avg",  # Timing Control
    "network_ip-flags_max",  # Header Flags
    "network_tcp-flags-psh_count",  # Header Flags
]

NUM_FEATURES = len(SELECTED_FEATURES)  # 17
NUM_CLASSES = 8


def load_processed_data(data_path):
    """
    Load the preprocessed CSV and return features, labels, and device names.

    Parameters
    ----------
    data_path : str
        Path to datasense_preprocessed.csv

    Returns
    -------
    X : np.ndarray, shape (n_samples, 17)
        Feature matrix (log1p already applied, NOT scaled)
    y : np.ndarray, shape (n_samples,)
        Integer-encoded labels (0–7)
    device_names : np.ndarray, shape (n_samples,)
        Device name strings for FL partitioning
    """
    df = pd.read_csv(data_path)

    X = df[SELECTED_FEATURES].values.astype(np.float32)
    y = df["attack_category"].values.astype(np.int64)
    device_names = df["device_name"].values

    return X, y, device_names


def get_class_weights(label_config_path):
    """
    Load precomputed inverse-frequency class weights.

    Parameters
    ----------
    label_config_path : str
        Path to label_config.json

    Returns
    -------
    weights : np.ndarray, shape (8,)
        Class weights indexed by class integer label
    label_mapping : dict
        {int: str} mapping of label index to class name
    """
    with open(label_config_path, "r") as f:
        config = json.load(f)

    weights = np.array(
        [config["class_weights"][str(i)] for i in range(config["n_classes"])],
        dtype=np.float32,
    )
    label_mapping = {int(k): v for k, v in config["label_mapping"].items()}

    return weights, label_mapping


def compute_class_weights_from_labels(y, n_classes=NUM_CLASSES):
    """
    Compute inverse-frequency class weights directly from a label array.
    Useful for per-client weight computation in FL.

    Parameters
    ----------
    y : np.ndarray
        Integer-encoded labels
    n_classes : int
        Total number of classes (default 8)

    Returns
    -------
    weights : np.ndarray, shape (n_classes,)
    """
    n_samples = len(y)
    class_counts = np.bincount(y, minlength=n_classes)
    # Avoid division by zero for classes absent from this client
    class_counts = np.maximum(class_counts, 1)
    weights = n_samples / (n_classes * class_counts)
    return weights.astype(np.float32)


def apply_scaler(X_train, X_test=None):
    """
    Fit StandardScaler on X_train, transform both train and test.

    In FL: call this per-client on the client's local train split.
    In centralized: call once on the global train split.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, 17)
    X_test : np.ndarray, shape (n_test, 17), optional

    Returns
    -------
    X_train_scaled : np.ndarray
    X_test_scaled : np.ndarray or None
    scaler : StandardScaler (fitted)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = None
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
