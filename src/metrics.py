"""
metrics.py — Metric computation, aggregation, and analysis for FL-IDS.

This module provides utilities that consume the result dicts already
produced by federated.py and baselines.py. It does NOT run any training
itself — it's a post-processing and analysis layer.

Components:
    EVALUATION
        compute_confusion_matrix  — confusion matrix from model predictions
        compute_roc_auc           — one-vs-rest ROC-AUC for multiclass

    MULTI-SEED AGGREGATION
        aggregate_seeds           — mean ± std across random seeds

    COMMUNICATION COST ANALYSIS
        comm_cost_summary         — total bytes, per-round cost, cost-per-F1-point
        rounds_to_target          — how many rounds to reach X% of centralized F1

    CONVERGENCE ANALYSIS
        convergence_summary       — plateau detection, speed, final vs best

    COMPARISON & STATISTICAL TESTS
        build_comparison_table    — side-by-side table of multiple experiments
        pairwise_significance     — paired t-test between two experiment runs

    PERSISTENCE
        save_results / load_results — JSON serialization for reproducibility

Usage:
    from metrics import (
        compute_confusion_matrix, compute_roc_auc,
        aggregate_seeds, comm_cost_summary, rounds_to_target,
        build_comparison_table, pairwise_significance,
        save_results, load_results,
    )
"""

import json
import os
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
from scipy import stats

from model import create_model, set_model_params
from preprocessing import NUM_CLASSES


# ======================================================================
#  EVALUATION — Extended metrics beyond what evaluate_model provides
# ======================================================================


def compute_confusion_matrix(model, X_test, y_test, device="cpu", batch_size=512):
    """
    Compute the confusion matrix for a trained model.

    Parameters
    ----------
    model : IDSNet
    X_test : np.ndarray — already scaled
    y_test : np.ndarray
    device : str
    batch_size : int

    Returns
    -------
    cm : np.ndarray, shape (NUM_CLASSES, NUM_CLASSES)
        Rows = true labels, columns = predicted labels.
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
    return confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))


def compute_roc_auc(model, X_test, y_test, device="cpu", batch_size=512):
    """
    Compute one-vs-rest ROC-AUC for multiclass classification.

    This is listed as a required metric in the proposal. It measures
    how well the model discriminates each class from all others.

    Parameters
    ----------
    model : IDSNet
    X_test : np.ndarray — already scaled
    y_test : np.ndarray
    device : str
    batch_size : int

    Returns
    -------
    auc_macro : float
        Macro-averaged ROC-AUC across all classes.
    auc_per_class : np.ndarray, shape (NUM_CLASSES,)
        Per-class ROC-AUC scores.
    """
    model.eval()
    all_probs = []

    dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    y_prob = np.concatenate(all_probs, axis=0)

    # One-hot encode true labels
    y_onehot = np.zeros((len(y_test), NUM_CLASSES), dtype=np.int32)
    y_onehot[np.arange(len(y_test)), y_test] = 1

    # Per-class AUC (handle classes with no positive samples gracefully)
    auc_per_class = np.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        if y_onehot[:, c].sum() > 0 and y_onehot[:, c].sum() < len(y_test):
            auc_per_class[c] = roc_auc_score(y_onehot[:, c], y_prob[:, c])
        else:
            auc_per_class[c] = float("nan")

    # Macro average (excluding NaN classes)
    valid_aucs = auc_per_class[~np.isnan(auc_per_class)]
    auc_macro = float(np.mean(valid_aucs)) if len(valid_aucs) > 0 else 0.0

    return auc_macro, auc_per_class


def full_classification_report(
    model, X_test, y_test, label_names=None, device="cpu", batch_size=512
):
    """
    Generate a full sklearn classification report as a dict.

    Useful for thesis tables — includes per-class precision, recall, F1,
    plus macro/weighted averages and support counts.

    Parameters
    ----------
    model : IDSNet
    X_test, y_test : np.ndarray
    label_names : list[str], optional
    device, batch_size : str, int

    Returns
    -------
    report : dict
        Nested dict from sklearn.metrics.classification_report.
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
            preds = model(xb).argmax(dim=1).cpu().numpy()
            all_preds.append(preds)

    y_pred = np.concatenate(all_preds)
    target_names = label_names or [str(i) for i in range(NUM_CLASSES)]

    return classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
        output_dict=True,
    )


# ======================================================================
#  MULTI-SEED AGGREGATION
# ======================================================================


def aggregate_seeds(results_list):
    """
    Aggregate results from multiple random seeds into mean ± std.

    The proposal requires 3 seeds per configuration. This function takes
    a list of result dicts (from run_federated or run_centralized) and
    computes statistical summaries.

    Parameters
    ----------
    results_list : list[dict]
        Each dict is the output of run_federated() or run_centralized().
        Must all have the same config (except seed).

    Returns
    -------
    summary : dict with keys:
        n_seeds : int
        final_metrics : dict of {metric_name: {"mean": float, "std": float}}
        per_round : list of per-round aggregated metrics (for FL results)
        configs : list of individual configs
    """
    n = len(results_list)
    if n == 0:
        return {}

    # ── Final metrics aggregation ────────────────────────────────────
    # Collect all scalar final metrics
    metric_keys = [
        "accuracy",
        "f1_macro",
        "precision_macro",
        "recall_macro",
    ]
    final_agg = {}
    for key in metric_keys:
        values = []
        for res in results_list:
            # Handle both baselines (final_metrics) and federated (final_metrics)
            fm = res.get("final_metrics") or res.get("best_metrics", {})
            if key in fm:
                values.append(fm[key])
        if values:
            final_agg[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }

    # ── Per-class F1 aggregation ─────────────────────────────────────
    per_class_f1s = []
    for res in results_list:
        fm = res.get("final_metrics") or res.get("best_metrics", {})
        if "per_class_f1" in fm:
            per_class_f1s.append(fm["per_class_f1"])

    if per_class_f1s:
        arr = np.array(per_class_f1s)
        final_agg["per_class_f1"] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
        }

    # ── Per-round aggregation (FL only) ──────────────────────────────
    per_round_agg = []
    if "history" in results_list[0] and len(results_list[0]["history"]) > 0:
        # Check if history entries have "round" key (FL) vs "epoch" key (baseline)
        first_entry = results_list[0]["history"][0]
        round_key = "round" if "round" in first_entry else "epoch"

        n_rounds = len(results_list[0]["history"])
        for r_idx in range(n_rounds):
            round_data = {}
            round_data[round_key] = results_list[0]["history"][r_idx][round_key]

            for key in metric_keys:
                vals = []
                for res in results_list:
                    if r_idx < len(res["history"]) and key in res["history"][r_idx]:
                        vals.append(res["history"][r_idx][key])
                if vals:
                    round_data[key + "_mean"] = float(np.mean(vals))
                    round_data[key + "_std"] = float(np.std(vals))

            per_round_agg.append(round_data)

    # ── Communication cost aggregation (FL only) ─────────────────────
    comm_agg = {}
    total_bytes_list = [r.get("total_bytes", 0) for r in results_list]
    if any(b > 0 for b in total_bytes_list):
        comm_agg = {
            "total_bytes_mean": float(np.mean(total_bytes_list)),
            "total_bytes_std": float(np.std(total_bytes_list)),
        }

    return {
        "n_seeds": n,
        "final_metrics": final_agg,
        "per_round": per_round_agg,
        "communication": comm_agg,
        "configs": [r.get("config", {}) for r in results_list],
    }


# ======================================================================
#  COMMUNICATION COST ANALYSIS
# ======================================================================


def comm_cost_summary(results):
    """
    Summarize communication costs from a federated experiment.

    Parameters
    ----------
    results : dict
        Output of run_federated(). Must have 'history' with per-round
        'bytes_exchanged' entries.

    Returns
    -------
    summary : dict with keys:
        model_size_bytes : int — size of one model update
        total_bytes : int — total bytes exchanged across all rounds
        total_mb : float — total in megabytes
        bytes_per_round : float — average bytes per round
        total_rounds : int
        bytes_per_f1_point : float — communication cost per F1 point
            (lower is better — how efficiently does communication
            translate into detection capability)
        cumulative_bytes : list[int] — running total per round
    """
    history = results.get("history", [])
    if not history:
        return {}

    bytes_per_round = [r.get("bytes_exchanged", 0) for r in history]
    cumulative = np.cumsum(bytes_per_round).tolist()
    total = sum(bytes_per_round)
    final_f1 = results.get("final_metrics", {}).get("f1_macro", 0)

    return {
        "total_bytes": total,
        "total_mb": total / (1024 * 1024),
        "bytes_per_round": float(np.mean(bytes_per_round)),
        "total_rounds": len(history),
        "bytes_per_f1_point": total / max(final_f1, 1e-6),
        "cumulative_bytes": cumulative,
        "final_f1": final_f1,
    }


def rounds_to_target(
    results, target_f1=None, centralized_f1=None, target_fraction=0.95
):
    """
    Determine how many communication rounds are needed to reach a target.

    This is a key metric from the proposal — "rounds-to-target accuracy"
    measures communication efficiency.

    Parameters
    ----------
    results : dict
        Output of run_federated().
    target_f1 : float, optional
        Absolute F1 target. If None, computed from centralized_f1.
    centralized_f1 : float, optional
        Centralized baseline F1. Used with target_fraction.
    target_fraction : float
        Fraction of centralized F1 to use as target (default: 0.95 = 95%).

    Returns
    -------
    analysis : dict with keys:
        target_f1 : float — the target used
        rounds_needed : int or None — first round that reached target
        reached : bool
        final_f1 : float
        gap_from_target : float — how far short (0 if reached)
        f1_at_each_round : list[float]
    """
    history = results.get("history", [])
    if not history:
        return {}

    # Determine target
    if target_f1 is None:
        if centralized_f1 is not None:
            target_f1 = centralized_f1 * target_fraction
        else:
            target_f1 = 0.90  # Fallback default

    f1s = [r.get("f1_macro", 0) for r in history]
    rounds_needed = None

    for i, f1 in enumerate(f1s):
        if f1 >= target_f1:
            rounds_needed = i + 1  # 1-indexed
            break

    final_f1 = f1s[-1] if f1s else 0.0

    return {
        "target_f1": target_f1,
        "rounds_needed": rounds_needed,
        "reached": rounds_needed is not None,
        "final_f1": final_f1,
        "gap_from_target": max(0, target_f1 - max(f1s)) if f1s else target_f1,
        "f1_at_each_round": f1s,
    }


# ======================================================================
#  CONVERGENCE ANALYSIS
# ======================================================================


def convergence_summary(results):
    """
    Analyze convergence behavior of an FL or baseline experiment.

    Parameters
    ----------
    results : dict
        Output of run_federated() or run_centralized().

    Returns
    -------
    summary : dict with keys:
        n_rounds_or_epochs : int
        final_f1 : float
        best_f1 : float
        best_round : int
        f1_improvement_last_10pct : float
            How much F1 improved in the last 10% of rounds —
            if this is small, training has converged.
        converged : bool
            True if improvement in last 10% is < 0.005
    """
    history = results.get("history", [])
    if not history:
        return {}

    f1s = [r.get("f1_macro", 0) for r in history]
    n = len(f1s)
    best_idx = int(np.argmax(f1s))

    # Check improvement in last 10% of training
    cutoff = max(1, int(n * 0.9))
    f1_at_90pct = f1s[cutoff - 1] if cutoff <= n else f1s[-1]
    f1_final = f1s[-1]
    improvement_last_10 = f1_final - f1_at_90pct

    return {
        "n_rounds_or_epochs": n,
        "final_f1": f1_final,
        "best_f1": f1s[best_idx],
        "best_round": best_idx + 1,
        "f1_improvement_last_10pct": improvement_last_10,
        "converged": abs(improvement_last_10) < 0.005,
    }


# ======================================================================
#  COMPARISON & STATISTICAL TESTS
# ======================================================================


def build_comparison_table(experiments, label_names=None):
    """
    Build a side-by-side comparison table from multiple experiments.

    Handles both single-seed results and aggregated multi-seed results.

    Parameters
    ----------
    experiments : dict
        {name: results_dict} where results_dict is from run_federated(),
        run_centralized(), run_local_only(), or aggregate_seeds().
    label_names : list[str], optional
        Class names for per-class F1 rows.

    Returns
    -------
    table : list[dict]
        Each dict is one row: {"method": str, "accuracy": str, "f1_macro": str, ...}
        Formatted as "value" or "mean ± std" strings.
    """
    rows = []

    for name, res in experiments.items():
        row = {"method": name}

        # Detect if this is aggregated (has nested mean/std) or single-seed
        fm = res.get("final_metrics", {})

        if isinstance(fm.get("f1_macro"), dict):
            # Aggregated multi-seed result
            for key in ["accuracy", "f1_macro", "precision_macro", "recall_macro"]:
                if key in fm:
                    m = fm[key]["mean"]
                    s = fm[key]["std"]
                    row[key] = f"{m:.4f} ± {s:.4f}"
        else:
            # Single-seed result
            for key in ["accuracy", "f1_macro", "precision_macro", "recall_macro"]:
                if key in fm:
                    row[key] = f"{fm[key]:.4f}"

        # Communication cost (FL only)
        total_bytes = res.get("total_bytes", 0)
        if total_bytes:
            row["comm_cost_mb"] = f"{total_bytes / (1024*1024):.2f}"

        # Time
        total_time = res.get("total_time_sec", 0)
        if total_time:
            row["time_sec"] = f"{total_time:.1f}"

        rows.append(row)

    return rows


def print_comparison_table(experiments, label_names=None):
    """
    Pretty-print a comparison table to the console.

    Parameters
    ----------
    experiments : dict
        {name: results_dict}
    label_names : list[str], optional
    """
    rows = build_comparison_table(experiments, label_names)
    if not rows:
        print("No experiments to compare.")
        return

    # Determine column widths
    headers = ["method", "accuracy", "f1_macro", "precision_macro", "recall_macro"]
    # Add optional columns if present
    if any("comm_cost_mb" in r for r in rows):
        headers.append("comm_cost_mb")
    if any("time_sec" in r for r in rows):
        headers.append("time_sec")

    col_names = {
        "method": "Method",
        "accuracy": "Accuracy",
        "f1_macro": "F1 (macro)",
        "precision_macro": "Precision",
        "recall_macro": "Recall",
        "comm_cost_mb": "Comm (MB)",
        "time_sec": "Time (s)",
    }

    widths = {}
    for h in headers:
        widths[h] = max(
            len(col_names.get(h, h)),
            max((len(r.get(h, "")) for r in rows), default=0),
        )

    # Print header
    header_str = "  ".join(col_names.get(h, h).rjust(widths[h]) for h in headers)
    print(header_str)
    print("-" * len(header_str))

    # Print rows
    for row in rows:
        row_str = "  ".join(row.get(h, "—").rjust(widths[h]) for h in headers)
        print(row_str)


def pairwise_significance(results_a_list, results_b_list, metric="f1_macro"):
    """
    Paired t-test between two experiment configurations.

    Each configuration should have been run with the same set of seeds.
    The test determines whether the performance difference is statistically
    significant (p < 0.05).

    Parameters
    ----------
    results_a_list : list[dict]
        Results from config A across multiple seeds.
    results_b_list : list[dict]
        Results from config B across multiple seeds.
    metric : str
        Which metric to compare (default: "f1_macro").

    Returns
    -------
    test_result : dict with keys:
        metric : str
        mean_a, mean_b : float
        diff : float — mean_a - mean_b
        t_statistic : float
        p_value : float
        significant : bool — p < 0.05
        n_seeds : int
    """
    values_a = [r["final_metrics"][metric] for r in results_a_list]
    values_b = [r["final_metrics"][metric] for r in results_b_list]

    if len(values_a) < 2 or len(values_b) < 2:
        return {
            "metric": metric,
            "mean_a": float(np.mean(values_a)),
            "mean_b": float(np.mean(values_b)),
            "diff": float(np.mean(values_a) - np.mean(values_b)),
            "t_statistic": float("nan"),
            "p_value": float("nan"),
            "significant": False,
            "n_seeds": min(len(values_a), len(values_b)),
            "note": "Need at least 2 seeds per config for t-test",
        }

    t_stat, p_val = stats.ttest_rel(values_a, values_b)

    return {
        "metric": metric,
        "mean_a": float(np.mean(values_a)),
        "mean_b": float(np.mean(values_b)),
        "diff": float(np.mean(values_a) - np.mean(values_b)),
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "significant": p_val < 0.05,
        "n_seeds": len(values_a),
    }


# ======================================================================
#  PERSISTENCE — Save/Load results for reproducibility
# ======================================================================


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types when serializing to JSON."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_results(results, filepath, overwrite=False):
    """
    Save experiment results to a JSON file.

    Parameters
    ----------
    results : dict
        Output of run_federated(), run_centralized(), or run_local_only().
    filepath : str or Path
        Output path (should end in .json).
    overwrite : bool
        If False and file exists, raise an error.

    Returns
    -------
    filepath : str
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.exists() and not overwrite:
        raise FileExistsError(
            f"{filepath} already exists. Use overwrite=True to replace."
        )

    # Add metadata
    results_with_meta = {
        "saved_at": datetime.now().isoformat(),
        **results,
    }

    # Remove model_params if present (large, not JSON-safe)
    _strip_model_params(results_with_meta)

    with open(filepath, "w") as f:
        json.dump(results_with_meta, f, indent=2, cls=_NumpyEncoder)

    return str(filepath)


def load_results(filepath):
    """
    Load experiment results from a JSON file.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    results : dict
    """
    with open(filepath, "r") as f:
        return json.load(f)


def _strip_model_params(d):
    """Recursively remove 'model_params' keys (numpy arrays, not serializable)."""
    if isinstance(d, dict):
        d.pop("model_params", None)
        for v in d.values():
            _strip_model_params(v)
    elif isinstance(d, list):
        for item in d:
            _strip_model_params(item)


# ======================================================================
#  CONVENIENCE: Generate thesis-ready summary
# ======================================================================


def experiment_summary(
    centralized_results,
    local_results,
    federated_results_dict,
    label_names=None,
):
    """
    Generate a comprehensive summary for the thesis results chapter.

    Parameters
    ----------
    centralized_results : dict
        Output of run_centralized() (single seed or best config).
    local_results : dict
        Output of run_local_only().
    federated_results_dict : dict
        {experiment_name: results_dict} for each FL configuration.
    label_names : list[str], optional

    Returns
    -------
    summary : dict with keys:
        comparison_table : list[dict] — formatted for thesis tables
        centralized_f1 : float
        local_avg_f1 : float
        gap : float — centralized - local average
        fl_results : dict — per-experiment analysis
    """
    cent_f1 = centralized_results.get(
        "best_metrics", centralized_results.get("final_metrics", {})
    ).get("f1_macro", 0)

    local_avg_f1 = local_results.get("avg_metrics", {}).get("f1_macro", 0)
    gap = cent_f1 - local_avg_f1

    # Build experiment dict for comparison table
    all_experiments = {"Centralized": centralized_results}

    # Add local results as a pseudo-experiment
    local_as_experiment = {
        "final_metrics": local_results.get("avg_metrics", {}),
        "total_time_sec": local_results.get("total_time_sec", 0),
    }
    all_experiments["Local-Only (avg)"] = local_as_experiment

    ensemble_as_experiment = {
        "final_metrics": local_results.get("ensemble_metrics", {}),
    }
    all_experiments["Ensemble"] = ensemble_as_experiment

    # Add FL experiments
    for name, res in federated_results_dict.items():
        all_experiments[name] = res

    # Analyze each FL experiment
    fl_analysis = {}
    for name, res in federated_results_dict.items():
        fl_analysis[name] = {
            "convergence": convergence_summary(res),
            "comm_cost": comm_cost_summary(res),
            "rounds_to_95pct": rounds_to_target(
                res, centralized_f1=cent_f1, target_fraction=0.95
            ),
            "rounds_to_90pct": rounds_to_target(
                res, centralized_f1=cent_f1, target_fraction=0.90
            ),
            "final_f1": res.get("final_metrics", {}).get("f1_macro", 0),
            "gap_closed": _gap_closed(
                cent_f1, local_avg_f1, res.get("final_metrics", {}).get("f1_macro", 0)
            ),
        }

    return {
        "comparison_table": build_comparison_table(all_experiments, label_names),
        "centralized_f1": cent_f1,
        "local_avg_f1": local_avg_f1,
        "gap": gap,
        "fl_analysis": fl_analysis,
    }


def _gap_closed(cent_f1, local_f1, fl_f1):
    """
    What fraction of the centralized-local gap did FL close?

    Returns a value between 0 (FL = local) and 1 (FL = centralized).
    Values > 1 mean FL exceeded centralized (unlikely but possible with noise).
    """
    gap = cent_f1 - local_f1
    if gap <= 0:
        return 1.0
    return (fl_f1 - local_f1) / gap
