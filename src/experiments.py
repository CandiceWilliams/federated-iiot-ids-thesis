"""
experiments.py — Experiment runner for the FL-IDS evaluation.

Orchestrates the full experiment grid from the thesis proposal, organized
into focused experiment groups that each answer one research question.

Experiment Groups (mapped to proposal objectives):
    EXP1 — Baselines:       Centralized + local-only (K=5, K=10)
    EXP2 — IID FL:          Does FedAvg work? K=5, E=5, R=50
    EXP3 — Local epochs:    Effect of E ∈ {1, 5, 10}
    EXP4 — Comm rounds:     Effect of R ∈ {10, 25, 50, 100}
    EXP5 — Non-IID impact:  IID vs skew=0.7 vs skew=0.9
    EXP6 — Scalability:     K=5 vs K=10
    EXP7 — Participation:   Fraction ∈ {0.3, 0.5, 0.7, 1.0}

Design principles:
    - Each group varies ONE factor while holding others at defaults
    - 3 seeds per config for mean ± std
    - Results saved as JSON — resume if interrupted
    - Partitions auto-generated if missing

Usage:
    # Run everything:
    python experiments.py --data_path ../data/processed/datasense_preprocessed.csv

    # Run a single group:
    python experiments.py --data_path ... --group EXP3

    # Dry run (show what would be executed):
    python experiments.py --data_path ... --dry_run
"""

import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime

import numpy as np

from preprocessing import load_processed_data, NUM_CLASSES
from partitioning import (
    partition_iid,
    partition_noniid_label_skew,
    save_partition,
    load_partition,
)
from federated import run_federated
from baselines import run_centralized, run_local_only
from metrics import save_results, load_results, aggregate_seeds, print_comparison_table


# ======================================================================
#  DEFAULTS — Hold these constant unless the experiment varies them
# ======================================================================

DEFAULTS = {
    "num_rounds": 50,  # R
    "local_epochs": 5,  # E
    "lr": 0.01,  # SGD learning rate (for FL)
    "batch_size": 64,
    "participation_fraction": 1.0,
    "test_fraction": 0.2,
    "num_clients": 5,  # K
}

SEEDS = [42, 123, 456]  # 3 seeds per config as specified in proposal


# ======================================================================
#  PARTITION GENERATION
# ======================================================================


def ensure_partitions(data_path, partition_dir):
    """
    Generate all partition files needed for the experiments.

    Creates them only if they don't already exist (idempotent).
    Each partition file is named: {strategy}_K{K}_seed{seed}.json

    Returns
    -------
    partition_map : dict
        {(strategy, K, seed): filepath} for all partitions.
    """
    partition_dir = Path(partition_dir)
    partition_dir.mkdir(parents=True, exist_ok=True)

    _, y_all, _ = load_processed_data(data_path)

    # Define all partitions we need
    configs = []
    for K in [5, 10]:
        for seed in SEEDS:
            configs.append(("iid", K, seed, {}))
            configs.append(("noniid_07", K, seed, {"dominant_fraction": 0.7}))
            configs.append(("noniid_09", K, seed, {"dominant_fraction": 0.9}))

    partition_map = {}

    print("=" * 70)
    print("PARTITION GENERATION")
    print("=" * 70)

    for strategy, K, seed, kwargs in configs:
        filename = f"{strategy}_K{K}_seed{seed}.json"
        filepath = partition_dir / filename

        partition_map[(strategy, K, seed)] = str(filepath)

        if filepath.exists():
            continue  # Already generated

        if strategy == "iid":
            indices = partition_iid(y_all, K, seed=seed)
        elif strategy.startswith("noniid"):
            indices = partition_noniid_label_skew(y_all, K, seed=seed, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        metadata = {
            "strategy": strategy,
            "num_clients": K,
            "seed": seed,
            "total_samples": len(y_all),
            **kwargs,
        }
        save_partition(indices, metadata, filepath)

    existing = sum(1 for p in partition_map.values() if Path(p).exists())
    print(f"\nPartitions ready: {existing}/{len(partition_map)}")

    return partition_map


# ======================================================================
#  EXPERIMENT GROUP DEFINITIONS
# ======================================================================


def build_experiment_groups(data_path, partition_map, results_dir):
    """
    Define all experiment configurations grouped by research question.

    Each experiment is a dict that can be passed to run_single_experiment().

    Returns
    -------
    groups : dict
        {group_name: list[experiment_config_dict]}
    """
    groups = {}

    # ── EXP1: Baselines ──────────────────────────────────────────────
    # Centralized (Adam, 100 epochs) + local-only for K=5 and K=10
    groups["EXP1"] = []
    for seed in SEEDS:
        groups["EXP1"].append(
            {
                "name": f"centralized_seed{seed}",
                "type": "centralized",
                "data_path": data_path,
                "epochs": 100,
                "lr": 0.001,
                "batch_size": 64,
                "optimizer_type": "adam",
                "seed": seed,
            }
        )
    for K in [5, 10]:
        for seed in SEEDS:
            part_path = partition_map[("iid", K, seed)]
            groups["EXP1"].append(
                {
                    "name": f"local_only_K{K}_iid_seed{seed}",
                    "type": "local_only",
                    "data_path": data_path,
                    "partition_path": part_path,
                    "epochs": 50,
                    "lr": 0.01,
                    "batch_size": 64,
                    "seed": seed,
                }
            )

    # ── EXP2: IID FL Baseline ────────────────────────────────────────
    # Does FedAvg work? K=5, default params, IID
    groups["EXP2"] = []
    for seed in SEEDS:
        part_path = partition_map[("iid", 5, seed)]
        groups["EXP2"].append(
            {
                "name": f"fl_iid_K5_E5_R50_seed{seed}",
                "type": "federated",
                "data_path": data_path,
                "partition_path": part_path,
                "num_rounds": 50,
                "local_epochs": 5,
                "lr": 0.01,
                "batch_size": 64,
                "participation_fraction": 1.0,
                "seed": seed,
            }
        )

    # ── EXP3: Effect of Local Epochs (E) ─────────────────────────────
    # Varies E ∈ {1, 5, 10}, holds everything else at defaults
    groups["EXP3"] = []
    for E in [1, 5, 10]:
        for seed in SEEDS:
            part_path = partition_map[("iid", 5, seed)]
            groups["EXP3"].append(
                {
                    "name": f"fl_iid_K5_E{E}_R50_seed{seed}",
                    "type": "federated",
                    "data_path": data_path,
                    "partition_path": part_path,
                    "num_rounds": 50,
                    "local_epochs": E,
                    "lr": 0.01,
                    "batch_size": 64,
                    "participation_fraction": 1.0,
                    "seed": seed,
                }
            )

    # ── EXP4: Effect of Communication Rounds (R) ─────────────────────
    # Varies R ∈ {10, 25, 50, 100}
    groups["EXP4"] = []
    for R in [10, 25, 50, 100]:
        for seed in SEEDS:
            part_path = partition_map[("iid", 5, seed)]
            groups["EXP4"].append(
                {
                    "name": f"fl_iid_K5_E5_R{R}_seed{seed}",
                    "type": "federated",
                    "data_path": data_path,
                    "partition_path": part_path,
                    "num_rounds": R,
                    "local_epochs": 5,
                    "lr": 0.01,
                    "batch_size": 64,
                    "participation_fraction": 1.0,
                    "seed": seed,
                }
            )

    # ── EXP5: IID vs Non-IID ─────────────────────────────────────────
    # The key research question — how does data heterogeneity affect FL?
    groups["EXP5"] = []
    for strategy in ["iid", "noniid_07", "noniid_09"]:
        for seed in SEEDS:
            part_path = partition_map[(strategy, 5, seed)]
            groups["EXP5"].append(
                {
                    "name": f"fl_{strategy}_K5_E5_R50_seed{seed}",
                    "type": "federated",
                    "data_path": data_path,
                    "partition_path": part_path,
                    "num_rounds": 50,
                    "local_epochs": 5,
                    "lr": 0.01,
                    "batch_size": 64,
                    "participation_fraction": 1.0,
                    "seed": seed,
                }
            )

    # ── EXP6: Scalability (K) ────────────────────────────────────────
    # K=5 vs K=10 — more clients = less data per client
    groups["EXP6"] = []
    for K in [5, 10]:
        for seed in SEEDS:
            part_path = partition_map[("iid", K, seed)]
            groups["EXP6"].append(
                {
                    "name": f"fl_iid_K{K}_E5_R50_seed{seed}",
                    "type": "federated",
                    "data_path": data_path,
                    "partition_path": part_path,
                    "num_rounds": 50,
                    "local_epochs": 5,
                    "lr": 0.01,
                    "batch_size": 64,
                    "participation_fraction": 1.0,
                    "seed": seed,
                }
            )

    # ── EXP7: Partial Participation ──────────────────────────────────
    # Robustness analysis — what if only some clients participate?
    groups["EXP7"] = []
    for frac in [0.3, 0.5, 0.7, 1.0]:
        for seed in SEEDS:
            part_path = partition_map[("iid", 5, seed)]
            frac_str = str(frac).replace(".", "")
            groups["EXP7"].append(
                {
                    "name": f"fl_iid_K5_E5_R50_frac{frac_str}_seed{seed}",
                    "type": "federated",
                    "data_path": data_path,
                    "partition_path": part_path,
                    "num_rounds": 50,
                    "local_epochs": 5,
                    "lr": 0.01,
                    "batch_size": 64,
                    "participation_fraction": frac,
                    "seed": seed,
                }
            )

    return groups


# ======================================================================
#  EXPERIMENT RUNNER
# ======================================================================


def run_single_experiment(config, results_dir, device="cpu", verbose=True):
    """
    Run a single experiment configuration and save results.

    Skips if a results file already exists (resume support).

    Parameters
    ----------
    config : dict
        Experiment configuration with 'name' and 'type' keys.
    results_dir : str
    device : str
    verbose : bool

    Returns
    -------
    results : dict or None (if skipped)
    """
    results_path = Path(results_dir) / f"{config['name']}.json"

    # Resume support: skip if already completed
    if results_path.exists():
        if verbose:
            print(f"  [SKIP] {config['name']} — already exists")
        return load_results(results_path)

    exp_type = config["type"]
    start_time = time.time()

    if exp_type == "centralized":
        results = run_centralized(
            data_path=config["data_path"],
            epochs=config["epochs"],
            lr=config["lr"],
            batch_size=config["batch_size"],
            optimizer_type=config.get("optimizer_type", "adam"),
            scheduler_type=config.get("scheduler_type"),
            device=device,
            seed=config["seed"],
            verbose=verbose,
        )

    elif exp_type == "local_only":
        results = run_local_only(
            data_path=config["data_path"],
            partition_path=config["partition_path"],
            epochs=config["epochs"],
            lr=config["lr"],
            batch_size=config["batch_size"],
            device=device,
            seed=config["seed"],
            verbose=verbose,
        )

    elif exp_type == "federated":
        results = run_federated(
            data_path=config["data_path"],
            partition_path=config["partition_path"],
            num_rounds=config["num_rounds"],
            local_epochs=config["local_epochs"],
            lr=config["lr"],
            batch_size=config["batch_size"],
            participation_fraction=config["participation_fraction"],
            device=device,
            seed=config["seed"],
            verbose=verbose,
        )

    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")

    wall_time = time.time() - start_time
    results["wall_time_sec"] = wall_time
    results["experiment_name"] = config["name"]

    # Save to disk
    save_results(results, results_path, overwrite=True)

    return results


def run_experiment_group(
    group_name, experiments, results_dir, device="cpu", verbose=True
):
    """
    Run all experiments in a group.

    Parameters
    ----------
    group_name : str
    experiments : list[dict]
    results_dir : str
    device : str
    verbose : bool

    Returns
    -------
    all_results : list[dict]
    """
    group_dir = Path(results_dir) / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(experiments)
    n_skipped = 0
    n_run = 0

    print("\n" + "=" * 70)
    print(f"EXPERIMENT GROUP: {group_name} ({n_total} experiments)")
    print("=" * 70)

    all_results = []
    group_start = time.time()

    for i, config in enumerate(experiments):
        print(f"\n[{i+1}/{n_total}] {config['name']}")

        result = run_single_experiment(
            config, str(group_dir), device=device, verbose=verbose
        )

        if result is not None:
            all_results.append(result)
            # Check if it was loaded from cache
            if (group_dir / f"{config['name']}.json").stat().st_mtime < group_start:
                n_skipped += 1
            else:
                n_run += 1

    group_time = time.time() - group_start
    print(f"\n{'─' * 70}")
    print(
        f"Group {group_name} complete: {n_run} run, {n_skipped} skipped, "
        f"{group_time:.0f}s total"
    )

    return all_results


# ======================================================================
#  GROUP SUMMARY — Aggregate results per group
# ======================================================================


def summarize_group(group_name, results_dir, verbose=True):
    """
    Load and aggregate results for an experiment group.

    Groups results by config (ignoring seed), computes mean ± std.

    Returns
    -------
    summary : dict
        {config_label: aggregated_results}
    """
    group_dir = Path(results_dir) / group_name
    if not group_dir.exists():
        print(f"No results for {group_name}")
        return {}

    # Load all results
    all_results = {}
    for fpath in sorted(group_dir.glob("*.json")):
        res = load_results(fpath)
        all_results[fpath.stem] = res

    if not all_results:
        return {}

    # Group by config (strip seed from name)
    config_groups = {}
    for name, res in all_results.items():
        # Remove _seed{N} suffix to group by config
        parts = name.rsplit("_seed", 1)
        config_key = parts[0] if len(parts) > 1 else name
        config_groups.setdefault(config_key, []).append(res)

    # Aggregate each config group
    summary = {}
    for config_key, results_list in sorted(config_groups.items()):
        agg = aggregate_seeds(results_list)
        summary[config_key] = agg

        if verbose and "final_metrics" in agg:
            fm = agg["final_metrics"]
            f1_str = (
                f"{fm['f1_macro']['mean']:.4f} ± {fm['f1_macro']['std']:.4f}"
                if "f1_macro" in fm
                else "N/A"
            )
            acc_str = (
                f"{fm['accuracy']['mean']:.4f} ± {fm['accuracy']['std']:.4f}"
                if "accuracy" in fm
                else "N/A"
            )
            print(
                f"  {config_key:<40}  F1={f1_str}  Acc={acc_str}  "
                f"(n={agg['n_seeds']})"
            )

    return summary


# ======================================================================
#  FULL PIPELINE
# ======================================================================


def run_all(
    data_path,
    partition_dir,
    results_dir,
    device="cpu",
    groups_to_run=None,
    verbose=True,
    dry_run=False,
):
    """
    Run the complete experiment pipeline.

    Parameters
    ----------
    data_path : str
    partition_dir : str
    results_dir : str
    device : str
    groups_to_run : list[str], optional
        If provided, only run these groups. Otherwise run all.
    verbose : bool
    dry_run : bool
        If True, just print what would be run without executing.
    """
    pipeline_start = time.time()

    print("╔" + "═" * 68 + "╗")
    print("║  FL-IDS EXPERIMENT PIPELINE                                        ║")
    print("║  Evaluating Federated Learning for IIoT Intrusion Detection        ║")
    print("╚" + "═" * 68 + "╝")
    print(f"\nData:       {data_path}")
    print(f"Partitions: {partition_dir}")
    print(f"Results:    {results_dir}")
    print(f"Device:     {device}")
    print(f"Seeds:      {SEEDS}")
    print(f"Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── 1. Generate partitions ───────────────────────────────────────
    partition_map = ensure_partitions(data_path, partition_dir)

    # ── 2. Build experiment grid ─────────────────────────────────────
    groups = build_experiment_groups(data_path, partition_map, results_dir)

    # Filter to requested groups
    if groups_to_run:
        groups = {k: v for k, v in groups.items() if k in groups_to_run}

    # Count total experiments
    total_exps = sum(len(v) for v in groups.values())
    print(f"\nExperiment groups: {list(groups.keys())}")
    print(f"Total experiments: {total_exps}")

    if dry_run:
        print("\n[DRY RUN] Experiments that would be executed:")
        for gname, exps in groups.items():
            print(f"\n  {gname} ({len(exps)} experiments):")
            for exp in exps:
                results_path = Path(results_dir) / gname / f"{exp['name']}.json"
                status = "SKIP (exists)" if results_path.exists() else "RUN"
                print(f"    [{status}] {exp['name']}")
        return

    # ── 3. Run each group ────────────────────────────────────────────
    for group_name, experiments in groups.items():
        run_experiment_group(
            group_name,
            experiments,
            results_dir,
            device=device,
            verbose=verbose,
        )

    # ── 4. Print summaries ───────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("EXPERIMENT SUMMARIES")
    print("=" * 70)

    for group_name in groups:
        print(f"\n── {group_name} ──")
        summarize_group(group_name, results_dir, verbose=True)

    pipeline_time = time.time() - pipeline_start
    print(f"\n{'=' * 70}")
    print(f"Pipeline complete in {pipeline_time/60:.1f} minutes")
    print(f"Results saved to: {results_dir}")


# ======================================================================
#  CLI ENTRY POINT
# ======================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FL-IDS Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Experiment Groups:
  EXP1  Baselines (centralized + local-only)
  EXP2  IID FL baseline (K=5, E=5, R=50)
  EXP3  Effect of local epochs E ∈ {1, 5, 10}
  EXP4  Effect of communication rounds R ∈ {10, 25, 50, 100}
  EXP5  IID vs Non-IID (skew=0.7, skew=0.9)
  EXP6  Scalability K=5 vs K=10
  EXP7  Partial participation {0.3, 0.5, 0.7, 1.0}

Examples:
  python experiments.py --data_path ../data/processed/datasense_preprocessed.csv
  python experiments.py --data_path ... --group EXP3 EXP5
  python experiments.py --data_path ... --dry_run
        """,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to datasense_preprocessed.csv",
    )
    parser.add_argument(
        "--partition_dir",
        type=str,
        default="../data/partitions",
        help="Directory for partition JSON files (default: ../data/partitions)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="../results",
        help="Directory to save experiment results (default: ../results)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cpu, cuda, or mps (auto-detected if not specified)",
    )
    parser.add_argument(
        "--group",
        type=str,
        nargs="+",
        default=None,
        help="Run only these experiment groups (e.g., --group EXP3 EXP5)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be run without executing",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-round training output",
    )

    args = parser.parse_args()

    # Auto-detect device
    import torch

    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    run_all(
        data_path=args.data_path,
        partition_dir=args.partition_dir,
        results_dir=args.results_dir,
        device=device,
        groups_to_run=args.group,
        verbose=not args.quiet,
        dry_run=args.dry_run,
    )
