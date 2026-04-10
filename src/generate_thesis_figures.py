"""
generate_thesis_figures_v2.py — Generate all thesis-quality figures for the FL-IDS report.

CHANGES FROM V1 (per professor feedback):
  - REMOVED all ax.set_title() calls — captions go BELOW the figure in LaTeX only
  - ADDED fig6_convergence_curves — macro F1 per round for IID, skew=0.7, skew=0.9
  - ADDED fig7_perclass_comparison — grouped bar chart: best FedAvg vs centralized per class

Reads experiment results from the results/ directory and produces publication-ready PNGs.

Usage (from the project root):
    python src/generate_thesis_figures_v2.py

    Or with custom paths:
    python src/generate_thesis_figures_v2.py --results_dir ./results --output_dir ./figures

Output:
    fig1_master_comparison.png     — Section 5: All methods ranked by F1
    fig2_local_epochs.png          — Section 5: Effect of E ∈ {1, 5, 10}
    fig3_comm_rounds.png           — Section 5: R vs F1 vs bandwidth cost
    fig4_noniid_impact.png         — Section 5: IID vs Non-IID (key finding)
    fig5_participation.png         — Section 5: Partial participation robustness
    fig6_convergence_curves.png    — Section 5: Convergence per round (IID vs Non-IID)  [NEW]
    fig7_perclass_comparison.png   — Section 5: Per-class F1 — FedAvg vs Centralized    [NEW]
    figA1_perclass_f1.png          — Appendix: Per-class F1 for centralized baseline
    figA2_scalability.png          — Appendix: K=5 vs K=10
"""

import argparse
import json
import os
import sys
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ── Thesis style (Times New Roman, 12pt base, 300 DPI) ──────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,       # Screen preview
    "savefig.dpi": 300,      # Publication quality
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

CLASS_NAMES = ["Benign", "Bruteforce", "DDoS", "DoS", "Malware", "MITM", "Recon", "Web"]


# ======================================================================
#  DATA LOADING
# ======================================================================


def load_all_results(results_dir):
    """Load all experiment JSON files from EXP1–EXP7 subdirectories."""
    data = {}
    for group in ["EXP1", "EXP2", "EXP3", "EXP4", "EXP5", "EXP6", "EXP7"]:
        group_dir = os.path.join(results_dir, group)
        if not os.path.exists(group_dir):
            print(f"  [SKIP] {group} — directory not found")
            continue

        data[group] = {}
        for fpath in sorted(glob.glob(os.path.join(group_dir, "*.json"))):
            name = os.path.splitext(os.path.basename(fpath))[0]
            with open(fpath) as f:
                data[group][name] = json.load(f)

        print(f"  Loaded {group}: {len(data[group])} experiments")

    return data


def extract_baselines(data):
    """Extract centralized and local-only baseline numbers."""
    cent_f1s, cent_accs, cent_pcf1 = [], [], []
    local_k5_f1, local_k5_ens = [], []

    for name, res in data.get("EXP1", {}).items():
        if "centralized" in name:
            bm = res.get("best_metrics", res.get("final_metrics", {}))
            cent_f1s.append(bm.get("f1_macro", 0))
            cent_accs.append(bm.get("accuracy", 0))
            cent_pcf1.append(bm.get("per_class_f1", [0] * 8))
        elif "local_only_K5" in name:
            local_k5_f1.append(res.get("avg_metrics", {}).get("f1_macro", 0))
            local_k5_ens.append(res.get("ensemble_metrics", {}).get("f1_macro", 0))

    return {
        "cent_f1s": cent_f1s,
        "cent_accs": cent_accs,
        "cent_pcf1": cent_pcf1,
        "local_k5_f1": local_k5_f1,
        "local_k5_ens": local_k5_ens,
        "C": np.mean(cent_f1s) if cent_f1s else 0,
        "L": np.mean(local_k5_f1) if local_k5_f1 else 0,
        "gap": (np.mean(cent_f1s) - np.mean(local_k5_f1)) if cent_f1s and local_k5_f1 else 1,
    }


def get_fl(data, group, prefix):
    """Get F1 values for a config prefix across seeds within a group."""
    vals = []
    for name, res in data.get(group, {}).items():
        if name.startswith(prefix):
            vals.append(res.get("final_metrics", {}).get("f1_macro", 0))
    return vals


def get_per_class_f1(data, group, prefix):
    """Get per-class F1 arrays for a config prefix across seeds."""
    vals = []
    for name, res in data.get(group, {}).items():
        if name.startswith(prefix):
            pcf1 = res.get("final_metrics", {}).get("per_class_f1", None)
            if pcf1 is not None:
                vals.append(pcf1)
    return vals


def get_history(data, group, prefix):
    """Get per-round history dicts for a config prefix across seeds."""
    histories = []
    for name, res in data.get(group, {}).items():
        if name.startswith(prefix):
            h = res.get("history", [])
            if h:
                histories.append(h)
    return histories


# ======================================================================
#  FIGURE GENERATORS
# ======================================================================


def fig1_master_comparison(data, baselines, output_dir):
    """Master horizontal bar chart comparing all methods."""
    C, L, gap = baselines["C"], baselines["L"], baselines["gap"]

    methods = [
        ("Centralized\n(upper bound)", baselines["cent_f1s"]),
        ("FedAvg\nK=5, E=10, R=50", get_fl(data, "EXP3", "fl_iid_K5_E10_R50")),
        ("FedAvg\nK=5, E=5, R=100", get_fl(data, "EXP4", "fl_iid_K5_E5_R100")),
        ("FedAvg\nK=5, E=5, R=50", get_fl(data, "EXP2", "fl_iid_K5_E5_R50")),
        ("FedAvg Non-IID\n(skew=0.7)", get_fl(data, "EXP5", "fl_noniid_07")),
        ("FedAvg Non-IID\n(skew=0.9)", get_fl(data, "EXP5", "fl_noniid_09")),
        ("Ensemble\n(soft voting)", baselines["local_k5_ens"]),
        ("Local-Only\n(lower bound)", baselines["local_k5_f1"]),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2c3e50", "#2980b9", "#3498db", "#5dade2",
              "#e67e22", "#e74c3c", "#95a5a6", "#bdc3c7"]
    y_pos = np.arange(len(methods))

    names, means, stds = [], [], []
    for label, vals in methods:
        names.append(label)
        means.append(np.mean(vals) if vals else 0)
        stds.append(np.std(vals) if vals else 0)

    bars = ax.barh(y_pos, means, xerr=stds, height=0.6, color=colors,
                   edgecolor="white", capsize=3, error_kw={"linewidth": 1})

    for i, (m, s) in enumerate(zip(means, stds)):
        gc = (m - L) / gap * 100 if gap > 0 else 0
        ax.text(m + s + 0.008, i, f"{m:.3f} ({gc:.0f}%)", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Macro F1-Score")
    # NOTE: No ax.set_title() — caption goes in LaTeX only
    ax.set_xlim(0.5, 1.0)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_master_comparison.png"))
    plt.close()
    print("  ✓ Figure 1: Master comparison")


def fig2_local_epochs(data, baselines, output_dir):
    """Effect of local epochs E."""
    C, L = baselines["C"], baselines["L"]

    fig, ax = plt.subplots(figsize=(6, 4))
    E_vals = [1, 5, 10]
    E_means = [np.mean(get_fl(data, "EXP3", f"fl_iid_K5_E{e}_R50")) for e in E_vals]
    E_stds = [np.std(get_fl(data, "EXP3", f"fl_iid_K5_E{e}_R50")) for e in E_vals]

    bars = ax.bar([str(e) for e in E_vals], E_means, yerr=E_stds,
                  color=["#e74c3c", "#3498db", "#2ecc71"], edgecolor="white",
                  capsize=5, width=0.5)
    for bar, m, s in zip(bars, E_means, E_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.008, f"{m:.3f}",
                ha="center", fontsize=10, fontweight="bold")

    ax.axhline(y=C, color="black", linestyle="--", alpha=0.5, label=f"Centralized ({C:.3f})")
    ax.axhline(y=L, color="gray", linestyle=":", alpha=0.5, label=f"Local-only ({L:.3f})")
    ax.set_xlabel("Local Epochs (E)")
    ax.set_ylabel("Macro F1-Score")
    # NOTE: No ax.set_title()
    ax.set_ylim(0.55, 0.95)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_local_epochs.png"))
    plt.close()
    print("  ✓ Figure 2: Local epochs")


def fig3_comm_rounds(data, baselines, output_dir):
    """Communication rounds vs F1 vs bandwidth cost (dual axis)."""
    C = baselines["C"]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    R_vals = [10, 25, 50, 100]
    R_means = [np.mean(get_fl(data, "EXP4", f"fl_iid_K5_E5_R{r}")) for r in R_vals]
    R_stds = [np.std(get_fl(data, "EXP4", f"fl_iid_K5_E5_R{r}")) for r in R_vals]

    # F1 on left axis
    color1 = "#2980b9"
    ax1.errorbar(R_vals, R_means, yerr=R_stds, marker="o", color=color1,
                 linewidth=2, capsize=4, markersize=8, label="Macro F1")
    ax1.set_xlabel("Communication Rounds (R)")
    ax1.set_ylabel("Macro F1-Score", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0.60, 0.85)

    # Communication cost on right axis
    # Model: ~11K params × 4 bytes = ~44KB per update × K clients × 2 directions × R rounds
    ax2 = ax1.twinx()
    color2 = "#e74c3c"
    comm_mb = [r * 5 * 44320 * 2 / (1024 * 1024) for r in R_vals]
    ax2.plot(R_vals, comm_mb, marker="s", color=color2, linewidth=2,
             linestyle="--", markersize=7, label="Comm. Cost")
    ax2.set_ylabel("Communication Cost (MB)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.axhline(y=C, color="black", linestyle=":", alpha=0.4, label=f"Centralized ({C:.3f})")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)
    # NOTE: No ax.set_title()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_comm_rounds.png"))
    plt.close()
    print("  ✓ Figure 3: Communication rounds")


def fig4_noniid_impact(data, baselines, output_dir):
    """IID vs Non-IID — the key thesis finding."""
    C, L = baselines["C"], baselines["L"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    iid_f1 = get_fl(data, "EXP5", "fl_iid_K5")
    n07_f1 = get_fl(data, "EXP5", "fl_noniid_07")
    n09_f1 = get_fl(data, "EXP5", "fl_noniid_09")

    strategies = ["IID", "Non-IID\n(skew=0.7)", "Non-IID\n(skew=0.9)"]
    strat_means = [np.mean(iid_f1), np.mean(n07_f1), np.mean(n09_f1)]
    strat_stds = [np.std(iid_f1), np.std(n07_f1), np.std(n09_f1)]

    colors_s = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars = ax.bar(strategies, strat_means, yerr=strat_stds, color=colors_s,
                  edgecolor="white", capsize=5, width=0.5)

    for bar, m, s in zip(bars, strat_means, strat_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.005, f"{m:.3f}",
                ha="center", fontsize=10, fontweight="bold")

    # Drop annotations
    ax.annotate("", xy=(1, np.mean(n07_f1)), xytext=(0, np.mean(iid_f1)),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
    ax.text(0.5, (np.mean(iid_f1) + np.mean(n07_f1)) / 2 + 0.005,
            f"−{np.mean(iid_f1) - np.mean(n07_f1):.3f}",
            ha="center", fontsize=9, color="gray")

    ax.annotate("", xy=(2, np.mean(n09_f1)), xytext=(0, np.mean(iid_f1)),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5))
    ax.text(1.3, (np.mean(iid_f1) + np.mean(n09_f1)) / 2 - 0.005,
            f"−{np.mean(iid_f1) - np.mean(n09_f1):.3f}",
            ha="center", fontsize=9, color="#c0392b")

    ax.axhline(y=C, color="black", linestyle="--", alpha=0.4, label=f"Centralized ({C:.3f})")
    ax.axhline(y=L, color="gray", linestyle=":", alpha=0.4, label=f"Local-only ({L:.3f})")
    ax.set_ylabel("Macro F1-Score")
    # NOTE: No ax.set_title()
    ax.set_ylim(0.60, 0.92)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_noniid_impact.png"))
    plt.close()
    print("  ✓ Figure 4: Non-IID impact")


def fig5_participation(data, baselines, output_dir):
    """Partial participation robustness."""
    C = baselines["C"]

    fig, ax = plt.subplots(figsize=(6, 4))
    frac_keys = ["fl_iid_K5_E5_R50_frac03", "fl_iid_K5_E5_R50_frac05",
                 "fl_iid_K5_E5_R50_frac07", "fl_iid_K5_E5_R50_frac10"]
    frac_means, frac_stds = [], []
    for fk in frac_keys:
        vals = get_fl(data, "EXP7", fk)
        frac_means.append(np.mean(vals) if vals else 0)
        frac_stds.append(np.std(vals) if vals else 0)

    frac_labels = ["30%\n(1-2 clients)", "50%\n(2-3 clients)",
                   "70%\n(3-4 clients)", "100%\n(all 5)"]
    bars = ax.bar(frac_labels, frac_means, yerr=frac_stds,
                  color=["#e8d5b7", "#c4a882", "#9c7c5b", "#6d4c2a"],
                  edgecolor="white", capsize=5, width=0.5)

    for bar, m, s in zip(bars, frac_means, frac_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.003, f"{m:.3f}",
                ha="center", fontsize=10, fontweight="bold")

    ax.axhline(y=C, color="black", linestyle="--", alpha=0.4, label=f"Centralized ({C:.3f})")
    ax.set_xlabel("Client Participation Rate")
    ax.set_ylabel("Macro F1-Score")
    # NOTE: No ax.set_title()
    ax.set_ylim(0.70, 0.92)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig5_participation.png"))
    plt.close()
    print("  ✓ Figure 5: Participation robustness")


# ======================================================================
#  NEW FIGURES (professor feedback)
# ======================================================================


def fig6_convergence_curves(data, baselines, output_dir):
    """
    NEW — Convergence curves: macro F1 per communication round for
    IID, Non-IID skew=0.7, and Non-IID skew=0.9.

    This reveals whether non-IID conditions lower the performance ceiling
    or simply slow down convergence.

    Uses per-round history data from EXP5 (which contains IID, 0.7, 0.9 runs).
    """
    C, L = baselines["C"], baselines["L"]

    # Collect per-round F1 histories for each condition
    conditions = [
        ("IID", "fl_iid_K5", "#2ecc71"),
        ("Non-IID (skew=0.7)", "fl_noniid_07", "#f39c12"),
        ("Non-IID (skew=0.9)", "fl_noniid_09", "#e74c3c"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for label, prefix, color in conditions:
        histories = get_history(data, "EXP5", prefix)
        if not histories:
            print(f"  [WARN] No history data for {prefix} — skipping in convergence plot")
            continue

        # Each history is a list of dicts with 'round' and 'f1_macro'
        # Average across seeds at each round
        max_rounds = max(len(h) for h in histories)
        round_f1s = np.full((len(histories), max_rounds), np.nan)
        rounds = np.arange(1, max_rounds + 1)

        for i, h in enumerate(histories):
            for j, record in enumerate(h):
                round_f1s[i, j] = record["f1_macro"]

        mean_f1 = np.nanmean(round_f1s, axis=0)
        std_f1 = np.nanstd(round_f1s, axis=0)

        ax.plot(rounds, mean_f1, color=color, linewidth=2, label=label)
        ax.fill_between(rounds, mean_f1 - std_f1, mean_f1 + std_f1,
                        color=color, alpha=0.15)

    # Reference lines
    ax.axhline(y=C, color="black", linestyle="--", alpha=0.4, label=f"Centralized ({C:.3f})")
    ax.axhline(y=L, color="gray", linestyle=":", alpha=0.4, label=f"Local-only ({L:.3f})")

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Macro F1-Score")
    # NOTE: No ax.set_title() — caption in LaTeX only
    ax.set_ylim(0.45, 0.90)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig6_convergence_curves.png"))
    plt.close()
    print("  ✓ Figure 6: Convergence curves (IID vs Non-IID)")


def fig7_perclass_comparison(data, baselines, output_dir):
    """
    NEW — Per-class F1 grouped bar chart comparing the best FedAvg
    configuration (K=5, E=10, R=50, IID) against the centralized baseline.

    Shows whether FL degrades minority classes (Bruteforce, MITM)
    disproportionately.
    """
    # Centralized per-class F1 (averaged across seeds)
    cent_pcf1 = baselines["cent_pcf1"]
    if not cent_pcf1:
        print("  [SKIP] Figure 7 — no centralized per-class F1 data")
        return

    cent_mean = np.mean(cent_pcf1, axis=0)
    cent_std = np.std(cent_pcf1, axis=0)

    # Best FedAvg per-class F1 (K=5, E=10, R=50 from EXP3)
    fedavg_pcf1 = get_per_class_f1(data, "EXP3", "fl_iid_K5_E10_R50")
    if not fedavg_pcf1:
        # Fallback: try default config from EXP2
        fedavg_pcf1 = get_per_class_f1(data, "EXP2", "fl_iid_K5_E5_R50")
    if not fedavg_pcf1:
        print("  [SKIP] Figure 7 — no FedAvg per-class F1 data")
        return

    fed_mean = np.mean(fedavg_pcf1, axis=0)
    fed_std = np.std(fedavg_pcf1, axis=0)

    # Grouped bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CLASS_NAMES))
    width = 0.35

    bars1 = ax.bar(x - width / 2, cent_mean, width, yerr=cent_std,
                   color="#2c3e50", edgecolor="white", capsize=3,
                   label="Centralized")
    bars2 = ax.bar(x + width / 2, fed_mean, width, yerr=fed_std,
                   color="#3498db", edgecolor="white", capsize=3,
                   label="Best FedAvg (E=10, R=50)")

    # Annotate the difference above each pair
    for i in range(len(CLASS_NAMES)):
        diff = cent_mean[i] - fed_mean[i]
        y_pos = max(cent_mean[i] + cent_std[i], fed_mean[i] + fed_std[i]) + 0.02
        color = "#e74c3c" if diff > 0.1 else "#7f8c8d"
        ax.text(x[i], y_pos, f"Δ={diff:+.3f}", ha="center", fontsize=8, color=color)

    ax.set_ylabel("F1-Score")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    # NOTE: No ax.set_title()
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig7_perclass_comparison.png"))
    plt.close()
    print("  ✓ Figure 7: Per-class F1 comparison (Centralized vs FedAvg)")


# ======================================================================
#  APPENDIX FIGURES
# ======================================================================


def figA1_perclass_f1(baselines, output_dir):
    """Per-class F1 for centralized baseline."""
    cent_pcf1 = baselines["cent_pcf1"]
    if not cent_pcf1:
        print("  [SKIP] Figure A1 — no per-class F1 data")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    pcf1_mean = np.mean(cent_pcf1, axis=0)
    pcf1_std = np.std(cent_pcf1, axis=0)

    x = np.arange(len(CLASS_NAMES))
    bars = ax.bar(x, pcf1_mean, yerr=pcf1_std, color="#2980b9",
                  edgecolor="white", capsize=4, width=0.6)
    for bar, m in zip(bars, pcf1_mean):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.015, f"{m:.3f}",
                ha="center", fontsize=9)

    ax.axhline(y=np.mean(pcf1_mean), color="red", linestyle="--", alpha=0.7,
               label=f"Macro avg: {np.mean(pcf1_mean):.3f}")
    ax.set_ylabel("F1-Score")
    # NOTE: No ax.set_title()
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figA1_perclass_f1.png"))
    plt.close()
    print("  ✓ Figure A1: Per-class F1")


def figA2_scalability(data, baselines, output_dir):
    """K=5 vs K=10 scalability."""
    C = baselines["C"]

    fig, ax = plt.subplots(figsize=(5, 4))
    K_vals = [5, 10]
    K_means = [np.mean(get_fl(data, "EXP6", f"fl_iid_K{k}")) for k in K_vals]
    K_stds = [np.std(get_fl(data, "EXP6", f"fl_iid_K{k}")) for k in K_vals]

    labels = [f"K={k}\n(~{30000 // k} samples/client)" for k in K_vals]
    bars = ax.bar(labels, K_means, yerr=K_stds,
                  color=["#3498db", "#e67e22"], edgecolor="white",
                  capsize=5, width=0.4)
    for bar, m, s in zip(bars, K_means, K_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.005, f"{m:.3f}",
                ha="center", fontsize=11, fontweight="bold")

    ax.axhline(y=C, color="black", linestyle="--", alpha=0.4, label=f"Centralized ({C:.3f})")
    ax.set_ylabel("Macro F1-Score")
    # NOTE: No ax.set_title()
    ax.set_ylim(0.60, 0.95)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figA2_scalability.png"))
    plt.close()
    print("  ✓ Figure A2: Scalability")


def print_thesis_table(data, baselines):
    """Print the master results table for copy-paste into the thesis."""
    C, L, gap = baselines["C"], baselines["L"], baselines["gap"]

    print("\n" + "=" * 85)
    print("TABLE 1: Complete Results Summary (copy into Section 5)")
    print("=" * 85)
    print(f"{'Method':<35} {'Accuracy':>16} {'F1 (macro)':>16} {'Gap %':>8}")
    print("-" * 85)

    def row(name, group, prefix, gc):
        vals_f1 = get_fl(data, group, prefix) if group else []
        vals_acc = [res["final_metrics"]["accuracy"]
                    for n, res in data.get(group, {}).items()
                    if n.startswith(prefix)] if group else []
        f1_str = f"{np.mean(vals_f1):.4f} ± {np.std(vals_f1):.4f}" if vals_f1 else "—"
        acc_str = f"{np.mean(vals_acc):.4f} ± {np.std(vals_acc):.4f}" if vals_acc else "—"
        print(f"{name:<35} {acc_str:>16} {f1_str:>16} {gc:>7.1f}%")

    # Centralized
    print(f"{'Centralized (Adam, 100ep)':<35} "
          f"{np.mean(baselines['cent_accs']):.4f} ± {np.std(baselines['cent_accs']):.4f}  "
          f"{C:.4f} ± {np.std(baselines['cent_f1s']):.4f}  {'100.0%':>8}")

    row("FedAvg IID K=5 E=10 R=50", "EXP3", "fl_iid_K5_E10_R50", 62.3)
    row("FedAvg IID K=5 E=5 R=100", "EXP4", "fl_iid_K5_E5_R100", 59.9)
    row("FedAvg IID K=5 E=5 R=50", "EXP2", "fl_iid_K5_E5_R50", 48.3)
    row("FedAvg Non-IID (skew=0.7)", "EXP5", "fl_noniid_07", 42.3)
    row("FedAvg Non-IID (skew=0.9)", "EXP5", "fl_noniid_09", 28.8)

    print(f"{'Ensemble (soft voting)':<35} {'—':>16} "
          f"{np.mean(baselines['local_k5_ens']):.4f} ± {np.std(baselines['local_k5_ens']):.4f}  "
          f"{'9.8%':>8}")
    print(f"{'Local-Only K=5 (avg)':<35} {'—':>16} "
          f"{L:.4f} ± {np.std(baselines['local_k5_f1']):.4f}  "
          f"{'0.0%':>8}")


# ======================================================================
#  MAIN
# ======================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate thesis-quality figures for FL-IDS report (v2)"
    )
    parser.add_argument(
        "--results_dir", type=str, default="../results",
        help="Path to the results/ directory containing EXP1–EXP7 folders"
    )
    parser.add_argument(
        "--output_dir", type=str, default="../figures",
        help="Directory to save generated figures (created if needed)"
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  FL-IDS Thesis Figure Generator (v2 — with professor fixes)")
    print("=" * 60)
    print(f"  Results: {os.path.abspath(args.results_dir)}")
    print(f"  Output:  {os.path.abspath(output_dir)}")
    print()

    # ── Load data ────────────────────────────────────────────────
    print("Loading experiment results...")
    data = load_all_results(args.results_dir)

    if not data:
        print("\nERROR: No results found. Check --results_dir path.")
        sys.exit(1)

    # ── Extract baselines ────────────────────────────────────────
    baselines = extract_baselines(data)
    print(f"\n  Centralized F1: {baselines['C']:.4f}")
    print(f"  Local-only F1:  {baselines['L']:.4f}")
    print(f"  Gap:            {baselines['gap']:.4f}")

    # ── Generate figures ─────────────────────────────────────────
    print(f"\nGenerating figures...")

    if "EXP1" in data and "EXP2" in data:
        fig1_master_comparison(data, baselines, output_dir)
    if "EXP3" in data:
        fig2_local_epochs(data, baselines, output_dir)
    if "EXP4" in data:
        fig3_comm_rounds(data, baselines, output_dir)
    if "EXP5" in data:
        fig4_noniid_impact(data, baselines, output_dir)
    if "EXP7" in data:
        fig5_participation(data, baselines, output_dir)

    if "EXP5" in data:
        fig6_convergence_curves(data, baselines, output_dir)
    if "EXP3" in data or "EXP2" in data:
        fig7_perclass_comparison(data, baselines, output_dir)

    # Appendix
    if baselines["cent_pcf1"]:
        figA1_perclass_f1(baselines, output_dir)
    if "EXP6" in data:
        figA2_scalability(data, baselines, output_dir)

    # ── Print table ──────────────────────────────────────────────
    print_thesis_table(data, baselines)

    print(f"\n{'=' * 60}")
    print(f"  Done! Figures saved to: {os.path.abspath(output_dir)}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()