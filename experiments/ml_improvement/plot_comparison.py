"""
Generate high-resolution benchmark visualization for TRACE ML Improvement Phase.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

def main():
    experiments = [
        "Rule-Based",
        "MLP (No Graph)",
        "GraphSAGE (Baseline)",
        "EXP_001 (Bidirectional)",
        "EXP_002 (29 Feats)",
        "EXP_003 (29F + Bidir)",
        "EXP_004 (Residuals)",
        "EXP_006 (Focal a=0.80)",
        "EXP_007 (Hybrid GNN+XGB)"
    ]
    pr_aucs = [0.0210, 0.1840, 0.3420, 0.3756, 0.3725, 0.4125, 0.4089, 0.4112, 0.4569]
    f1_scores = [0.0440, 0.2610, 0.3680, 0.4102, 0.4096, 0.4460, 0.4400, 0.4399, 0.4886]

    x = np.arange(len(experiments))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)
    plt.subplots_adjust(bottom=0.25)

    rects1 = ax.bar(x - width/2, pr_aucs, width, label="PR-AUC (Primary Metric)", color="#2563eb", alpha=0.9)
    rects2 = ax.bar(x + width/2, f1_scores, width, label="F1-Score (Tuned Operating Point)", color="#059669", alpha=0.9)

    # Highlight baseline and targets
    ax.axhline(0.3420, color="#dc2626", linestyle="--", linewidth=1.5, label="Frozen Baseline PR-AUC (0.342)")
    ax.axhline(0.4000, color="#d97706", linestyle="-.", linewidth=1.5, label="Primary Target (0.400)")
    ax.axhline(0.4500, color="#7c3aed", linestyle=":", linewidth=1.5, label="Stretch Target (0.450)")

    ax.set_ylabel("Metric Score", fontsize=12, fontweight="bold")
    ax.set_title("TRACE Anti-Money Laundering Detection Benchmark\nSystematic Progression from GraphSAGE Baseline to 0.45+ PR-AUC", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=30, ha="right", fontsize=10, fontweight="medium")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=10, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_ylim(0, 0.55)

    # Add value annotations
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1e3a8a")

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#064e3b")

    out_path = Path("experiments/ml_improvement/pr_auc_comparison.png")
    plt.savefig(out_path, bbox_inches="tight")
    print(f"[plot] Saved benchmark comparison chart -> {out_path}")

if __name__ == "__main__":
    main()
