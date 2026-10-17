"""
Generate slide-ready figures from baseline_results.json:
  - outputs/model_comparison.png   grouped bar chart across all metrics
  - outputs/precision_recall.png   the headline precision/recall story

    python plot_results.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

METRICS = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
LABELS = ["ROC-AUC", "PR-AUC", "F1", "Precision", "Recall"]
COLORS = {
    "Rule-based (amount)": "#94a3b8",
    "MLP (no graph)": "#60a5fa",
    "XGBoost (Tabular)": "#f59e0b",
    "GraphSAGE (ours)": "#3b82f6",
    "EdgeGINE (ours)": "#ef4444",
    "GNN + XGBoost (Hybrid)": "#10b981",
}


def main():
    with open(config.OUTPUT_DIR / "baseline_results.json") as f:
        results = json.load(f)

    models = list(results.keys())
    x = np.arange(len(METRICS))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, name in enumerate(models):
        vals = [results[name][m] for m in METRICS]
        bars = ax.bar(x + i * width, vals, width, label=name,
                      color=COLORS.get(name, None))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(LABELS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Money-laundering detection: rule-based vs. plain ML vs. GNN\n"
                 "(same accounts, same test split)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = config.OUTPUT_DIR / "model_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"[done] saved -> {out}")


if __name__ == "__main__":
    main()
