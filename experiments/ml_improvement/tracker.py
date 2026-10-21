"""
Experiment tracking infrastructure for ML Improvement Phase.
Maintains results.csv with standardized metrics and comparison utilities.
"""
import csv
import json
import os
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent / "results.csv"

FIELDNAMES = [
    "experiment_id",
    "description",
    "features",
    "graph_config",
    "model_config",
    "loss_config",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "threshold",
    "training_time",
    "status"
]

def init_registry():
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

def log_experiment(record: dict):
    init_registry()
    # Read existing
    existing = []
    if CSV_PATH.exists():
        with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing = [r for r in reader if r.get("experiment_id") != record["experiment_id"]]
    existing.append(record)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for r in existing:
            writer.writerow(r)

def print_summary_table():
    init_registry()
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print("\n" + "=" * 105)
    print(f"| {'Exp ID':<16} | {'PR-AUC':>7} | {'ROC-AUC':>7} | {'F1':>6} | {'Prec':>6} | {'Recall':>6} | {'Thr':>6} | {'Status':<14} |")
    print(f"|:{'-'*16}-|-{'-'*7}:|-{'-'*7}:|-{'-'*6}:|-{'-'*6}:|-{'-'*6}:|-{'-'*6}:|-{'-'*14}:|")
    for r in rows:
        pr = float(r.get("pr_auc", 0))
        roc = float(r.get("roc_auc", 0))
        f1 = float(r.get("f1", 0))
        p = float(r.get("precision", 0))
        rec = float(r.get("recall", 0))
        thr = float(r.get("threshold", 0))
        print(f"| {r['experiment_id']:<16} | {pr:>7.4f} | {roc:>7.4f} | {f1:>6.4f} | {p:>6.4f} | {rec:>6.4f} | {thr:>6.4f} | {r['status']:<14} |")
    print("=" * 105 + "\n")

if __name__ == "__main__":
    init_registry()
    print("Experiment registry initialized at:", CSV_PATH)
