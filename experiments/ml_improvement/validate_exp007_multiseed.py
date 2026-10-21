"""
Step 5: Multi-Seed Robustness Validation for Leading Candidate EXP_007_HYBRID.
Runs 4 distinct random seeds (42, 123, 999, 2026) on the Hybrid architecture.
Calculates mean, std, min, and max across all metrics to verify stability.
"""
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, average_precision_score,
    f1_score, precision_recall_fscore_support
)

import config
from experiments.ml_improvement.runner import AdvancedGraphSAGE

def main():
    print("=== EXP_007_HYBRID MULTI-SEED VALIDATION (Seeds: 42, 123, 999, 2026) ===")
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data and features
    data = torch.load(config.GRAPH_PT, weights_only=False)
    x_29 = torch.load("experiments/ml_improvement/extended_features.pt", weights_only=False)
    undir_edges = to_undirected(data.edge_index)

    # Load frozen EXP_003 GNN checkpoint to extract representations
    ckpt = torch.load("experiments/ml_improvement/checkpoints/EXP_003_FEAT29_UNDIR.pt", weights_only=False)
    gnn = AdvancedGraphSAGE(
        in_channels=29, hidden_channels=128, num_layers=3, dropout=0.2,
        use_residual=False, norm_type="batch"
    ).to(device)
    gnn.load_state_dict(ckpt["model_state"])
    gnn.eval()

    print("[multiseed] Extracting 128-dim GNN embeddings...")
    with torch.no_grad():
        x_dev = x_29.to(device)
        edges_dev = undir_edges.to(device)
        h = x_dev
        for block in gnn.blocks:
            conv = block["conv"]
            norm = block["norm"]
            h = conv(h, edges_dev)
            h = norm(h)
            h = F.relu(h)
        emb = h.cpu().numpy()

    X_comb = np.hstack([x_29.numpy(), emb])
    y = data.y.numpy()
    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()

    y_train = y[train_mask]
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw = float(n_neg / max(1, n_pos)) * 0.7

    seeds = [42, 123, 999, 2026]
    results = []

    print(f"\n| {'Seed':>6} | {'PR-AUC':>8} | {'ROC-AUC':>8} | {'F1':>8} | {'Precision':>10} | {'Recall':>8} | {'Threshold':>10} |")
    print(f"|:{'-'*6}:|-{'-'*8}:|-{'-'*8}:|-{'-'*8}:|-{'-'*10}:|-{'-'*8}:|-{'-'*10}:|")

    for seed in seeds:
        clf = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            scale_pos_weight=spw,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="aucpr",
            tree_method="hist",
            device="cuda" if torch.cuda.is_available() else "cpu",
            random_state=seed,
        )
        clf.fit(X_comb[train_mask], y_train)

        # Validation threshold selection (Zero test leakage!)
        val_probs = clf.predict_proba(X_comb[val_mask])[:, 1]
        prec, rec, thr_arr = precision_recall_curve(y[val_mask], val_probs)
        f1_arr = 2 * prec * rec / (prec + rec + 1e-12)
        best_idx = int(np.nanargmax(f1_arr[:-1])) if len(thr_arr) else 0
        thr = float(thr_arr[best_idx]) if len(thr_arr) else 0.5

        # Test evaluation at fixed validation threshold
        test_probs = clf.predict_proba(X_comb[test_mask])[:, 1]
        y_test = y[test_mask]
        y_pred = (test_probs >= thr).astype(int)

        roc_auc = float(roc_auc_score(y_test, test_probs))
        pr_auc = float(average_precision_score(y_test, test_probs))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        p, r, _, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

        record = {
            "seed": seed,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "f1": f1,
            "precision": float(p),
            "recall": float(r),
            "threshold": thr,
        }
        results.append(record)
        print(f"| {seed:>6} | {pr_auc:>8.4f} | {roc_auc:>8.4f} | {f1:>8.4f} | {p*100:>9.2f}% | {r*100:>7.2f}% | {thr:>10.4f} |")

    pr_aucs = [r["pr_auc"] for r in results]
    roc_aucs = [r["roc_auc"] for r in results]
    f1s = [r["f1"] for r in results]
    precs = [r["precision"] for r in results]
    recs = [r["recall"] for r in results]
    thrs = [r["threshold"] for r in results]

    print("\n" + "="*80)
    print("### EXP_007_HYBRID MULTI-SEED STABILITY SUMMARY (4 Seeds)")
    print("="*80)
    print(f"  PR-AUC:    Mean = {np.mean(pr_aucs):.4f} ± {np.std(pr_aucs):.4f} | Min = {np.min(pr_aucs):.4f} | Max = {np.max(pr_aucs):.4f}")
    print(f"  ROC-AUC:   Mean = {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f} | Min = {np.min(roc_aucs):.4f} | Max = {np.max(roc_aucs):.4f}")
    print(f"  F1:        Mean = {np.mean(f1s):.4f} ± {np.std(f1s):.4f} | Min = {np.min(f1s):.4f} | Max = {np.max(f1s):.4f}")
    print(f"  Precision: Mean = {np.mean(precs)*100:.2f}% ± {np.std(precs)*100:.2f}% | Min = {np.min(precs)*100:.2f}% | Max = {np.max(precs)*100:.2f}%")
    print(f"  Recall:    Mean = {np.mean(recs)*100:.2f}% ± {np.std(recs)*100:.2f}% | Min = {np.min(recs)*100:.2f}% | Max = {np.max(recs)*100:.2f}%")
    print(f"  Threshold: Mean = {np.mean(thrs):.4f} ± {np.std(thrs):.4f}")
    print(f"  Runtime:   Total execution time: {time.time()-t0:.1f}s")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
