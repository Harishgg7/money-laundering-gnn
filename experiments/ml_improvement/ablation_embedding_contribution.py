"""
Step 9: Graph Embedding Ablation Study.
Proves whether the 128-dim GraphSAGE embeddings contribute genuine predictive signal
by comparing:
  A: 29 engineered features alone -> XGBoost
  B: 128-dim GraphSAGE embeddings alone -> XGBoost
  C: 29 engineered features + 128-dim GraphSAGE embeddings -> XGBoost (Hybrid)
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

def evaluate_subset(X, y, train_mask, val_mask, test_mask, name: str, spw: float):
    t0 = time.time()
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
        random_state=42,
    )
    clf.fit(X[train_mask], y[train_mask])

    # Validation threshold
    val_probs = clf.predict_proba(X[val_mask])[:, 1]
    prec, rec, thr_arr = precision_recall_curve(y[val_mask], val_probs)
    f1_arr = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = int(np.nanargmax(f1_arr[:-1])) if len(thr_arr) else 0
    thr = float(thr_arr[best_idx]) if len(thr_arr) else 0.5

    # Test evaluation
    test_probs = clf.predict_proba(X[test_mask])[:, 1]
    y_test = y[test_mask]
    preds = (test_probs >= thr).astype(int)

    pr_auc = float(average_precision_score(y_test, test_probs))
    roc_auc = float(roc_auc_score(y_test, test_probs))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    p, r, _, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)

    elapsed = time.time() - t0
    print(f"| {name:<35} | {pr_auc:>7.4f} | {roc_auc:>7.4f} | {f1:>6.4f} | {p*100:>6.2f}% | {r*100:>6.2f}% | {thr:>6.4f} | {elapsed:>5.1f}s |")
    return {"pr_auc": pr_auc, "roc_auc": roc_auc, "f1": f1, "precision": p, "recall": r, "threshold": thr}

def main():
    print("=== STEP 9: GRAPH EMBEDDING ABLATION STUDY ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(config.GRAPH_PT, weights_only=False)
    x_29 = torch.load("experiments/ml_improvement/extended_features.pt", weights_only=False)
    undir_edges = to_undirected(data.edge_index)

    # Extract 128-dim GNN embeddings
    ckpt = torch.load("experiments/ml_improvement/checkpoints/EXP_003_FEAT29_UNDIR.pt", weights_only=False)
    gnn = AdvancedGraphSAGE(
        in_channels=29, hidden_channels=128, num_layers=3, dropout=0.2,
        use_residual=False, norm_type="batch"
    ).to(device)
    gnn.load_state_dict(ckpt["model_state"])
    gnn.eval()

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

    X_feat = x_29.numpy()
    X_emb = emb
    X_hybrid = np.hstack([X_feat, X_emb])

    y = data.y.numpy()
    train_m = data.train_mask.numpy()
    val_m = data.val_mask.numpy()
    test_m = data.test_mask.numpy()

    y_train = y[train_m]
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw = float(n_neg / max(1, n_pos)) * 0.7

    print(f"\n| {'Configuration':<35} | {'PR-AUC':>7} | {'ROC-AUC':>7} | {'F1':>6} | {'Prec':>7} | {'Recall':>7} | {'Thr':>6} | {'Time':>6} |")
    print(f"|:{'-'*35}-|-{'-'*7}:|-{'-'*7}:|-{'-'*6}:|-{'-'*7}:|-{'-'*7}:|-{'-'*6}:|-{'-'*6}:|")

    evaluate_subset(X_feat, y, train_m, val_m, test_m, "A: 29 Features Only (No Graph)", spw)
    evaluate_subset(X_emb, y, train_m, val_m, test_m, "B: 128-dim GNN Embeddings Only", spw)
    evaluate_subset(X_hybrid, y, train_m, val_m, test_m, "C: 29 Feat + GNN Embeddings (Hybrid)", spw)
    print("="*95 + "\n")

if __name__ == "__main__":
    main()
