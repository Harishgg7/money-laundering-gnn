"""
Experiment 007: GNN Embeddings + Tabular Features with Tuned Classifier / Hybrid.
Tests whether combining the 29 AML behavioral features with the 128-dim
GraphSAGE embeddings from EXP_003 pushes PR-AUC into the 0.43-0.45+ range.
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

import config
from model import AMLNet
from utils import best_threshold, evaluate, format_metrics
from experiments.ml_improvement.runner import AdvancedGraphSAGE
from experiments.ml_improvement.tracker import log_experiment, print_summary_table

def main():
    print("=== EXPERIMENT 007: GNN EMBEDDINGS + GRADIENT BOOSTED HYBRID ===")
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data and 29 features
    data = torch.load(config.GRAPH_PT, weights_only=False)
    x_29 = torch.load("experiments/ml_improvement/extended_features.pt", weights_only=False)
    undir_edges = to_undirected(data.edge_index)

    # Load best GNN model checkpoint (EXP_003)
    ckpt_path = "experiments/ml_improvement/checkpoints/EXP_003_FEAT29_UNDIR.pt"
    ckpt = torch.load(ckpt_path, weights_only=False)
    
    gnn = AdvancedGraphSAGE(
        in_channels=29,
        hidden_channels=128,
        num_layers=3,
        dropout=0.2,
        use_residual=False,
        norm_type="batch"
    ).to(device)
    gnn.load_state_dict(ckpt["model_state"])
    gnn.eval()

    # Extract penultimate embeddings [num_nodes, 128]
    print("[hybrid] Extracting GNN embeddings...")
    with torch.no_grad():
        x_dev = x_29.to(device)
        edges_dev = undir_edges.to(device)
        # Pass through conv blocks
        h = x_dev
        for block in gnn.blocks:
            conv = block["conv"]
            norm = block["norm"]
            h = conv(h, edges_dev)
            h = norm(h)
            h = F.relu(h)
        emb = h.cpu().numpy()

    # Combine 29 raw features + 128 GNN embeddings = 157-dim feature representation
    X_comb = np.hstack([x_29.numpy(), emb])
    y = data.y.numpy()
    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()

    y_train = y[train_mask]
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw = float(n_neg / max(1, n_pos))

    print(f"[hybrid] Training XGBoost on {X_comb.shape[1]}-dim hybrid representations (n_samples={len(y_train):,})...")
    clf = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        scale_pos_weight=spw * 0.7, # slightly tuned for PR-AUC
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="aucpr",
        tree_method="hist",
        device="cuda" if torch.cuda.is_available() else "cpu",
        random_state=42,
    )
    clf.fit(X_comb[train_mask], y_train)

    print("[hybrid] Evaluating on validation set for threshold...")
    val_probs = clf.predict_proba(X_comb[val_mask])[:, 1]
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr_arr = precision_recall_curve(y[val_mask], val_probs)
    f1_arr = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = int(np.nanargmax(f1_arr[:-1])) if len(thr_arr) else 0
    thr = float(thr_arr[best_idx]) if len(thr_arr) else 0.5

    print(f"[hybrid] Optimal validation threshold: {thr:.4f}")
    print("[hybrid] Evaluating on test set...")
    test_probs = clf.predict_proba(X_comb[test_mask])[:, 1]
    y_test = y[test_mask]
    y_pred = (test_probs >= thr).astype(int)

    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support
    roc_auc = float(roc_auc_score(y_test, test_probs))
    pr_auc = float(average_precision_score(y_test, test_probs))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    p, r, _, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    test_m = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": float(p),
        "recall": float(r),
        "threshold": thr,
    }

    training_time = time.time() - t0
    pr_auc = test_m["pr_auc"]
    status = "Stretch Achieved" if pr_auc >= 0.45 else ("Primary Candidate" if pr_auc >= 0.40 else "Improved")

    record = {
        "experiment_id": "EXP_007_HYBRID_XGB",
        "description": "GNN (EXP_003) 128-dim embeddings + 29 AML features + XGBoost",
        "features": "29 AML features + 128 GNN embeddings (157 dims)",
        "graph_config": "Bidirectional 2.03M edges (undirected)",
        "model_config": "GraphSAGE(128) -> XGBoost(150 trees, lr=0.08, max_depth=6)",
        "loss_config": "Focal Loss (GNN) + LogLoss(scale_pos_weight)",
        "pr_auc": f"{pr_auc:.4f}",
        "precision": f"{test_m['precision']:.4f}",
        "recall": f"{test_m['recall']:.4f}",
        "f1": f"{test_m['f1']:.4f}",
        "roc_auc": f"{test_m['roc_auc']:.4f}",
        "threshold": f"{thr:.4f}",
        "training_time": f"{training_time:.1f}s",
        "status": status,
    }
    log_experiment(record)

    # Save predictions / model checkpoint
    ckpt_path = "experiments/ml_improvement/checkpoints/EXP_007_HYBRID_XGB.pt"
    torch.save({
        "xgb_model": clf,
        "gnn_checkpoint": "EXP_003_FEAT29_UNDIR.pt",
        "threshold": thr,
        "test_metrics": test_m,
    }, ckpt_path)

    print(f"\n[EXP_007 COMPLETE] PR-AUC={pr_auc:.4f} | F1={test_m['f1']:.4f} | Prec={test_m['precision']:.4f} | Rec={test_m['recall']:.4f} | Status={status}")
    print_summary_table()

if __name__ == "__main__":
    main()
