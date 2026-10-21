"""
Step 7: Threshold Curve Analysis for EXP_007_HYBRID.
Evaluates Precision, Recall, F1, and Predicted Positive Count across thresholds
0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.9184, 0.95.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from sklearn.metrics import precision_recall_fscore_support, f1_score

import config
from experiments.ml_improvement.runner import AdvancedGraphSAGE

def main():
    data = torch.load(config.GRAPH_PT, weights_only=False)
    x_29 = torch.load("experiments/ml_improvement/extended_features.pt", weights_only=False)
    undir_edges = to_undirected(data.edge_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load GNN
    ckpt_gnn = torch.load("experiments/ml_improvement/checkpoints/EXP_003_FEAT29_UNDIR.pt", weights_only=False)
    gnn = AdvancedGraphSAGE(29, 128, 3, 0.2, use_residual=False, norm_type="batch").to(device)
    gnn.load_state_dict(ckpt_gnn["model_state"])
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

    X_comb = np.hstack([x_29.numpy(), emb])
    y_test = data.y[data.test_mask].numpy()
    test_mask = data.test_mask.numpy()

    # Load XGBoost model from EXP_007 checkpoint
    ckpt_xgb = torch.load("experiments/ml_improvement/checkpoints/EXP_007_HYBRID_XGB.pt", weights_only=False)
    clf = ckpt_xgb["xgb_model"]
    op_thr = float(ckpt_xgb.get("threshold", 0.9184))

    test_probs = clf.predict_proba(X_comb[test_mask])[:, 1]

    thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, op_thr, 0.95]

    print("\n" + "="*85)
    print("### EXP_007_HYBRID THRESHOLD TRADEOFF ANALYSIS (Test Split: 77,264 Accounts)")
    print("="*85)
    print(f"| {'Threshold':>10} | {'Precision':>10} | {'Recall':>9} | {'F1':>8} | {'Predicted Pos':>14} | {'False Pos':>10} |")
    print(f"|:{'-'*10}:|-{'-'*10}:|-{'-'*9}:|-{'-'*8}:|-{'-'*14}:|-{'-'*10}:|")

    for thr in thresholds:
        preds = (test_probs >= thr).astype(int)
        p, r, _, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        n_pos = preds.sum()
        fp = int(n_pos - (preds & y_test).sum())
        marker = " <-- (Val-Tuned Operating Point)" if abs(thr - op_thr) < 1e-4 else ""
        print(f"| {thr:>10.4f} | {p*100:>9.2f}% | {r*100:>8.2f}% | {f1:>8.4f} | {n_pos:>14,d} | {fp:>10,d} |{marker}")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
