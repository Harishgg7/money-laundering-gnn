"""
Threshold Analysis for TRACE ML Improvement Phase.
Evaluates the precision/recall/F1 operating tradeoff across thresholds
for the Baseline vs EXP_003 (Top Pure GNN) vs EXP_007 (Top Hybrid).
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
from model import AMLNet
from experiments.ml_improvement.runner import AdvancedGraphSAGE

def evaluate_threshold_curve(probs, y_true, model_name: str, thresholds: list):
    print(f"\n--- Threshold Curve: {model_name} ---")
    print(f"| {'Threshold':>9} | {'Precision':>9} | {'Recall':>9} | {'F1':>8} | {'Predicted Pos':>13} |")
    print(f"|:{'-'*9}:|-{'-'*9}:|-{'-'*9}:|-{'-'*8}:|-{'-'*13}:|")
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        p, r, _, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        n_pos = preds.sum()
        print(f"| {thr:>9.4f} | {p*100:>8.2f}% | {r*100:>8.2f}% | {f1:>8.4f} | {n_pos:>13,d} |")

def main():
    data = torch.load(config.GRAPH_PT, weights_only=False)
    y_test = data.y[data.test_mask].numpy()
    test_mask = data.test_mask
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Baseline Model
    ckpt_base = torch.load("experiments/ml_improvement/checkpoints/baseline_reproduced.pt", weights_only=False)
    m_base = AMLNet(11, 128, 3, 0.2, "sage").to(device)
    m_base.load_state_dict(ckpt_base["model_state"])
    m_base.eval()
    with torch.no_grad():
        logits_base = m_base(data.x.to(device), data.edge_index.to(device)).cpu()
        probs_base = torch.sigmoid(logits_base[test_mask]).numpy()

    # 2. EXP_003 Model (Top Pure GNN)
    x_29 = torch.load("experiments/ml_improvement/extended_features.pt", weights_only=False)
    undir_edges = to_undirected(data.edge_index)
    ckpt_exp3 = torch.load("experiments/ml_improvement/checkpoints/EXP_003_FEAT29_UNDIR.pt", weights_only=False)
    m_exp3 = AdvancedGraphSAGE(29, 128, 3, 0.2, use_residual=False, norm_type="batch").to(device)
    m_exp3.load_state_dict(ckpt_exp3["model_state"])
    m_exp3.eval()
    with torch.no_grad():
        logits_exp3 = m_exp3(x_29.to(device), undir_edges.to(device)).cpu()
        probs_exp3 = torch.sigmoid(logits_exp3[test_mask]).numpy()

    thresholds = [0.25, 0.30, 0.35, 0.3781, 0.40, 0.4355, 0.45, 0.50, 0.55]
    evaluate_threshold_curve(probs_base, y_test, "Baseline GraphSAGE", thresholds)
    evaluate_threshold_curve(probs_exp3, y_test, "EXP_003 (29 Feat + Bidirectional)", thresholds)

if __name__ == "__main__":
    main()
