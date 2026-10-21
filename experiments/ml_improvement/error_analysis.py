"""
Step 8: Forensic Error Analysis for EXP_007_HYBRID.
Analyzes feature distributions of True Positives, False Positives,
True Negatives, and False Negatives to understand failure modes.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

import config
from experiments.ml_improvement.runner import AdvancedGraphSAGE

def main():
    data = torch.load(config.GRAPH_PT, weights_only=False)
    x_29 = torch.load("experiments/ml_improvement/extended_features.pt", weights_only=False)
    y_test = data.y[data.test_mask].numpy()
    test_mask = data.test_mask.numpy()

    # Load trained model and embeddings
    from torch_geometric.utils import to_undirected
    import torch.nn.functional as F
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_gnn = torch.load("experiments/ml_improvement/checkpoints/EXP_003_FEAT29_UNDIR.pt", weights_only=False)
    gnn = AdvancedGraphSAGE(29, 128, 3, 0.2, use_residual=False, norm_type="batch").to(device)
    gnn.load_state_dict(ckpt_gnn["model_state"])
    gnn.eval()

    with torch.no_grad():
        x_dev = x_29.to(device)
        undir_edges = to_undirected(data.edge_index).to(device)
        h = x_dev
        for block in gnn.blocks:
            h = F.relu(block["norm"](block["conv"](h, undir_edges)))
        emb = h.cpu().numpy()

    X_comb = np.hstack([x_29.numpy(), emb])

    ckpt_xgb = torch.load("experiments/ml_improvement/checkpoints/EXP_007_HYBRID_XGB.pt", weights_only=False)
    clf = ckpt_xgb["xgb_model"]
    thr = ckpt_xgb["threshold"] # 0.9184

    test_probs = clf.predict_proba(X_comb[test_mask])[:, 1]
    preds = (test_probs >= thr).astype(int)

    tp_mask = (preds == 1) & (y_test == 1)
    fp_mask = (preds == 1) & (y_test == 0)
    tn_mask = (preds == 0) & (y_test == 0)
    fn_mask = (preds == 0) & (y_test == 1)

    print("\n" + "="*80)
    print("### EXP_007_HYBRID CONFUSION MATRIX & ERROR PROFILING")
    print("="*80)
    print(f"Total Test Accounts: {len(y_test):,}")
    print(f"  True Positives  (TP): {tp_mask.sum():,} ({100*tp_mask.sum()/y_test.sum():.2f}% of illicit accounts caught)")
    print(f"  False Positives (FP): {fp_mask.sum():,} ({100*fp_mask.sum()/(y_test==0).sum():.2f}% false alarm rate on licit)")
    print(f"  False Negatives (FN): {fn_mask.sum():,} ({100*fn_mask.sum()/y_test.sum():.2f}% missed illicit accounts)")
    print(f"  True Negatives  (TN): {tn_mask.sum():,}")
    print(f"  Precision: {tp_mask.sum() / max(1, preds.sum())*100:.2f}% | Recall: {tp_mask.sum() / y_test.sum()*100:.2f}%")

    # Feature comparison across cohorts
    X_test_raw = x_29.numpy()[test_mask]
    feature_labels = [
        "Out-Degree", "In-Degree", "Total Sent", "Total Recv",
        "Mean Sent", "Mean Recv", "Uniq Out Counterparties", "Uniq In Counterparties"
    ]

    print("\nCohort Feature Profiles (Mean Standardized Values):")
    print(f"| {'Feature':<28} | {'TP (Caught)':>12} | {'FP (False Alarm)':>16} | {'FN (Missed)':>12} | {'TN (Clean)':>12} |")
    print(f"|:{'-'*28}-|-{'-'*12}:|-{'-'*16}:|-{'-'*12}:|-{'-'*12}:|")
    for i in range(8):
        f_name = feature_labels[i]
        tp_val = X_test_raw[tp_mask, i].mean() if tp_mask.sum() > 0 else 0
        fp_val = X_test_raw[fp_mask, i].mean() if fp_mask.sum() > 0 else 0
        fn_val = X_test_raw[fn_mask, i].mean() if fn_mask.sum() > 0 else 0
        tn_val = X_test_raw[tn_mask, i].mean() if tn_mask.sum() > 0 else 0
        print(f"| {f_name:<28} | {tp_val:>12.3f} | {fp_val:>16.3f} | {fn_val:>12.3f} | {tn_val:>12.3f} |")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
