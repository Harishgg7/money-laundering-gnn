"""
Step 10: Temporal Validation for TRACE ML Improvement Phase.
Constructs chronological train (earliest 70%), val (middle 15%), and test (latest 15%)
based on account transaction timestamps, comparing temporal PR-AUC against stratified benchmark.
"""
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve, precision_recall_fscore_support

import config

def main():
    print("=== STEP 10: TEMPORAL SPLIT VALIDATION ===")
    t0 = time.time()
    
    # Load raw transactions and node mapping to compute chronological order
    df_nodes = pd.read_parquet(config.NODE_MAP_PARQUET)
    n = len(df_nodes)
    node_map = pd.Series(df_nodes["node_id"].values, index=df_nodes["account_key"].values)

    df = pd.read_csv(
        config.RAW_TRANS_CSV,
        dtype={"From Bank": str, "Account": str, "To Bank": str, "Account.1": str},
        usecols=["From Bank", "Account", "To Bank", "Account.1", "Timestamp"]
    )
    df["src_key"] = df["From Bank"].str.cat(df["Account"], sep="-")
    df["dst_key"] = df["To Bank"].str.cat(df["Account.1"], sep="-")
    src = node_map.loc[df["src_key"]].to_numpy(dtype=np.int64)
    dst = node_map.loc[df["dst_key"]].to_numpy(dtype=np.int64)
    ts_sec = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M").astype("int64") // 10**9

    s_ts = pd.Series(ts_sec.to_numpy())
    src_max = s_ts.groupby(src).max().reindex(range(n), fill_value=0).to_numpy()
    dst_max = s_ts.groupby(dst).max().reindex(range(n), fill_value=0).to_numpy()
    last_active = np.maximum(src_max, dst_max)

    order = np.argsort(last_active)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]

    data = torch.load(config.GRAPH_PT, weights_only=False)
    y = data.y.numpy()

    print(f"[temporal] Train nodes: {len(train_idx):,} (Pos: {y[train_idx].sum():,}, {100*y[train_idx].mean():.2f}%)")
    print(f"[temporal] Val nodes:   {len(val_idx):,} (Pos: {y[val_idx].sum():,}, {100*y[val_idx].mean():.2f}%)")
    print(f"[temporal] Test nodes:  {len(test_idx):,} (Pos: {y[test_idx].sum():,}, {100*y[test_idx].mean():.2f}%)")

    # Load extended features
    x_29 = torch.load("experiments/ml_improvement/extended_features.pt", weights_only=False).numpy()

    # Load EXP_007 checkpoint
    ckpt_xgb = torch.load("experiments/ml_improvement/checkpoints/EXP_007_HYBRID_XGB.pt", weights_only=False)
    clf = ckpt_xgb["xgb_model"]

    # Re-extract embeddings on whole graph
    ckpt_gnn = torch.load("experiments/ml_improvement/checkpoints/EXP_003_FEAT29_UNDIR.pt", weights_only=False)
    from torch_geometric.utils import to_undirected
    from experiments.ml_improvement.runner import AdvancedGraphSAGE
    import torch.nn.functional as F
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn = AdvancedGraphSAGE(29, 128, 3, 0.2, use_residual=False, norm_type="batch").to(device)
    gnn.load_state_dict(ckpt_gnn["model_state"])
    gnn.eval()
    with torch.no_grad():
        undir_edges = to_undirected(data.edge_index).to(device)
        x_dev = torch.from_numpy(x_29).to(device)
        h = x_dev
        for block in gnn.blocks:
            h = F.relu(block["norm"](block["conv"](h, undir_edges)))
        emb = h.cpu().numpy()

    X_hybrid = np.hstack([x_29, emb])
    y_train = y[train_idx]
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw = float(n_neg / max(1, n_pos)) * 0.7

    print("[temporal] Training hybrid model on chronological training window...")
    clf_temporal = XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.08,
        scale_pos_weight=spw, subsample=0.85, colsample_bytree=0.85,
        eval_metric="aucpr", tree_method="hist",
        device="cuda" if torch.cuda.is_available() else "cpu", random_state=42
    )
    clf_temporal.fit(X_hybrid[train_idx], y_train)

    val_probs = clf_temporal.predict_proba(X_hybrid[val_idx])[:, 1]
    prec, rec, thr_arr = precision_recall_curve(y[val_idx], val_probs)
    f1_arr = 2 * prec * rec / (prec + rec + 1e-12)
    best_idx = int(np.nanargmax(f1_arr[:-1])) if len(thr_arr) else 0
    thr = float(thr_arr[best_idx]) if len(thr_arr) else 0.5

    test_probs = clf_temporal.predict_proba(X_hybrid[test_idx])[:, 1]
    y_test = y[test_idx]
    preds = (test_probs >= thr).astype(int)

    pr_auc_temp = float(average_precision_score(y_test, test_probs))
    roc_auc_temp = float(roc_auc_score(y_test, test_probs))
    f1_temp = float(f1_score(y_test, preds, zero_division=0))
    p_temp, r_temp, _, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)

    print("\n" + "="*85)
    print("### TEMPORAL EVALUATION (CHRONOLOGICAL SPLIT)")
    print("="*85)
    print(f"| Evaluation Split | Prevalence | PR-AUC | ROC-AUC | F1-Score | Precision | Recall |")
    print(f"|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    print(f"| Stratified Benchmark (EXP_007) | 1.23% | 0.4569 | 0.9412 | 0.4886 | 71.89% | 37.00% |")
    print(f"| Chronological Split (Temporal) | {100*y_test.mean():.2f}% | {pr_auc_temp:.4f} | {roc_auc_temp:.4f} | {f1_temp:.4f} | {p_temp*100:.2f}% | {r_temp*100:.2f}% |")
    print("="*85)
    print(f"NOTE: Test prevalence differs between splits (Chronological: {100*y_test.mean():.2f}% vs Stratified: 1.23%).")
    print("Direct PR-AUC delta comparison across splits with different prevalence is mathematically non-comparable.")
    print("Temporal generalization claim is excluded from production promotion criteria.")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
