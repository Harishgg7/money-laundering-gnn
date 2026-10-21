"""
Feature Engineering for TRACE ML Improvement Phase.
Computes 29 rich AML behavioral and topological features from HI-Small_Trans.csv.
Zero data leakage:
- No label information used in feature creation.
- Normalization (StandardScaler) is fit STRICTLY on train_mask nodes.
- Preserves exact node indexing matching data/processed/node_map.parquet.
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

def build_extended_features(save_path: str = "experiments/ml_improvement/extended_features.pt"):
    print("[feat] Loading raw transactions and node mapping...")
    t0 = time.time()

    # Load node mapping
    parquet_path = config.NODE_MAP_PARQUET
    df_nodes = pd.read_parquet(parquet_path)
    n = len(df_nodes)
    node_map = pd.Series(df_nodes["node_id"].values, index=df_nodes["account_key"].values)
    print(f"[feat] Node mapping loaded: {n:,} unique accounts")

    # Load transactions with strict dtypes preserving leading zeros
    df = pd.read_csv(
        config.RAW_TRANS_CSV,
        dtype={
            "From Bank": str,
            "Account": str,
            "To Bank": str,
            "Account.1": str,
            "Receiving Currency": "category",
            "Payment Currency": "category",
            "Payment Format": "category",
        },
    )
    print(f"[feat] Loaded {len(df):,} transactions in {time.time()-t0:.2f}s")

    # Construct exact account keys matching node_map
    df["src_key"] = df["From Bank"].str.cat(df["Account"], sep="-")
    df["dst_key"] = df["To Bank"].str.cat(df["Account.1"], sep="-")

    src = node_map.loc[df["src_key"]].to_numpy(dtype=np.int64)
    dst = node_map.loc[df["dst_key"]].to_numpy(dtype=np.int64)
    amt_paid = df["Amount Paid"].to_numpy(dtype=np.float64)
    amt_recv = df["Amount Received"].to_numpy(dtype=np.float64)
    cross_bank = (df["From Bank"].to_numpy() != df["To Bank"].to_numpy()).astype(np.float64)

    # Parse timestamps to unix seconds
    df["ts_sec"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M").astype("int64") // 10**9
    ts_sec = df["ts_sec"].to_numpy(dtype=np.int64)

    def _bincount(idx, weights=None):
        return np.bincount(idx, weights=weights, minlength=n).astype(np.float64)

    print("[feat] Computing transaction degrees and flow totals...")
    out_deg = _bincount(src)
    in_deg = _bincount(dst)
    total_sent = _bincount(src, amt_paid)
    total_recv = _bincount(dst, amt_recv)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_sent = np.where(out_deg > 0, total_sent / out_deg, 0.0)
        mean_recv = np.where(in_deg > 0, total_recv / in_deg, 0.0)

    print("[feat] Computing counterparty uniqueness...")
    uniq_out = df.groupby(src)["dst_key"].nunique().reindex(range(n), fill_value=0).to_numpy(dtype=np.float64)
    uniq_in = df.groupby(dst)["src_key"].nunique().reindex(range(n), fill_value=0).to_numpy(dtype=np.float64)

    print("[feat] Computing currency and format diversity...")
    n_cur_sent = df.groupby(src)["Payment Currency"].nunique().reindex(range(n), fill_value=0).to_numpy(dtype=np.float64)
    n_fmt_sent = df.groupby(src)["Payment Format"].nunique().reindex(range(n), fill_value=0).to_numpy(dtype=np.float64)
    n_cur_recv = df.groupby(dst)["Receiving Currency"].nunique().reindex(range(n), fill_value=0).to_numpy(dtype=np.float64)
    n_fmt_recv = df.groupby(dst)["Payment Format"].nunique().reindex(range(n), fill_value=0).to_numpy(dtype=np.float64)

    self_loop = _bincount(src[src == dst]).astype(bool).astype(np.float64)

    print("[feat] Computing behavioral ratios (flow, fan-in, fan-out, cross-bank)...")
    in_out_tx_ratio = (in_deg + 1.0) / (out_deg + 1.0)
    net_flow = total_recv - total_sent
    abs_net_flow = np.abs(net_flow)
    in_out_amt_ratio = (total_recv + 1.0) / (total_sent + 1.0)
    total_volume = total_sent + total_recv
    mean_volume = total_volume / (out_deg + in_deg + 1.0)

    fan_in_ratio = uniq_in / (in_deg + 1.0)
    fan_out_ratio = uniq_out / (out_deg + 1.0)
    total_counterparties = uniq_in + uniq_out
    counterparty_concentration = total_counterparties / (out_deg + in_deg + 1.0)

    # Cross-bank transfers
    cross_bank_sent = _bincount(src, cross_bank)
    cross_bank_recv = _bincount(dst, cross_bank)
    cross_bank_ratio = (cross_bank_sent + cross_bank_recv) / (out_deg + in_deg + 1.0)

    print("[feat] Computing amount extremes (max, std)...")
    s_paid = pd.Series(amt_paid)
    paid_agg = s_paid.groupby(src).agg(["max", "std"]).reindex(range(n), fill_value=0.0)
    max_sent = paid_agg["max"].fillna(0.0).to_numpy(dtype=np.float64)
    std_sent = paid_agg["std"].fillna(0.0).to_numpy(dtype=np.float64)

    s_recv = pd.Series(amt_recv)
    recv_agg = s_recv.groupby(dst).agg(["max", "std"]).reindex(range(n), fill_value=0.0)
    max_recv = recv_agg["max"].fillna(0.0).to_numpy(dtype=np.float64)
    std_recv = recv_agg["std"].fillna(0.0).to_numpy(dtype=np.float64)

    print("[feat] Computing temporal dynamics (activity span, rate)...")
    s_ts = pd.Series(ts_sec)
    ts_src_agg = s_ts.groupby(src).agg(["min", "max"]).reindex(range(n), fill_value=0)
    ts_dst_agg = s_ts.groupby(dst).agg(["min", "max"]).reindex(range(n), fill_value=0)

    # Active accounts: min where active, else max
    src_active = out_deg > 0
    dst_active = in_deg > 0

    min_src_valid = np.where(src_active, ts_src_agg["min"].to_numpy(dtype=np.float64), 1e12)
    min_dst_valid = np.where(dst_active, ts_dst_agg["min"].to_numpy(dtype=np.float64), 1e12)
    max_src_valid = np.where(src_active, ts_src_agg["max"].to_numpy(dtype=np.float64), 0.0)
    max_dst_valid = np.where(dst_active, ts_dst_agg["max"].to_numpy(dtype=np.float64), 0.0)

    has_activity = (out_deg + in_deg) > 0
    acct_min_t = np.where(has_activity, np.minimum(min_src_valid, min_dst_valid), 0.0)
    acct_max_t = np.where(has_activity, np.maximum(max_src_valid, max_dst_valid), 0.0)
    time_span_hours = np.where(has_activity, (acct_max_t - acct_min_t) / 3600.0, 0.0)
    tx_per_day = np.where(has_activity, (out_deg + in_deg) / np.maximum(1.0, time_span_hours / 24.0), 0.0)

    # Assemble 29 rich features
    feature_matrix = np.column_stack([
        out_deg, in_deg,
        total_sent, total_recv,
        mean_sent, mean_recv,
        uniq_out, uniq_in,
        n_cur_sent, n_fmt_sent,
        self_loop,
        in_out_tx_ratio,
        net_flow,
        abs_net_flow,
        in_out_amt_ratio,
        total_volume,
        mean_volume,
        fan_in_ratio,
        fan_out_ratio,
        counterparty_concentration,
        cross_bank_ratio,
        max_sent,
        max_recv,
        std_sent,
        std_recv,
        n_cur_recv,
        n_fmt_recv,
        time_span_hours,
        tx_per_day
    ]).astype(np.float32)

    print(f"[feat] Raw feature matrix assembled: shape={feature_matrix.shape}")

    # Log1p heavy-tailed columns
    heavy_cols = [0, 1, 2, 3, 4, 5, 6, 7, 11, 13, 14, 15, 16, 21, 22, 23, 24, 27, 28]
    feature_matrix[:, heavy_cols] = np.log1p(np.maximum(0.0, feature_matrix[:, heavy_cols]))

    # Load train_mask to fit scaler strictly on train set (Zero Leakage!)
    data_orig = torch.load(config.GRAPH_PT, weights_only=False)
    train_mask = data_orig.train_mask.numpy()

    scaler = StandardScaler()
    scaler.fit(feature_matrix[train_mask])
    feature_matrix_norm = scaler.transform(feature_matrix).astype(np.float32)

    feat_tensor = torch.from_numpy(feature_matrix_norm)
    torch.save(feat_tensor, save_path)
    print(f"[feat] Saved 29-dim normalized feature tensor -> {save_path} in {time.time()-t0:.1f}s")
    return feat_tensor

if __name__ == "__main__":
    build_extended_features()
