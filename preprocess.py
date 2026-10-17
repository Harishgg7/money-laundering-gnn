"""
Step 1 & 2 of the pipeline: Data Preprocessing + PyG Dataset Creation.

Reads the raw IBM AML transaction log and builds a directed account graph:

    Nodes  = accounts  (unique (Bank, Account) pairs)
    Edges  = aggregated money flows sender -> receiver
    x      = engineered per-account behavioural features
    y      = 1 if the account is involved in ANY laundering transaction

Parallel transactions between the same ordered pair of accounts are aggregated
into a single edge carrying [transaction_count, total_amount] as edge_attr, so
the 5M raw transactions collapse to a graph the GNN can train on efficiently.

Output: a single PyTorch Geometric `Data` object saved to data/processed/.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

import config


# Column names as pandas reads them (the two "Account" columns become
# "Account" for the sender and "Account.1" for the receiver).
COL_TS = "Timestamp"
COL_FROM_BANK = "From Bank"
COL_FROM_ACCT = "Account"
COL_TO_BANK = "To Bank"
COL_TO_ACCT = "Account.1"
COL_AMT_RECV = "Amount Received"
COL_AMT_PAID = "Amount Paid"
COL_RECV_CUR = "Receiving Currency"
COL_PAY_CUR = "Payment Currency"
COL_FORMAT = "Payment Format"
COL_LABEL = "Is Laundering"


def load_transactions(path=config.RAW_TRANS_CSV) -> pd.DataFrame:
    """Load the raw transaction CSV with sensible dtypes and parsed timestamp."""
    print(f"[load] reading {path} ...")
    df = pd.read_csv(
        path,
        dtype={
            COL_FROM_BANK: str,
            COL_FROM_ACCT: str,
            COL_TO_BANK: str,
            COL_TO_ACCT: str,
            COL_RECV_CUR: "category",
            COL_PAY_CUR: "category",
            COL_FORMAT: "category",
            COL_LABEL: np.int8,
        },
    )
    # Parse timestamp into unix seconds for time-delta and temporal splitting
    df["ts_sec"] = pd.to_datetime(df[COL_TS], format="%Y/%m/%d %H:%M").astype("int64") // 10**9

    # Build a globally-unique account key: "<bank>-<account>".
    df["src_key"] = df[COL_FROM_BANK].str.cat(df[COL_FROM_ACCT], sep="-")
    df["dst_key"] = df[COL_TO_BANK].str.cat(df[COL_TO_ACCT], sep="-")
    print(f"[load] {len(df):,} transactions, "
          f"{int(df[COL_LABEL].sum()):,} laundering "
          f"({100 * df[COL_LABEL].mean():.4f}%)")
    return df


def build_node_index(df: pd.DataFrame):
    """Map every unique account key to an integer node id."""
    keys = pd.Index(pd.unique(pd.concat([df["src_key"], df["dst_key"]],
                                        ignore_index=True)))
    node_map = pd.Series(np.arange(len(keys), dtype=np.int64), index=keys)
    print(f"[nodes] {len(node_map):,} unique accounts")
    return node_map


def build_node_features(df: pd.DataFrame, node_map: pd.Series):
    """Engineer per-account behavioural features and node labels.

    Features (11): out/in degree, total & mean amount sent/received,
    unique out/in counterparties, #currencies, #payment formats, self-loop flag.
    """
    n = len(node_map)
    src = node_map.loc[df["src_key"]].to_numpy()
    dst = node_map.loc[df["dst_key"]].to_numpy()
    amt_paid = df[COL_AMT_PAID].to_numpy(dtype=np.float64)
    amt_recv = df[COL_AMT_RECV].to_numpy(dtype=np.float64)

    def _bincount(idx, weights=None):
        return np.bincount(idx, weights=weights, minlength=n)

    out_deg = _bincount(src)
    in_deg = _bincount(dst)
    total_sent = _bincount(src, amt_paid)
    total_recv = _bincount(dst, amt_recv)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_sent = np.where(out_deg > 0, total_sent / out_deg, 0.0)
        mean_recv = np.where(in_deg > 0, total_recv / in_deg, 0.0)

    # Unique counterparties (distinct destinations per source, and vice versa).
    uniq_out = (df.groupby(src)["dst_key"].nunique()
                  .reindex(range(n), fill_value=0).to_numpy())
    uniq_in = (df.groupby(dst)["src_key"].nunique()
                 .reindex(range(n), fill_value=0).to_numpy())

    # Diversity of currencies / formats used by each account (as sender).
    n_cur = (df.groupby(src)[COL_PAY_CUR].nunique()
               .reindex(range(n), fill_value=0).to_numpy())
    n_fmt = (df.groupby(src)[COL_FORMAT].nunique()
               .reindex(range(n), fill_value=0).to_numpy())

    self_loop = _bincount(src[src == dst]).astype(bool).astype(np.float32)

    feats = np.vstack([
        out_deg, in_deg,
        total_sent, total_recv,
        mean_sent, mean_recv,
        uniq_out, uniq_in,
        n_cur, n_fmt,
        self_loop,
    ]).T.astype(np.float32)

    # log1p the heavy-tailed magnitude/count columns, then standardise all.
    heavy = [0, 1, 2, 3, 4, 5, 6, 7]
    feats[:, heavy] = np.log1p(feats[:, heavy])
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    feats = (feats - mean) / std

    # Node label: illicit if involved (either direction) in a laundering txn.
    y = np.zeros(n, dtype=np.int64)
    launder = df[COL_LABEL].to_numpy().astype(bool)
    illicit_nodes = np.unique(np.concatenate([src[launder], dst[launder]]))
    y[illicit_nodes] = 1
    print(f"[nodes] {y.sum():,} illicit accounts "
          f"({100 * y.mean():.4f}%)  |  {feats.shape[1]} features/node")
    return torch.from_numpy(feats), torch.from_numpy(y)


def build_edges(df: pd.DataFrame, node_map: pd.Series):
    """Aggregate parallel transactions into directed edges with 6 aggregated edge attributes:
    1. tx_count (log1p scaled transaction frequency)
    2. total_amount (log1p scaled total payment volume)
    3. mean_amount (log1p scaled average payment volume)
    4. std_amount (log1p scaled std of payment volume, 0 if single tx)
    5. min_time_delta (log1p scaled minimum time delta between consecutive transfers)
    6. unique_formats_ratio (count of distinct payment formats used on this edge)
    """
    src = node_map.loc[df["src_key"]].to_numpy()
    dst = node_map.loc[df["dst_key"]].to_numpy()
    amt = df[COL_AMT_PAID].to_numpy(dtype=np.float64)
    ts = df["ts_sec"].to_numpy(dtype=np.int64)
    fmt = df[COL_FORMAT].cat.codes.to_numpy(dtype=np.int32) if hasattr(df[COL_FORMAT], "cat") else np.zeros(len(df), dtype=np.int32)

    g = pd.DataFrame({"src": src, "dst": dst, "amt": amt, "ts": ts, "fmt": fmt})
    g.sort_values(["src", "dst", "ts"], inplace=True)
    g["ts_diff"] = g.groupby(["src", "dst"], sort=False)["ts"].diff()

    agg = g.groupby(["src", "dst"], sort=False).agg(
        tx_count=("amt", "size"),
        total_amount=("amt", "sum"),
        mean_amount=("amt", "mean"),
        std_amount=("amt", "std"),
        min_time_delta=("ts_diff", "min"),
        unique_formats=("fmt", "nunique")
    ).reset_index()

    agg["std_amount"] = agg["std_amount"].fillna(0.0)
    agg["min_time_delta"] = agg["min_time_delta"].fillna(0.0)

    edge_index = torch.tensor(
        np.vstack([agg["src"].to_numpy(dtype=np.int64), agg["dst"].to_numpy(dtype=np.int64)]),
        dtype=torch.long)

    edge_attr_raw = np.column_stack([
        np.log1p(agg["tx_count"].to_numpy(dtype=np.float32)),
        np.log1p(agg["total_amount"].to_numpy(dtype=np.float32)),
        np.log1p(agg["mean_amount"].to_numpy(dtype=np.float32)),
        np.log1p(agg["std_amount"].to_numpy(dtype=np.float32)),
        np.log1p(agg["min_time_delta"].to_numpy(dtype=np.float32)),
        agg["unique_formats"].to_numpy(dtype=np.float32),
    ])

    print(f"[edges] {edge_index.size(1):,} aggregated edges "
          f"(from {len(df):,} raw transactions) with {edge_attr_raw.shape[1]} features/edge")
    return edge_index, edge_attr_raw


def make_splits(y: torch.Tensor, split_mode: str = "stratified",
                df: pd.DataFrame | None = None,
                node_map: pd.Series | None = None):
    """Generate train/val/test node masks (stratified or temporal)."""
    n = y.numel()

    def _mask(ix):
        m = torch.zeros(n, dtype=torch.bool)
        m[ix] = True
        return m

    if split_mode == "temporal":
        assert df is not None and node_map is not None, "Temporal split requires df and node_map"
        print("[split] using temporal (chronological) split based on account last-active timestamp")
        src = node_map.loc[df["src_key"]].to_numpy()
        dst = node_map.loc[df["dst_key"]].to_numpy()

        src_max = df.groupby(src)["ts_sec"].max().reindex(range(n), fill_value=0).to_numpy()
        dst_max = df.groupby(dst)["ts_sec"].max().reindex(range(n), fill_value=0).to_numpy()
        last_active = np.maximum(src_max, dst_max)

        order = np.argsort(last_active)
        n_train = int(n * config.TRAIN_FRAC)
        n_val = int(n * config.VAL_FRAC)
        train_idx = order[:n_train]
        val_idx = order[n_train:n_train + n_val]
        test_idx = order[n_train + n_val:]
    else:
        print("[split] using stratified node split")
        from sklearn.model_selection import train_test_split
        idx = np.arange(n)
        y_np = y.numpy()
        train_idx, tmp_idx = train_test_split(
            idx, train_size=config.TRAIN_FRAC, stratify=y_np, random_state=config.SEED)
        rel_val = config.VAL_FRAC / (config.VAL_FRAC + config.TEST_FRAC)
        val_idx, test_idx = train_test_split(
            tmp_idx, train_size=rel_val, stratify=y_np[tmp_idx],
            random_state=config.SEED)

    train_m, val_m, test_m = _mask(train_idx), _mask(val_idx), _mask(test_idx)
    print(f"[split] train: {train_m.sum():,} ({100*y[train_m].float().mean():.2f}% illicit) | "
          f"val: {val_m.sum():,} ({100*y[val_m].float().mean():.2f}% illicit) | "
          f"test: {test_m.sum():,} ({100*y[test_m].float().mean():.2f}% illicit)")
    return train_m, val_m, test_m


def main():
    parser = argparse.ArgumentParser(description="Preprocess IBM AML dataset into PyG graph.")
    parser.add_argument("--split-mode", type=str, default="stratified",
                        choices=["stratified", "temporal"],
                        help="Node split strategy: stratified (default) or temporal (chronological)")
    args = parser.parse_args()

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    df = load_transactions()
    node_map = build_node_index(df)
    x, y = build_node_features(df, node_map)
    edge_index, edge_attr_raw = build_edges(df, node_map)
    train_mask, val_mask, test_mask = make_splits(y, split_mode=args.split_mode, df=df, node_map=node_map)

    # Normalize edge_attr strictly using StandardScaler fit on training edges only (zero leakage)
    train_edge_mask = (train_mask[edge_index[0]] & train_mask[edge_index[1]]).numpy()
    scaler = StandardScaler()
    if train_edge_mask.sum() > 0:
        scaler.fit(edge_attr_raw[train_edge_mask])
    else:
        scaler.fit(edge_attr_raw)
    edge_attr_norm = scaler.transform(edge_attr_raw)
    edge_attr = torch.from_numpy(edge_attr_norm.astype(np.float32))

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.split_mode = args.split_mode

    torch.save(data, config.GRAPH_PT)
    # Persist node_id -> account key mapping for the explainability report.
    (node_map.rename("node_id").rename_axis("account_key").reset_index()
        .to_parquet(config.NODE_MAP_PARQUET, index=False))

    print(f"\n[done] saved graph with edge_attr -> {config.GRAPH_PT}")
    print(data)


if __name__ == "__main__":
    main()
