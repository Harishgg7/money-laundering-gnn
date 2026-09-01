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

import numpy as np
import pandas as pd
import torch
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
    """Load the raw transaction CSV with sensible dtypes."""
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
    """Aggregate parallel transactions into weighted directed edges."""
    src = node_map.loc[df["src_key"]].to_numpy()
    dst = node_map.loc[df["dst_key"]].to_numpy()
    g = pd.DataFrame({"src": src, "dst": dst,
                      "amt": df[COL_AMT_PAID].to_numpy(dtype=np.float64)})
    agg = g.groupby(["src", "dst"], sort=False).agg(
        count=("amt", "size"), total=("amt", "sum")).reset_index()
    edge_index = torch.tensor(
        np.vstack([agg["src"].to_numpy(), agg["dst"].to_numpy()]),
        dtype=torch.long)
    edge_attr = torch.tensor(
        np.vstack([np.log1p(agg["count"].to_numpy()),
                   np.log1p(agg["total"].to_numpy())]).T,
        dtype=torch.float32)
    print(f"[edges] {edge_index.size(1):,} aggregated edges "
          f"(from {len(df):,} raw transactions)")
    return edge_index, edge_attr


def make_splits(y: torch.Tensor):
    """Stratified train/val/test node masks."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(y.numel())
    y_np = y.numpy()
    train_idx, tmp_idx = train_test_split(
        idx, train_size=config.TRAIN_FRAC, stratify=y_np, random_state=config.SEED)
    rel_val = config.VAL_FRAC / (config.VAL_FRAC + config.TEST_FRAC)
    val_idx, test_idx = train_test_split(
        tmp_idx, train_size=rel_val, stratify=y_np[tmp_idx],
        random_state=config.SEED)

    def _mask(ix):
        m = torch.zeros(y.numel(), dtype=torch.bool)
        m[ix] = True
        return m

    return _mask(train_idx), _mask(val_idx), _mask(test_idx)


def main():
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    df = load_transactions()
    node_map = build_node_index(df)
    x, y = build_node_features(df, node_map)
    edge_index, edge_attr = build_edges(df, node_map)
    train_mask, val_mask, test_mask = make_splits(y)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    torch.save(data, config.GRAPH_PT)
    # Persist node_id -> account key mapping for the explainability report.
    (node_map.rename("node_id").rename_axis("account_key").reset_index()
        .to_parquet(config.NODE_MAP_PARQUET, index=False))

    print(f"\n[done] saved graph -> {config.GRAPH_PT}")
    print(data)


if __name__ == "__main__":
    main()
