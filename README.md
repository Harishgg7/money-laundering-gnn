# Explainable GNN for Anti-Money Laundering (AML)

Explainable Graph Neural Networks for detecting money laundering in digital
payment networks. Accounts are modelled as **nodes** and transactions as
**edges**; a GraphSAGE GNN learns laundering patterns from multi-hop money
flows, and **GNNExplainer** produces a regulator-ready audit trail for each
flagged account.

> SRM Institute of Science and Technology — CSE
> Harish GG · Sibins Koshy · Hemakeshwar

## Pipeline

| Stage | Script | Output |
|-------|--------|--------|
| 1. Preprocess transaction logs | `preprocess.py` | node IDs, `edge_index`, features `x` |
| 2. Build PyG dataset | `dataset.py` | `Data` object (`data/processed/aml_graph.pt`) |
| 3. GNN model | `model.py` | GraphSAGE + MLP classifier |
| 4. Train / evaluate | `train.py` | trained weights, ROC-AUC / PR / F1 |
| 5. Explain a flag | `explain.py` | important sub-graph for a flagged node |

## Dataset

IBM Transactions for Anti Money Laundering (AML) — the `HI-Small` tier.

```bash
python -m kaggle datasets download ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Trans.csv -p data
python -m kaggle datasets download ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_accounts.csv -p data
```

Files land in `data/`.

## Setup

```bash
pip install -r requirements.txt
```

> **Note (Python 3.13):** PyTorch Geometric wheels may lag the newest Python.
> If install fails, use a Python 3.11/3.12 virtual environment.

## Run

```bash
python preprocess.py                 # CSV -> PyG graph (data/processed/aml_graph.pt)
python train.py                      # train GraphSAGE + evaluate
python explain.py --top 1            # explain the highest-scoring account
python explain.py --node <node_id>   # explain a specific account
```

`preprocess.py` builds the PyG `Data` object directly, so there is no separate
`dataset.py` step to run.

## Graph (HI-Small tier)

| | |
|---|---|
| Accounts (nodes) | 515,088 |
| Transactions (raw) | 5,078,345 |
| Edges (aggregated) | 1,015,736 |
| Illicit accounts | 6,357 (1.23%) |
| Node features | 11 (degrees, amounts, counterparties, currency/format diversity, self-loop) |

## Results (test set)

Same accounts, same held-out 15% test split, threshold tuned on validation:

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|-------|:-------:|:------:|:---------:|:------:|:--:|
| Rule-based (amount) | 0.651 | 0.021 | 0.024 | 0.220 | 0.044 |
| MLP (features, no graph) | 0.852 | 0.184 | 0.487 | 0.178 | 0.261 |
| **GraphSAGE (ours, 3-hop)** | **0.906** | **0.342** | **0.570** | **0.271** | **0.368** |

The jump from the MLP to GraphSAGE isolates the value of the **graph structure**
alone (PR-AUC 0.184 → 0.342). Rule-based flagging drowns in false positives
(2.4% precision). Reproduce the table with `python baselines.py`.

Trained with **focal loss** (α=0.75, γ=2) for the ~80:1 class imbalance and a
**validation-tuned decision threshold** (0.5 is meaningless here). `GNNExplainer`
then produces a per-account audit trail (`outputs/audit_node_<id>.png`).

> **Environment note:** on Python 3.13, PyG's `NeighborLoader` needs compiled
> `pyg-lib`/`torch-sparse` wheels that don't yet exist for 3.13, so training runs
> **full-batch** (the whole graph fits in memory). Swap to `NeighborLoader` on a
> 3.11/3.12 env if you scale to the Medium/Large tiers.
