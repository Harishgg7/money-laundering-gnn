# Explainable GNN for Anti-Money Laundering (AML)

Explainable Graph Neural Networks for detecting money laundering in digital payment networks on the **IBM Transactions for Anti Money Laundering (AML)** dataset (`HI-Small` tier). Accounts are modeled as **nodes** and transactions as **edges**; inductive GNNs (**GraphSAGE** and **EdgeGINE**) learn multi-hop laundering chains (smurfing, structuring, layering), and **GNNExplainer** produces regulator-ready audit trails with quantitative faithfulness scores.

> **SRM Institute of Science and Technology — CSE**  
> Harish GG · Sibins Koshy · Hemakeshwar

---

## Pipeline

| Stage | Script | Description & Outputs |
|---|---|---|
| **1. Preprocess & Graph Creation** | `preprocess.py` | Collapses 5.07M transactions into 1.01M edges; builds 11 node features `x`, 6-dim `edge_attr`, and node splits (`data/processed/aml_graph.pt`). |
| **2. GNN Architectures** | `model.py` | Inductive `GraphSAGE`, `EdgeGraphSAGE`, and `EdgeGINE` with penultimate embedding extractors (`get_embeddings`). |
| **3. Training & Evaluation** | `train.py` | Full-batch training with focal loss (α=0.75, γ=2.0) and validation-tuned threshold; logs ROC-AUC, PR-AUC, F1, and Recall@1%FPR (`checkpoints/best_model.pt`). |
| **4. Baseline Benchmarking** | `baselines.py` | Evaluates Rule-based, MLP, Tabular XGBoost, GNN, and GNN+XGBoost Hybrid (`outputs/baseline_results.json`). |
| **5. Explainability & SAR Memo** | `explain.py` | Extracts decision subgraphs, calculates Fidelity+ and Fidelity- metrics, and exports Markdown SAR memos, PNGs, and interactive HTML visualizers. |
| **6. Showcase Dashboard** | `server.py` / `run_dashboard.py` | Interactive FastAPI dashboard on port 8000 for live graph navigation, threshold sandboxing, and statutory SAR generation. |

---

## Dataset

IBM Transactions for Anti Money Laundering (AML) — `HI-Small` tier:

```bash
python -m kaggle datasets download ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Trans.csv -p data
python -m kaggle datasets download ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_accounts.csv -p data
```

### Graph Topology & Attributes

| Dimension | Specification |
|---|---|
| **Accounts (Nodes)** | 515,088 unique accounts across institutional banks |
| **Raw Transactions** | 5,078,345 transactions |
| **Aggregated Edges** | 1,015,736 directed money-flow connections |
| **Illicit Prevalence** | 6,357 accounts (1.23% positive class) |
| **Node Features (`data.x`)** | 11 dimensions: in/out degrees, total & mean sent/received, unique counterparties, currency diversity, format diversity, self-loop flag |
| **Edge Attributes (`data.edge_attr`)** | 6 dimensions: `tx_count`, `total_amount`, `mean_amount`, `std_amount`, `min_time_delta`, `unique_formats_ratio` (StandardScaler fit on training edges only) |

---

## Installation & Setup

```bash
pip install -r requirements.txt
```

> **Environment Note:** PyTorch and PyTorch Geometric run in full-batch mode on both GPU (CUDA) and CPU. No compiled C++ extensions (`pyg-lib` or `torch-sparse`) are required. Recommended Python environment: Python 3.11 or 3.12.

---

## CLI Usage & Workflow

### 1. Preprocess & Split Strategy
```bash
# Stratified node split (default)
python preprocess.py --split-mode stratified

# Temporal split (chronological order by account last-active timestamp)
python preprocess.py --split-mode temporal
```

### 2. Train GNN Models
```bash
# Train EdgeGINE (uses data.edge_attr)
python train.py --model egine --epochs 60 --lr 0.005

# Train classic GraphSAGE (topology only)
python train.py --model graphsage --epochs 60
```

### 3. Run Benchmark Comparisons
```bash
python baselines.py
```

### 4. Explain Flagged Accounts & Generate SAR Audits
```bash
# Explain highest-risk account
python explain.py --top 1

# Explain specific node
python explain.py --node 496618
```
Outputs generated in `outputs/`:
- `audit_node_<id>.md`: Human-readable FinCEN Form 111 Suspicious Activity Report (SAR) memo
- `audit_node_<id>.png`: Static publication figure
- `audit_node_<id>.html`: Standalone interactive Vis.js drag-and-zoom network visualizer

### 5. Launch Live Regulatory Showcase Dashboard
```bash
python run_dashboard.py
# Opens interactive dashboard at http://127.0.0.1:8000
```

---

## Benchmark Results (Test Split)

Evaluated on the exact same 15% held-out test split (77,264 accounts) with operating thresholds tuned on validation:

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall | Architecture Notes |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Rule-based (amount)** | 0.6510 | 0.0210 | 0.0440 | 0.0240 | 0.2200 | Industry status quo ("flag if > $X"); 97.6% false positives |
| **MLP (features only)** | 0.8520 | 0.1840 | 0.2610 | 0.4870 | 0.1780 | 2-layer MLP on 11 raw node features without graph connectivity |
| **XGBoost (Tabular)** | 0.8654 | 0.2180 | 0.2980 | 0.5120 | 0.2110 | Tuned gradient boosting on tabular node features |
| **GraphSAGE (ours)** | 0.9060 | 0.3420 | 0.3680 | 0.5700 | 0.2710 | 3-hop topological inductive message passing |
| **EdgeGINE (ours)** | **0.9185** | **0.3790** | **0.4020** | **0.5980** | **0.3040** | Multi-hop message passing with 6-dim aggregated edge features |
| **GNN + XGBoost (Hybrid)**| **0.9240** | **0.3950** | **0.4180** | **0.6150** | **0.3180** | Combines raw tabular features with penultimate GNN embeddings |

---

## Explainability & Regulatory Compliance

To satisfy financial regulations (BSA Section 314(a), FinCEN Form 111, EU 6AMLD), every flagged prediction is verified quantitatively and qualitatively:

1. **Fidelity+ (Necessity):** Quantifies the probability drop when explanatory money-flow conduits are severed ($P_{orig} - P_{drop}$). High positive scores confirm that isolated transactions are necessary to trigger the alert.
2. **Fidelity- (Sufficiency):** Measures the prediction when preserving *only* the explanatory subgraph ($P_{orig} - P_{keep}$).
3. **Statutory Audit Trail:** Outputs comprehensive Markdown memos detailing depositing smurfing rings, layering hubs, and recommended subpoenas.
