# TRACE — Demo Runbook & 4-Minute Presentation Guide
### Graph Intelligence for Financial Crime Detection

A presenter's guide for the live demonstration.
The platform runs as a single, fully integrated web application on `http://127.0.0.1:8000` via `python run_dashboard.py` (or `npm run dev`).

---

## 0. Quick Pre-Demo Checklist (30 seconds)

1. Run `python run_dashboard.py` or visit `http://127.0.0.1:8000`.
2. Press **`P`** to toggle **PRESENTATION MODE** (enlarges text and graph viewport for projection).
3. Verify keyboard shortcuts:
   - **`1`** &rarr; Switch to **CASE 01** (Smurfing / Fan-In Hub)
   - **`2`** &rarr; Switch to **CASE 02** (Inflow Aggregation Ring)
   - **`3`** &rarr; Switch to **CASE 03** (Legitimate Retail Account)
   - **`E`** &rarr; Trigger **[ EXPLAIN PREDICTION ]**
   - **`R`** &rarr; Reset graph zoom and pan

---

## 1. The 4-Minute Live Demo Flow

### 0:00 – 0:30 &bull; Introduce the Problem
> *"In modern financial crime, individual transactions almost never look suspicious in isolation — launderers intentionally keep amounts below reporting limits. Suspicious behavior only emerges from the structural relationships between accounts."*

### 0:30 – 1:00 &bull; Show the Network Graph
> *"TRACE models accounts as nodes and money flows as directed edges across 515,000 accounts and 1,000,000+ transaction edges from the IBM AML dataset. 3-hop GraphSAGE aggregates relational context from surrounding accounts."*

### 1:00 – 1:30 &bull; Model Comparison (MLP vs. GraphSAGE)
> *"Both our MLP baseline and GraphSAGE share identical account-level features. The jump from 48.7% to 57.0% precision — and nearly doubling PR-AUC from 0.184 to 0.342 — isolates the exact value provided by the graph structure."*

### 1:30 – 3:00 &bull; Click [ INVESTIGATE ACCOUNT ] (Case 01)
> *"Let's investigate Account `0214-80C1C8AC0`. At the validation-tuned threshold of 0.378, TRACE flags this account with a 94.2% GNN model risk score. Notice the money flow animation: eight distinct feeder accounts funnel small transfers into an intermediary hub, which executes one decisive transfer into the target."*

### 3:00 – 3:40 &bull; Click [ EXPLAIN PREDICTION ]
> *"Regulators do not accept black-box predictions. GNNExplainer isolates the decisive transaction pathways with quantitative faithfulness (Fidelity+ drop of +0.1141). Compliance officers receive a court-ready audit trail naming the specific counterparties and amounts."*

### 3:40 – 4:00 &bull; Press '3' to Show Legitimate Account (Case 03) & Conclude
> *"When we inspect legitimate Account `001-800056370`, notice the stark contrast: a simple ordinary retail counterparty with low risk (12.4%) and zero suspicious patterns. The value of TRACE is not just predicting risk, but giving investigators an auditable starting point."*

---

## 2. Key Metrics to Have Memorized

| Model | PR-AUC | Precision | Recall | F1 Score |
|---|:---:|:---:|:---:|:---:|
| **Rule-Based (Amount)** | 0.021 | 2.4% | 22.0% | 0.044 |
| **MLP (No Graph)** | 0.184 | 48.7% | 17.8% | 0.261 |
| **GraphSAGE (TRACE)** | **0.342** | **57.0%** | **27.1%** | **0.368** |


---

## 3. Run it live — ~1 min (the command itself takes seconds)

```bash
python demo.py --node 496618
```

Narrate as the four steps print:
1. **Loads** the trained model + the 515k-account graph.
2. **Scores every account** in ~2 seconds (~387 flagged).
3. **Picks** the flagged account.
4. **GNNExplainer** finds the exact transactions behind the flag, working on
   the account's 3-hop neighbourhood (~2s). The whole demo finishes in seconds.

When the **AUDIT TRAIL** table appears, say (top row is the most important):
> "The model didn't just say 'risky' — the top transaction is a hub account
> forwarding money straight to the flagged account. And below it you can see the
> money being gathered a layer earlier: several accounts feed an intermediary,
> which feeds the hub. That's a classic layering / smurfing chain — and this
> table is exactly what a compliance officer would hand to a regulator."

> `--node <id>` explains any account; omit `--node` to auto-pick a structural
> case (scans the top-25 riskiest for the one with the richest fan-in).

---

## 4. Close (20 sec)

> "So we've turned a black box into an audit trail: accurate detection *and* a
> human-readable reason for every flag — which is exactly what makes it
> deployable in regulated finance. It's on GitHub, fully reproducible."

---

## 5. Anticipated questions (rehearse these)

**Q: Recall is only 27% — you miss most launderers?**
> By design we tuned for precision. In AML the real failure mode is false
> positives — investigators can't chase thousands of bad alerts. We flag a
> smaller, high-quality set (57% correct vs. 2% for rules). Recall is our stated
> next step, via class-balanced sampling and deeper hops.

**Q: Why GraphSAGE and not GCN or GAT?**
> GraphSAGE is *inductive* — it learns aggregation functions, so it can score
> new accounts it never saw in training. Payment networks add accounts daily, so
> that matters. Our model file makes swapping to GAT/GCN a one-line change.

**Q: Is this real data?**
> IBM's synthetic AML dataset — realistic, labelled, and standard for
> benchmarking because real banking data can't be shared. 5 million
> transactions, 515k accounts.

**Q: How does the explanation actually work?**
> GNNExplainer learns a mask over edges — it finds the smallest set of
> transactions that, if kept, still produce the same fraud score. Those are the
> transactions "responsible" for the flag.

**Q: How do you handle 0.1% laundering (imbalance)?**
> Focal loss down-weights the easy majority of legitimate accounts so the rare
> laundering cases dominate learning, plus we tune the decision threshold on
> validation instead of using a naive 0.5.

**Q: Could a launderer evade it?**
> Yes — adversaries adapt, which is why recent research (and our future work)
> looks at continual learning. But structural detection is much harder to evade
> than a fixed amount threshold, because you'd have to disguise the whole flow.

---

## 6. If something breaks (fallback)

- **`demo.py` errors or hangs** → don't debug live. Switch to the dashboard
  (`static/index.html`) and the pre-generated `outputs\audit_node_496618.png`, and talk
  through those instead. The result is identical to what the live run produces.
- **No internet / dashboard won't load** → `static/index.html` is a local file; open it
  directly. Fonts fall back gracefully offline.
