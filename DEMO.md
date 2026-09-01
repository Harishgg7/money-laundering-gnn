# Demo Runbook — Explainable GNN for Anti-Money Laundering

A presenter's guide for the ~7-minute teacher demo. Everything below runs on
the project machine where the dataset, trained model, and graph already exist.

---

## 0. Before you walk in (2-minute checklist)

```bash
cd "F:\porject ml"
python demo.py --node 496618
```

- [ ] The command above completes and prints an **AUDIT TRAIL** table.
- [ ] `pitch.html` opens in a browser (double-click it) — this is your visual backup.
- [ ] `outputs\model_comparison.png` opens (your results chart).
- [ ] Terminal font size bumped up so the class can read it.

If `demo.py` says artifacts are missing, run `python preprocess.py` then
`python train.py` once (only needed on a fresh machine).

---

## 1. The 60-second opening (what to say first)

> "Money laundering doesn't look suspicious one transaction at a time — it hides
> as a *shape* across many accounts: money split up and funnelled through mule
> accounts. Banks today use fixed rules like 'flag anything over ₹50,000'. Those
> drown investigators in false alarms, and the accurate AI models that exist are
> black boxes a regulator won't accept.
>
> We built a Graph Neural Network that treats **accounts as nodes and
> transactions as edges**, learns the *shape* of laundering across multiple hops,
> and — crucially — **explains every flag** as a named trail of accounts. So it's
> both accurate *and* auditable."

---

## 2. Show the problem & results (dashboard) — 2 min

Open **`pitch.html`** (local file, works offline) OR the published dashboard.
Walk through, top to bottom:

1. **The compliance gap** — rules vs. our approach.
2. **The results chart.** Say the one killer line:
   > "The old rule-based approach flags laundering with **2.4% precision** — 97
   > out of 100 alerts are false. Our GNN reaches **57%**. And moving from a
   > plain neural net to the *graph* nearly **doubles** the score — proof it's
   > the network structure doing the work, not just the features."
3. Point at the **fan-in diagram** — "this is a real flagged account."

**Numbers to have memorised:**

| | Rule-based | MLP (no graph) | **GNN (ours)** |
|---|---|---|---|
| Precision | 0.024 | 0.487 | **0.570** |
| PR-AUC | 0.021 | 0.184 | **0.342** |
| ROC-AUC | 0.651 | 0.852 | **0.906** |

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
  (`pitch.html`) and the pre-generated `outputs\audit_node_496618.png`, and talk
  through those instead. The result is identical to what the live run produces.
- **No internet / dashboard won't load** → `pitch.html` is a local file; open it
  directly. Fonts fall back gracefully offline.
