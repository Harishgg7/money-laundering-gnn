"""
Baseline comparison: proves the GNN's value for the pitch.

Compares three approaches on the SAME accounts / SAME test split:

  1. Rule-based        — the industry status quo: flag accounts whose total
                         transacted amount crosses a threshold (a stand-in for
                         "flag if > Rs 50,000"-style rules).
  2. MLP (no graph)    — a strong single-account ML model that uses the exact
                         same 11 node features but IGNORES the graph structure.
  3. GraphSAGE (ours)  — reads from checkpoints/best_model.pt (uses the graph).

The gap between (2) and (3) is the money slide: it isolates how much the
*network structure* — not just the features — contributes.

    python baselines.py
"""
from __future__ import annotations

import json

import numpy as np
import torch

import config
from model import build_model
from utils import best_threshold, evaluate, format_metrics


def _metrics_dict(y_true, scores, thr):
    from sklearn.metrics import (average_precision_score, f1_score,
                                 precision_recall_fscore_support, roc_auc_score)
    y_pred = (scores >= thr).astype(int)
    p, r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": p,
        "recall": r,
    }


def _best_thr_np(y_true, scores):
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[int(np.nanargmax(f1[:-1]))]) if len(thr) else 0.5


def run():
    data = torch.load(config.GRAPH_PT, weights_only=False)
    y = data.y.numpy()
    val = data.val_mask.numpy()
    test = data.test_mask.numpy()

    results = {}

    # --- 1. Rule-based -------------------------------------------------------
    # Feature 2 is (standardised, log) total amount sent; a monotone stand-in
    # for a "large-transaction" rule. Threshold tuned on validation for fairness.
    total_sent = data.x[:, 2].numpy()
    thr = _best_thr_np(y[val], total_sent[val])
    results["Rule-based (amount)"] = _metrics_dict(y[test], total_sent[test], thr)

    # --- 2. MLP on node features (no graph) ----------------------------------
    results["MLP (no graph)"] = _train_mlp(data, y, val, test)

    # --- 3. GraphSAGE (ours) -------------------------------------------------
    results["GraphSAGE (ours)"] = _eval_gnn(data)

    # --- Report --------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'Model':<22}{'ROC-AUC':>9}{'PR-AUC':>9}{'F1':>9}"
          f"{'Precision':>11}{'Recall':>9}")
    print("-" * 78)
    for name, m in results.items():
        print(f"{name:<22}{m['roc_auc']:>9.4f}{m['pr_auc']:>9.4f}"
              f"{m['f1']:>9.4f}{m['precision']:>11.4f}{m['recall']:>9.4f}")
    print("=" * 78)

    with open(config.OUTPUT_DIR / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] saved -> {config.OUTPUT_DIR / 'baseline_results.json'}")
    return results


def _train_mlp(data, y, val, test):
    """A 2-layer MLP on the same node features, no message passing."""
    import torch.nn as nn
    from utils import focal_loss_with_logits

    torch.manual_seed(config.SEED)
    x = data.x
    train = data.train_mask
    mlp = nn.Sequential(
        nn.Linear(x.size(1), 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 1),
    )
    opt = torch.optim.Adam(mlp.parameters(), lr=5e-3, weight_decay=5e-4)
    yt = torch.from_numpy(y).float()
    for _ in range(200):
        mlp.train()
        opt.zero_grad()
        logit = mlp(x).squeeze(-1)
        loss = focal_loss_with_logits(logit[train], yt[train],
                                      alpha=config.FOCAL_ALPHA,
                                      gamma=config.FOCAL_GAMMA)
        loss.backward()
        opt.step()
    mlp.eval()
    with torch.no_grad():
        scores = torch.sigmoid(mlp(x).squeeze(-1)).numpy()
    thr = _best_thr_np(y[val], scores[val])
    return _metrics_dict(y[test], scores[test], thr)


def _eval_gnn(data):
    ckpt = torch.load(config.CHECKPOINT_DIR / "best_model.pt", weights_only=False)
    # Rebuild with the architecture the checkpoint was trained with.
    saved = ckpt.get("config", {})
    for k, v in saved.items():
        setattr(config, k, v)
    model = build_model(in_channels=ckpt["in_channels"], cfg=config)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    thr = ckpt.get("threshold") or best_threshold(logits, data.y, data.val_mask)
    m = evaluate(logits, data.y, data.test_mask, threshold=thr)
    return {k: m[k] for k in ["roc_auc", "pr_auc", "f1", "precision", "recall"]}


if __name__ == "__main__":
    run()
