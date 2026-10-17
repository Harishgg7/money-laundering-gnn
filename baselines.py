"""
Baseline comparison: proves the GNN's value for the pitch.

Compares six approaches on the SAME accounts / SAME test split:

  1. Rule-based             — the industry status quo: flag accounts whose total
                              transacted amount crosses a threshold (a stand-in for
                              "flag if > Rs 50,000"-style rules).
  2. MLP (no graph)         — a strong single-account ML model that uses the exact
                              same 11 node features but IGNORES the graph structure.
  3. XGBoost (Tabular)      — Gradient Boosted Decision Trees on raw node features.
  4. GNN (ours)             — Trained GraphSAGE / EdgeGINE from best_model.pt.
  5. GNN + XGBoost (Hybrid) — Combines raw node features with penultimate GNN embeddings.

The gaps isolate how much the network structure and edge intelligence contribute.

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
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(p),
        "recall": float(r),
        "threshold": float(thr),
    }


def _best_thr_np(y_true, scores):
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[int(np.nanargmax(f1[:-1]))]) if len(thr) else 0.5


def run():
    data = torch.load(config.GRAPH_PT, weights_only=False)
    y = data.y.numpy()
    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()

    results = {}

    # --- 1. Rule-based -------------------------------------------------------
    print("[1/5] Evaluating Rule-based baseline (amount sent) ...")
    total_sent = data.x[:, 2].numpy()
    thr = _best_thr_np(y[val_mask], total_sent[val_mask])
    results["Rule-based (amount)"] = _metrics_dict(y[test_mask], total_sent[test_mask], thr)

    # --- 2. MLP on node features (no graph) ----------------------------------
    print("[2/5] Training & evaluating MLP baseline (features only) ...")
    results["MLP (no graph)"] = _train_mlp(data, y, val_mask, test_mask)

    # --- 3. XGBoost on node features (Tabular Only) --------------------------
    print("[3/5] Training & evaluating XGBoost baseline (tabular only) ...")
    results["XGBoost (Tabular)"] = _train_xgboost_tabular(data, y, train_mask, val_mask, test_mask)

    # --- 4. GNN (ours) -------------------------------------------------------
    print("[4/5] Evaluating trained GNN (best_model.pt) ...")
    gnn_name, gnn_res = _eval_gnn(data)
    results[gnn_name] = gnn_res

    # --- 5. GNN + XGBoost Hybrid ---------------------------------------------
    print("[5/5] Training & evaluating GNN + XGBoost Hybrid ...")
    results["GNN + XGBoost (Hybrid)"] = _train_hybrid(data, y, train_mask, val_mask, test_mask)

    # --- Clean Markdown Table Report -----------------------------------------
    print("\n" + "=" * 80)
    print("### Model Benchmark Comparison (IBM AML HI-Small Test Split)")
    print("=" * 80)
    print(f"| {'Model':<26} | {'ROC-AUC':>7} | {'PR-AUC':>7} | {'F1':>6} | {'Precision':>9} | {'Recall':>7} |")
    print(f"|:{'-'*26}-|-{'-'*7}:|-{'-'*7}:|-{'-'*6}:|-{'-'*9}:|-{'-'*7}:|")
    for name, m in results.items():
        print(f"| {name:<26} | {m['roc_auc']:>7.4f} | {m['pr_auc']:>7.4f} | "
              f"{m['f1']:>6.4f} | {m['precision']:>9.4f} | {m['recall']:>7.4f} |")
    print("=" * 80 + "\n")

    with open(config.OUTPUT_DIR / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] saved results -> {config.OUTPUT_DIR / 'baseline_results.json'}")
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


def _train_xgboost_tabular(data, y, train_mask, val_mask, test_mask):
    """XGBoost trained strictly on the 11 engineered tabular node features."""
    from xgboost import XGBClassifier

    X = data.x.numpy()
    y_train = y[train_mask]
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw = float(n_neg / max(1, n_pos))

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=config.SEED,
        n_jobs=-1,
    )
    clf.fit(X[train_mask], y_train)
    val_probs = clf.predict_proba(X[val_mask])[:, 1]
    thr = _best_thr_np(y[val_mask], val_probs)
    test_probs = clf.predict_proba(X[test_mask])[:, 1]
    return _metrics_dict(y[test_mask], test_probs, thr)


def _eval_gnn(data):
    """Evaluate trained GNN from best_model.pt."""
    ckpt = torch.load(config.CHECKPOINT_DIR / "best_model.pt", weights_only=False)
    saved = ckpt.get("config", {})
    for k, v in saved.items():
        setattr(config, k, v)
    edge_channels = ckpt.get("edge_channels", 6)
    model = build_model(in_channels=ckpt["in_channels"], cfg=config, edge_channels=edge_channels)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device) if hasattr(data, "edge_attr") and data.edge_attr is not None else None

    with torch.no_grad():
        logits = model(x, edge_index, edge_attr).cpu()
    thr = ckpt.get("threshold") or best_threshold(logits, data.y, data.val_mask)
    m = evaluate(logits, data.y, data.test_mask, threshold=thr)
    model_name = "EdgeGINE (ours)" if getattr(config, "MODEL_TYPE", "").lower() in ["egine", "gine"] else "GraphSAGE (ours)"
    return model_name, {k: m[k] for k in ["roc_auc", "pr_auc", "f1", "precision", "recall"]}


def _train_hybrid(data, y, train_mask, val_mask, test_mask):
    """GNN + XGBoost Hybrid: Concatenates tabular node features with GNN embeddings."""
    from xgboost import XGBClassifier

    ckpt = torch.load(config.CHECKPOINT_DIR / "best_model.pt", weights_only=False)
    saved = ckpt.get("config", {})
    for k, v in saved.items():
        setattr(config, k, v)
    edge_channels = ckpt.get("edge_channels", 6)
    model = build_model(in_channels=ckpt["in_channels"], cfg=config, edge_channels=edge_channels)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device) if hasattr(data, "edge_attr") and data.edge_attr is not None else None

    with torch.no_grad():
        h = model.get_embeddings(x, edge_index, edge_attr).cpu().numpy()

    X_comb = np.hstack([data.x.numpy(), h])
    y_train = y[train_mask]
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    spw = float(n_neg / max(1, n_pos))

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=config.SEED,
        n_jobs=-1,
    )
    clf.fit(X_comb[train_mask], y_train)
    val_probs = clf.predict_proba(X_comb[val_mask])[:, 1]
    thr = _best_thr_np(y[val_mask], val_probs)
    test_probs = clf.predict_proba(X_comb[test_mask])[:, 1]
    return _metrics_dict(y[test_mask], test_probs, thr)


if __name__ == "__main__":
    run()
