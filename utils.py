"""Shared utilities: focal loss for extreme imbalance, and evaluation metrics."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_fscore_support, roc_auc_score)


def focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0,
                           reduction="mean"):
    """Binary focal loss (Lin et al., 2017) for a single-logit output.

    Down-weights the vast majority of easy negatives so the ~0.1% laundering
    nodes dominate the gradient. `alpha` balances the positive class.
    """
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t).pow(gamma) * bce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


@torch.no_grad()
def best_threshold(logits, y, mask):
    """Pick the probability threshold that maximises F1 on the given split.

    Under extreme imbalance a fixed 0.5 cutoff is meaningless; the model ranks
    well (high AUC) but its rare-class probabilities sit low, so we choose the
    operating point from validation data instead.
    """
    from sklearn.metrics import precision_recall_curve
    probs = torch.sigmoid(logits[mask]).cpu().numpy()
    y_true = y[mask].cpu().numpy()
    if len(np.unique(y_true)) < 2:
        return 0.5
    prec, rec, thr = precision_recall_curve(y_true, probs)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    # thr has one fewer element than prec/rec.
    best = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[best]) if len(thr) else 0.5


@torch.no_grad()
def evaluate(logits, y, mask, threshold=0.5):
    """Compute the metrics that matter under extreme imbalance."""
    probs = torch.sigmoid(logits[mask]).cpu().numpy()
    y_true = y[mask].cpu().numpy()
    out = {}
    # AUC metrics are undefined if only one class is present in the split.
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, probs)
        out["pr_auc"] = average_precision_score(y_true, probs)
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    y_pred = (probs >= threshold).astype(int)
    out["f1"] = f1_score(y_true, y_pred, zero_division=0)
    p, r, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    out["precision"] = p
    out["recall"] = r
    return out


def format_metrics(m):
    return (f"ROC-AUC {m['roc_auc']:.4f} | PR-AUC {m['pr_auc']:.4f} | "
            f"F1 {m['f1']:.4f} | P {m['precision']:.4f} | R {m['recall']:.4f}")
