"""
Step 4 & 5 of the pipeline: Training loop + Loss & Backpropagation.

Trains AMLNet on the account graph in full-batch mode (the whole 515k-node
graph fits in memory), optimising a focal loss to cope with the ~1000:1 class
imbalance. Loss is computed only on training-mask nodes; the rest of the graph
still participates in message passing. Reports ROC-AUC / PR-AUC / F1 each epoch
and saves the best-by-PR-AUC checkpoint.

    python train.py
"""
from __future__ import annotations

import numpy as np
import torch

import config
from model import build_model
from utils import (best_threshold, evaluate, focal_loss_with_logits,
                   format_metrics)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_graph():
    # PyG Data is a trusted local artifact we created in preprocess.py.
    data = torch.load(config.GRAPH_PT, weights_only=False)
    print(f"[data] {data}")
    return data


@torch.no_grad()
def full_inference(model, data, device):
    """Full-graph forward pass for evaluation (no sampling)."""
    model.eval()
    logits = model(data.x.to(device), data.edge_index.to(device))
    return logits.cpu()


def train():
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = get_device()
    print(f"[device] {device}")

    data = load_graph()
    model = build_model(in_channels=data.num_node_features, cfg=config).to(device)
    print(f"[model] {config.MODEL_TYPE.upper()} | "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)

    # Move the whole graph to the device once (full-batch training).
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)

    best_pr = -1.0
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = focal_loss_with_logits(
            logits[train_mask], y[train_mask],
            alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
        loss.backward()
        optimizer.step()

        # Evaluate on the full graph each epoch.
        logits = full_inference(model, data, device)
        val_m = evaluate(logits, data.y, data.val_mask)
        print(f"[epoch {epoch:02d}] loss {loss.item():.4f} "
              f"| val: {format_metrics(val_m)}", flush=True)

        if val_m["pr_auc"] > best_pr:
            best_pr = val_m["pr_auc"]
            torch.save({"model_state": model.state_dict(),
                        "in_channels": data.num_node_features,
                        "config": {k: getattr(config, k) for k in
                                   ["HIDDEN_CHANNELS", "NUM_LAYERS", "DROPOUT",
                                    "MODEL_TYPE"]}},
                       config.CHECKPOINT_DIR / "best_model.pt")

    # Final test evaluation with the best checkpoint, at a validation-tuned
    # decision threshold (0.5 is meaningless under extreme imbalance).
    ckpt = torch.load(config.CHECKPOINT_DIR / "best_model.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    logits = full_inference(model, data, device)
    thr = best_threshold(logits, data.y, data.val_mask)
    test_m = evaluate(logits, data.y, data.test_mask, threshold=thr)
    print(f"\n[test] (threshold={thr:.4f}) {format_metrics(test_m)}")

    # Persist the operating threshold alongside the checkpoint.
    ckpt["threshold"] = thr
    torch.save(ckpt, config.CHECKPOINT_DIR / "best_model.pt")
    print(f"[done] best model saved -> {config.CHECKPOINT_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    train()
