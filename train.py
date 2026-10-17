"""
Step 4 & 5 of the pipeline: Training loop + Loss & Backpropagation.

Trains EdgeGINE or GraphSAGE on the account graph in full-batch mode (the whole 515k-node
graph fits in memory), optimising a focal loss to cope with the class imbalance.
Loss is computed only on training-mask nodes; the rest of the graph
still participates in message passing. Reports ROC-AUC / PR-AUC / F1 / Recall@1%FPR each epoch
and saves the best-by-PR-AUC checkpoint.

    python train.py                      # defaults to egine if data.edge_attr present
    python train.py --model graphsage    # classic topology-only GraphSAGE
    python train.py --model egine        # edge-aware GINE
"""
from __future__ import annotations

import argparse
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
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device) if hasattr(data, "edge_attr") and data.edge_attr is not None else None
    logits = model(x, edge_index, edge_attr)
    return logits.cpu()


def train():
    parser = argparse.ArgumentParser(description="Train GNN for Anti-Money Laundering.")
    parser.add_argument("--model", type=str, default=None,
                        choices=["graphsage", "egine", "edge_sage", "sage"],
                        help="GNN model architecture (default: egine if edge_attr present, else graphsage)")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = get_device()
    print(f"[device] {device}")

    data = load_graph()
    has_edge_attr = hasattr(data, "edge_attr") and data.edge_attr is not None

    if args.model is not None:
        model_type = args.model
    else:
        model_type = "egine" if has_edge_attr else "graphsage"

    config.MODEL_TYPE = model_type
    edge_channels = data.edge_attr.size(1) if has_edge_attr else 6

    model = build_model(
        in_channels=data.num_node_features,
        cfg=config,
        edge_channels=edge_channels
    ).to(device)

    print(f"[model] {model_type.upper()} | "
          f"{sum(p.numel() for p in model.parameters()):,} params | "
          f"edge_channels={edge_channels if model_type != 'graphsage' else 0}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=config.WEIGHT_DECAY)

    # Move the whole graph to the device once (full-batch training).
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device) if has_edge_attr else None
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)

    best_pr = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index, edge_attr)
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
                        "model_type": model_type,
                        "in_channels": data.num_node_features,
                        "edge_channels": edge_channels,
                        "config": {
                            "HIDDEN_CHANNELS": config.HIDDEN_CHANNELS,
                            "NUM_LAYERS": config.NUM_LAYERS,
                            "DROPOUT": config.DROPOUT,
                            "MODEL_TYPE": model_type,
                        }},
                       config.CHECKPOINT_DIR / "best_model.pt")

    # Final test evaluation with the best checkpoint, at a validation-tuned
    # decision threshold (0.5 is meaningless under extreme imbalance).
    ckpt = torch.load(config.CHECKPOINT_DIR / "best_model.pt", weights_only=False)
    for k, v in ckpt.get("config", {}).items():
        setattr(config, k, v)
    model = build_model(in_channels=ckpt["in_channels"], cfg=config, edge_channels=ckpt.get("edge_channels", 6)).to(device)
    model.load_state_dict(ckpt["model_state"])
    logits = full_inference(model, data, device)
    thr = best_threshold(logits, data.y, data.val_mask)
    test_m = evaluate(logits, data.y, data.test_mask, threshold=thr)
    print(f"\n[test] (threshold={thr:.4f}) {format_metrics(test_m)}")

    # Persist the operating threshold and test metrics alongside the checkpoint.
    ckpt["threshold"] = thr
    ckpt["test_metrics"] = test_m
    torch.save(ckpt, config.CHECKPOINT_DIR / "best_model.pt")
    print(f"[done] best model saved -> {config.CHECKPOINT_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    train()
