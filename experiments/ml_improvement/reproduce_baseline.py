"""
Reproduce and verify the frozen GraphSAGE baseline.
Trains 3-layer GraphSAGE on aml_graph.pt using exact baseline parameters.
Saves checkpoint strictly to experiments/ml_improvement/checkpoints/.
"""
import time
import numpy as np
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from model import AMLNet
from utils import best_threshold, evaluate, focal_loss_with_logits, format_metrics

def main():
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Load frozen graph
    data = torch.load(config.GRAPH_PT, weights_only=False)
    print(f"[data] nodes={data.num_nodes:,}, edges={data.edge_index.size(1):,}")

    model = AMLNet(
        in_channels=data.num_node_features,
        hidden_channels=config.HIDDEN_CHANNELS,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        model_type="sage"
    ).to(device)

    print(f"[model] GraphSAGE (3-layer) | {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)

    best_val_pr = -1.0
    best_state = None
    best_epoch = 0

    t0 = time.time()
    for epoch in range(1, config.EPOCHS + 1):
        ep_t0 = time.time()
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = focal_loss_with_logits(
            logits[train_mask], y[train_mask],
            alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA
        )
        loss.backward()
        optimizer.step()

        # Evaluate on full graph
        model.eval()
        with torch.no_grad():
            full_logits = model(x, edge_index).cpu()
        val_m = evaluate(full_logits, data.y, data.val_mask)
        ep_time = time.time() - ep_t0

        if epoch % 5 == 0 or epoch == 1 or val_m["pr_auc"] > best_val_pr:
            print(f"[epoch {epoch:02d} - {ep_time:.2f}s] loss {loss.item():.4f} | val: {format_metrics(val_m)}", flush=True)

        if val_m["pr_auc"] > best_val_pr:
            best_val_pr = val_m["pr_auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    total_time = time.time() - t0
    print(f"\n[training complete in {total_time:.1f}s] Best Val PR-AUC = {best_val_pr:.4f} at epoch {best_epoch}")

    # Evaluate best model on test split
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_logits = model(x, edge_index).cpu()

    thr = best_threshold(final_logits, data.y, data.val_mask)
    test_m = evaluate(final_logits, data.y, data.test_mask, threshold=thr)
    print(f"\n[FINAL TEST EVALUATION] (threshold={thr:.4f})")
    print(f"  ROC-AUC:   {test_m['roc_auc']:.4f}")
    print(f"  PR-AUC:    {test_m['pr_auc']:.4f}")
    print(f"  F1:        {test_m['f1']:.4f}")
    print(f"  Precision: {test_m['precision']:.4f}")
    print(f"  Recall:    {test_m['recall']:.4f}")
    if "recall_at_1pct_fpr" in test_m:
        print(f"  R@1%FPR:   {test_m['recall_at_1pct_fpr']:.4f}")

    # Save to experiments area
    ckpt_path = "experiments/ml_improvement/checkpoints/baseline_reproduced.pt"
    torch.save({
        "model_state": best_state,
        "model_type": "sage",
        "in_channels": data.num_node_features,
        "config": {
            "HIDDEN_CHANNELS": config.HIDDEN_CHANNELS,
            "NUM_LAYERS": config.NUM_LAYERS,
            "DROPOUT": config.DROPOUT,
            "MODEL_TYPE": "sage",
        },
        "threshold": thr,
        "test_metrics": test_m,
        "val_pr_auc": best_val_pr,
        "best_epoch": best_epoch,
        "total_time": total_time,
    }, ckpt_path)
    print(f"[saved] -> {ckpt_path}")

if __name__ == "__main__":
    main()
