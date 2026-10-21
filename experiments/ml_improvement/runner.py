"""
Modular Experiment Runner for TRACE ML Improvement Phase.
Enforces zero data leakage:
- Evaluates on held-out test split only.
- Validation split used strictly for model selection and threshold tuning.
- Saves checkpoints strictly to experiments/ml_improvement/checkpoints/.
- Records results in experiments/ml_improvement/results.csv.
"""
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from utils import best_threshold, evaluate, focal_loss_with_logits, format_metrics
from experiments.ml_improvement.tracker import log_experiment, print_summary_table

# --- Architecture Variations --------------------------------------------------

class ResSAGEConvBlock(nn.Module):
    """SAGEConv with Batch/LayerNorm, ReLU, Dropout, and Residual Skip Connection."""
    def __init__(self, in_c, out_c, dropout=0.2, norm_type="batch"):
        super().__init__()
        self.conv = SAGEConv(in_c, out_c)
        if norm_type == "layer":
            self.norm = nn.LayerNorm(out_c)
        else:
            self.norm = nn.BatchNorm1d(out_c)
        self.dropout = dropout
        self.residual = nn.Linear(in_c, out_c) if in_c != out_c else nn.Identity()

    def forward(self, x, edge_index):
        identity = self.residual(x)
        h = self.conv(x, edge_index)
        h = self.norm(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h + identity


class AdvancedGraphSAGE(nn.Module):
    """Configurable GraphSAGE with optional Residuals, LayerNorm, and customizable depth."""
    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 num_layers: int = 3, dropout: float = 0.2,
                 use_residual: bool = True, norm_type: str = "batch"):
        super().__init__()
        self.use_residual = use_residual
        self.blocks = nn.ModuleList()

        dims = [in_channels] + [hidden_channels] * num_layers
        for i in range(num_layers):
            if use_residual:
                self.blocks.append(ResSAGEConvBlock(dims[i], dims[i + 1], dropout=dropout, norm_type=norm_type))
            else:
                conv = SAGEConv(dims[i], dims[i + 1])
                norm = nn.LayerNorm(dims[i + 1]) if norm_type == "layer" else nn.BatchNorm1d(dims[i + 1])
                self.blocks.append(nn.ModuleDict({"conv": conv, "norm": norm}))
        self.dropout = dropout

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x, edge_index):
        for block in self.blocks:
            if self.use_residual:
                x = block(x, edge_index)
            else:
                conv = block["conv"]
                norm = block["norm"]
                x = conv(x, edge_index)
                x = norm(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x).squeeze(-1)


# --- Training & Evaluation Engine ---------------------------------------------

def run_experiment(
    experiment_id: str,
    description: str,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    model_type: str = "sage",
    hidden_channels: int = 128,
    num_layers: int = 3,
    dropout: float = 0.2,
    use_residual: bool = False,
    norm_type: str = "batch",
    loss_type: str = "focal",
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
    lr: float = 5e-3,
    weight_decay: float = 5e-4,
    epochs: int = 60,
    seed: int = 42,
    feature_desc: str = "11 standard node features",
    graph_desc: str = "Directed 1.01M aggregated edges",
    verbose: bool = True
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AdvancedGraphSAGE(
        in_channels=x.size(1),
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        use_residual=use_residual,
        norm_type=norm_type
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_dev = x.to(device)
    edge_index_dev = edge_index.to(device)
    y_dev = y.to(device)
    train_mask_dev = train_mask.to(device)

    best_val_pr = -1.0
    best_state = None
    best_epoch = 0

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_dev, edge_index_dev)

        if loss_type == "focal":
            loss = focal_loss_with_logits(
                logits[train_mask_dev], y_dev[train_mask_dev],
                alpha=focal_alpha, gamma=focal_gamma
            )
        else: # bce with pos_weight
            pos_weight = torch.tensor([80.0], device=device)
            loss = F.binary_cross_entropy_with_logits(
                logits[train_mask_dev], y_dev[train_mask_dev].float(),
                pos_weight=pos_weight
            )

        loss.backward()
        optimizer.step()

        # Evaluate on validation
        model.eval()
        with torch.no_grad():
            full_logits = model(x_dev, edge_index_dev).cpu()
        val_m = evaluate(full_logits, y, val_mask)

        if val_m["pr_auc"] > best_val_pr:
            best_val_pr = val_m["pr_auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = ep

        if verbose and (ep % 10 == 0 or ep == epochs):
            print(f"[{experiment_id} ep {ep:02d}] loss {loss.item():.4f} | val PR-AUC: {val_m['pr_auc']:.4f} (best: {best_val_pr:.4f})")

    training_time = time.time() - t0

    # Final evaluation using validation-selected threshold
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_logits = model(x_dev, edge_index_dev).cpu()

    thr = best_threshold(final_logits, y, val_mask)
    test_m = evaluate(final_logits, y, test_mask, threshold=thr)

    # Status classification
    pr_auc = test_m["pr_auc"]
    if pr_auc >= 0.45:
        status = "Stretch Achieved"
    elif pr_auc >= 0.40:
        status = "Primary Candidate"
    elif pr_auc >= 0.38:
        status = "Improved"
    elif pr_auc >= 0.342:
        status = "Marginal Lift"
    else:
        status = "Below Baseline"

    # Save checkpoint
    ckpt_dir = Path("experiments/ml_improvement/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{experiment_id}.pt"
    torch.save({
        "model_state": best_state,
        "config": {
            "in_channels": x.size(1),
            "hidden_channels": hidden_channels,
            "num_layers": num_layers,
            "dropout": dropout,
            "use_residual": use_residual,
            "norm_type": norm_type,
            "loss_type": loss_type,
            "focal_alpha": focal_alpha,
            "focal_gamma": focal_gamma,
            "lr": lr,
            "weight_decay": weight_decay,
            "seed": seed,
        },
        "threshold": thr,
        "test_metrics": test_m,
        "best_val_pr": best_val_pr,
        "best_epoch": best_epoch,
        "training_time": training_time,
    }, ckpt_path)

    # Log to registry
    model_cfg_str = f"SAGE(L={num_layers}, H={hidden_channels}, drop={dropout}, res={use_residual}, norm={norm_type})"
    loss_cfg_str = f"{loss_type}(a={focal_alpha}, g={focal_gamma})" if loss_type == "focal" else "BCE(pw=80)"

    record = {
        "experiment_id": experiment_id,
        "description": description,
        "features": feature_desc,
        "graph_config": graph_desc,
        "model_config": model_cfg_str,
        "loss_config": loss_cfg_str,
        "pr_auc": f"{pr_auc:.4f}",
        "precision": f"{test_m['precision']:.4f}",
        "recall": f"{test_m['recall']:.4f}",
        "f1": f"{test_m['f1']:.4f}",
        "roc_auc": f"{test_m['roc_auc']:.4f}",
        "threshold": f"{thr:.4f}",
        "training_time": f"{training_time:.1f}s",
        "status": status,
    }
    log_experiment(record)

    print(f"\n[{experiment_id} COMPLETE] PR-AUC={pr_auc:.4f} | F1={test_m['f1']:.4f} | Prec={test_m['precision']:.4f} | Rec={test_m['recall']:.4f} | Status={status}")
    return {
        "record": record,
        "test_metrics": test_m,
        "best_val_pr": best_val_pr,
        "best_epoch": best_epoch,
        "threshold": thr,
        "logits": final_logits,
    }
