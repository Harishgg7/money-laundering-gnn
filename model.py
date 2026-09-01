"""
Step 3 of the pipeline: GNN Model Definition.

A modular message-passing encoder (GraphSAGE by default, with GCN/GAT
selectable) followed by an MLP classification head that outputs a single
fraud logit per account node.

GraphSAGE is the default because it is *inductive* (learns aggregation
functions, so it can score accounts unseen during training), scales to large
transaction graphs via neighbour sampling, and works cleanly with
GNNExplainer for the regulator-ready audit trail.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


def _make_conv(model_type: str, in_c: int, out_c: int):
    model_type = model_type.lower()
    if model_type == "sage":
        return SAGEConv(in_c, out_c)
    if model_type == "gcn":
        return GCNConv(in_c, out_c)
    if model_type == "gat":
        # 4 attention heads, concatenated back to out_c.
        assert out_c % 4 == 0, "GAT hidden channels must be divisible by heads"
        return GATConv(in_c, out_c // 4, heads=4)
    raise ValueError(f"unknown model_type: {model_type!r}")


class AMLNet(nn.Module):
    """GNN encoder + MLP classifier for account-level fraud detection."""

    def __init__(self, in_channels: int, hidden_channels: int = 64,
                 num_layers: int = 2, dropout: float = 0.3,
                 model_type: str = "sage"):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = [in_channels] + [hidden_channels] * num_layers
        for i in range(num_layers):
            self.convs.append(_make_conv(model_type, dims[i], dims[i + 1]))
            self.norms.append(nn.BatchNorm1d(dims[i + 1]))

        # MLP classification head -> single logit.
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x, edge_index, edge_attr=None):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x).squeeze(-1)  # [num_nodes] logits


def build_model(in_channels: int, cfg) -> AMLNet:
    return AMLNet(
        in_channels=in_channels,
        hidden_channels=cfg.HIDDEN_CHANNELS,
        num_layers=cfg.NUM_LAYERS,
        dropout=cfg.DROPOUT,
        model_type=cfg.MODEL_TYPE,
    )
