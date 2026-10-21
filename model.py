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
from torch_geometric.nn import GATConv, GCNConv, GINEConv, MessagePassing, SAGEConv


def _make_conv(model_type: str, in_c: int, out_c: int):
    model_type = model_type.lower()
    if model_type in ["sage", "graphsage"]:
        return SAGEConv(in_c, out_c)
    if model_type == "gcn":
        return GCNConv(in_c, out_c)
    if model_type == "gat":
        assert out_c % 4 == 0, "GAT hidden channels must be divisible by heads"
        return GATConv(in_c, out_c // 4, heads=4)
    raise ValueError(f"unknown model_type: {model_type!r}")


class AMLNet(nn.Module):
    """GNN encoder + MLP classifier for account-level fraud detection (GraphSAGE/GCN/GAT)."""

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 num_layers: int = 3, dropout: float = 0.2,
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

    def get_embeddings(self, x, edge_index, edge_attr=None):
        """Extract penultimate hidden layer representations."""
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # [num_nodes, hidden_channels]

    def forward(self, x, edge_index, edge_attr=None):
        h = self.get_embeddings(x, edge_index, edge_attr)
        return self.head(h).squeeze(-1)  # [num_nodes] logits


# Alias for backward compatibility
GraphSAGE = AMLNet


class EdgeSAGEConv(MessagePassing):
    """SAGE convolution with linear edge feature projection and mean aggregation."""

    def __init__(self, in_channels: int, out_channels: int, edge_channels: int = 6):
        super().__init__(aggr="mean")
        self.edge_proj = nn.Linear(edge_channels, in_channels)
        self.lin_l = nn.Linear(in_channels, out_channels)
        self.lin_r = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        edge_proj = self.edge_proj(edge_attr) if edge_attr is not None else None
        out = self.propagate(edge_index, x=x, edge_proj=edge_proj)
        return self.lin_l(x) + self.lin_r(out)

    def message(self, x_j, edge_proj):
        return F.relu(x_j + edge_proj) if edge_proj is not None else x_j


class EdgeGraphSAGE(nn.Module):
    """Edge-Aware GraphSAGE integrating 6-dim edge attributes into message passing."""

    def __init__(self, in_channels: int, edge_channels: int = 6,
                 hidden_channels: int = 128, num_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = [in_channels] + [hidden_channels] * num_layers
        for i in range(num_layers):
            self.convs.append(EdgeSAGEConv(dims[i], dims[i + 1], edge_channels=edge_channels))
            self.norms.append(nn.BatchNorm1d(dims[i + 1]))

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def get_embeddings(self, x, edge_index, edge_attr=None):
        """Extract penultimate hidden layer representations."""
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index, edge_attr=None):
        h = self.get_embeddings(x, edge_index, edge_attr)
        return self.head(h).squeeze(-1)


class EdgeGINE(nn.Module):
    """Edge-Aware Graph Isomorphism Network (GINE) using GINEConv.

    Uses an edge-projection MLP matching hidden_channels to integrate
    edge_attr into expressive multi-hop relational message passing.
    """

    def __init__(self, in_channels: int, edge_channels: int = 6,
                 hidden_channels: int = 128, num_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.node_proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINEConv(mlp, edge_dim=edge_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def get_embeddings(self, x, edge_index, edge_attr=None):
        """Extract penultimate hidden layer representations."""
        x = self.node_proj(x)
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), 6), device=x.device, dtype=x.dtype)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index, edge_attr=None):
        h = self.get_embeddings(x, edge_index, edge_attr)
        return self.head(h).squeeze(-1)


def build_model(in_channels: int, cfg, edge_channels: int = 6) -> nn.Module:
    m_type = getattr(cfg, "MODEL_TYPE", "sage").lower()
    hidden_channels = getattr(cfg, "HIDDEN_CHANNELS", 128)
    num_layers = getattr(cfg, "NUM_LAYERS", 3)
    dropout = getattr(cfg, "DROPOUT", 0.2)

    if m_type in ["egine", "gine"]:
        return EdgeGINE(
            in_channels=in_channels,
            edge_channels=edge_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
    if m_type in ["edge_sage", "edgesage", "edgegraphsage"]:
        return EdgeGraphSAGE(
            in_channels=in_channels,
            edge_channels=edge_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
        )
    if m_type in ["advanced_sage", "advanced_graphsage", "hybrid_encoder"]:
        return AdvancedGraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            use_residual=getattr(cfg, "USE_RESIDUAL", False),
            norm_type=getattr(cfg, "NORM_TYPE", "batch"),
        )
    return AMLNet(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        model_type=m_type,
    )


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

    def forward(self, x, edge_index, edge_attr=None):
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

    def get_embeddings(self, x, edge_index, edge_attr=None):
        for block in self.blocks:
            if self.use_residual:
                x = block(x, edge_index)
            else:
                conv = block["conv"]
                norm = block["norm"]
                x = conv(x, edge_index)
                x = norm(x)
                x = F.relu(x)
        return x


