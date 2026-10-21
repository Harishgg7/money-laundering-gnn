"""
Experiment Group 3: Graph Topology & Bidirectional Message Passing.
Tests whether bidirectional fund-flow propagation allows GraphSAGE to capture
both upstream smurfing funnels and downstream layering distribution hubs.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch_geometric.utils import to_undirected
import config
from experiments.ml_improvement.runner import run_experiment
from experiments.ml_improvement.tracker import print_summary_table

def main():
    print("=== EXPERIMENT GROUP 3: GRAPH TOPOLOGY & MESSAGE PASSING ===")
    data = torch.load(config.GRAPH_PT, weights_only=False)

    # 1. Bidirectional / Undirected Message Passing
    print("\n[Running EXP_001_UNDIRECTED]...")
    undir_edge_index = to_undirected(data.edge_index)

    res = run_experiment(
        experiment_id="EXP_001_UNDIRECTED",
        description="GraphSAGE with bidirectional message passing (to_undirected)",
        x=data.x,
        edge_index=undir_edge_index,
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        model_type="sage",
        hidden_channels=128,
        num_layers=3,
        dropout=0.2,
        use_residual=False,
        norm_type="batch",
        loss_type="focal",
        focal_alpha=0.75,
        focal_gamma=2.0,
        lr=5e-3,
        weight_decay=5e-4,
        epochs=60,
        seed=42,
        feature_desc="11 standard node features",
        graph_desc="Bidirectional 2.03M edges (undirected)",
    )

    print_summary_table()

if __name__ == "__main__":
    main()
