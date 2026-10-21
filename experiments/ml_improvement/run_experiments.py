"""
Unified Experiment Suite for TRACE ML Improvement Phase.
Executes systematic controlled experiments across:
- Group 1: Feature Engineering (29 AML behavioral features)
- Group 2: Class Imbalance Tuning
- Group 3: Graph Topology (Bidirectional message passing)
- Group 4: GraphSAGE Architecture (Residual skip-connections, LayerNorm)
- Group 5: Hyperparameters and Capacity
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

def run_all():
    data = torch.load(config.GRAPH_PT, weights_only=False)
    feat_path = "experiments/ml_improvement/extended_features.pt"
    assert Path(feat_path).exists(), f"Features file {feat_path} does not exist yet."
    x_29 = torch.load(feat_path, weights_only=False)
    print(f"[data] Loaded 29-dim feature tensor: shape={x_29.shape}")

    undir_edge_index = to_undirected(data.edge_index)
    print(f"[data] Edge index: directed={data.edge_index.size(1):,}, bidirectional={undir_edge_index.size(1):,}")

    # --- EXP_002: Extended 29 Features + Directed Graph (Isolates Feature Effect)
    print("\n" + "="*80)
    print("RUNNING EXP_002: 29 Features on Directed Topology")
    print("="*80)
    run_experiment(
        experiment_id="EXP_002_FEAT29_DIR",
        description="GraphSAGE with 29 engineered AML features on directed graph",
        x=x_29,
        edge_index=data.edge_index,
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
        feature_desc="29 engineered AML features",
        graph_desc="Directed 1.01M aggregated edges",
    )

    # --- EXP_003: Extended 29 Features + Bidirectional Topology (Synergy)
    print("\n" + "="*80)
    print("RUNNING EXP_003: 29 Features + Bidirectional Message Passing")
    print("="*80)
    run_experiment(
        experiment_id="EXP_003_FEAT29_UNDIR",
        description="GraphSAGE with 29 AML features + bidirectional message passing",
        x=x_29,
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
        feature_desc="29 engineered AML features",
        graph_desc="Bidirectional 2.03M edges (undirected)",
    )

    # --- EXP_004: Extended 29 Features + Bidirectional + Residual Skip Connections
    print("\n" + "="*80)
    print("RUNNING EXP_004: 29 Features + Bidirectional + Residual Skip-Connections")
    print("="*80)
    run_experiment(
        experiment_id="EXP_004_RESIDUALS",
        description="GraphSAGE with 29 features + bidirectional + residual skip connections",
        x=x_29,
        edge_index=undir_edge_index,
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        model_type="sage",
        hidden_channels=128,
        num_layers=3,
        dropout=0.2,
        use_residual=True,
        norm_type="batch",
        loss_type="focal",
        focal_alpha=0.75,
        focal_gamma=2.0,
        lr=5e-3,
        weight_decay=5e-4,
        epochs=60,
        seed=42,
        feature_desc="29 engineered AML features",
        graph_desc="Bidirectional 2.03M edges (undirected)",
    )

    # --- EXP_005: Architecture Capacity (Hidden Dim 256)
    print("\n" + "="*80)
    print("RUNNING EXP_005: Residual GraphSAGE with Hidden Dim 256")
    print("="*80)
    run_experiment(
        experiment_id="EXP_005_CAPACITY_256",
        description="Residual GraphSAGE with hidden_channels=256 and 29 features",
        x=x_29,
        edge_index=undir_edge_index,
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        model_type="sage",
        hidden_channels=256,
        num_layers=3,
        dropout=0.2,
        use_residual=True,
        norm_type="batch",
        loss_type="focal",
        focal_alpha=0.75,
        focal_gamma=2.0,
        lr=3e-3,
        weight_decay=5e-4,
        epochs=60,
        seed=42,
        feature_desc="29 engineered AML features",
        graph_desc="Bidirectional 2.03M edges (undirected)",
    )

    # --- EXP_006: Class Imbalance Loss Tuning (Focal alpha=0.80, gamma=2.0)
    print("\n" + "="*80)
    print("RUNNING EXP_006: Focal Loss Imbalance Tuning (alpha=0.80)")
    print("="*80)
    run_experiment(
        experiment_id="EXP_006_FOCAL_A80",
        description="Residual GraphSAGE with focal alpha=0.80 to up-weight illicit class",
        x=x_29,
        edge_index=undir_edge_index,
        y=data.y,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        model_type="sage",
        hidden_channels=128,
        num_layers=3,
        dropout=0.2,
        use_residual=True,
        norm_type="batch",
        loss_type="focal",
        focal_alpha=0.80,
        focal_gamma=2.0,
        lr=5e-3,
        weight_decay=5e-4,
        epochs=60,
        seed=42,
        feature_desc="29 engineered AML features",
        graph_desc="Bidirectional 2.03M edges (undirected)",
    )

    print_summary_table()

if __name__ == "__main__":
    run_all()
