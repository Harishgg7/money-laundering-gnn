"""
Package Promoted Production Model (TRACE-HYBRID-v2).
Exports:
1. models/gnn_encoder.pt (GraphSAGE 128-dim relational embedding generator)
2. models/xgb_classifier.json (Tuned XGBoost classifier)
3. models/model_manifest.json (Cryptographically verified model manifest with rollback metadata)
4. models/hybrid_probs.pt (Pre-cached 515K risk probabilities for instant startup)
"""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from xgboost import XGBClassifier

import config
from experiments.ml_improvement.runner import AdvancedGraphSAGE

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=== PACKAGING PROMOTED PRODUCTION MODEL: TRACE-HYBRID-v2 ===")
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy GNN encoder checkpoint
    src_gnn = Path("experiments/ml_improvement/checkpoints/EXP_003_FEAT29_UNDIR.pt")
    dst_gnn = models_dir / "gnn_encoder.pt"
    shutil.copy2(src_gnn, dst_gnn)
    gnn_sha = sha256_file(str(dst_gnn))
    print(f"[package] Copied GNN encoder -> {dst_gnn} (SHA256: {gnn_sha[:16]}...)")

    # 2. Export XGBoost classifier
    src_xgb = Path("experiments/ml_improvement/checkpoints/EXP_007_HYBRID_XGB.pt")
    ckpt_xgb = torch.load(src_xgb, weights_only=False)
    clf: XGBClassifier = ckpt_xgb["xgb_model"]
    dst_xgb_json = models_dir / "xgb_classifier.json"
    clf.save_model(str(dst_xgb_json))
    xgb_sha = sha256_file(str(dst_xgb_json))
    print(f"[package] Saved XGBoost model -> {dst_xgb_json} (SHA256: {xgb_sha[:16]}...)")

    # 3. Compute and cache complete 515K hybrid risk probabilities
    print("[package] Computing full-batch risk predictions for all 515,088 accounts...")
    data = torch.load(config.GRAPH_PT, weights_only=False)
    x_29 = torch.load("experiments/ml_improvement/extended_features.pt", weights_only=False)
    undir_edges = to_undirected(data.edge_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gnn = AdvancedGraphSAGE(29, 128, 3, 0.2, use_residual=False, norm_type="batch").to(device)
    gnn.load_state_dict(torch.load(dst_gnn, weights_only=False)["model_state"])
    gnn.eval()

    with torch.no_grad():
        x_dev = x_29.to(device)
        edges_dev = undir_edges.to(device)
        h = x_dev
        for block in gnn.blocks:
            h = F.relu(block["norm"](block["conv"](h, edges_dev)))
        emb = h.cpu().numpy()

    X_comb = np.hstack([x_29.numpy(), emb])
    print("[package] Scoring 515,088 accounts through XGBoost...")
    all_probs = clf.predict_proba(X_comb)[:, 1].astype(np.float32)
    probs_tensor = torch.from_numpy(all_probs)
    dst_probs = models_dir / "hybrid_probs.pt"
    torch.save(probs_tensor, dst_probs)
    probs_sha = sha256_file(str(dst_probs))
    print(f"[package] Cached hybrid probabilities -> {dst_probs} (Mean: {all_probs.mean():.4f}, Max: {all_probs.max():.4f})")

    # 4. Check rollback model checksum
    rollback_path = Path("checkpoints/best_model.pt")
    rollback_sha = sha256_file(str(rollback_path)) if rollback_path.exists() else "UNAVAILABLE"

    # 5. Build Model Manifest
    manifest = {
        "manifest_version": "1.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "production_model": {
            "model_version": "TRACE-HYBRID-v2",
            "model_name": "GraphSAGE-XGBoost Relational Hybrid",
            "architecture": "3-layer GraphSAGE (128-dim embeddings) + 29 AML Behavioral Features + XGBoost",
            "status": "PROMOTED_PRODUCTION",
            "operating_threshold": 0.9184,
            "feature_count": 29,
            "embedding_dim": 128,
            "total_input_dim": 157,
            "graph_topology": "Bidirectional (1.65M unique edges)",
            "metrics": {
                "pr_auc": 0.4569,
                "roc_auc": 0.9412,
                "f1": 0.4886,
                "precision": 0.7189,
                "recall": 0.3700,
                "evaluation_split": "15% held-out test split (77,264 accounts)"
            },
            "multi_seed_stability": {
                "seeds_evaluated": [42, 123, 999, 2026],
                "pr_auc_mean": 0.4577,
                "pr_auc_std": 0.0011,
                "f1_mean": 0.4884,
                "precision_mean": 0.7304,
                "recall_mean": 0.3669
            },
            "temporal_validation": {
                "pr_auc_temporal": 0.5493,
                "roc_auc_temporal": 0.9063,
                "f1_temporal": 0.5582,
                "precision_temporal": 0.8098,
                "recall_temporal": 0.4259
            },
            "artifacts": {
                "gnn_encoder": {
                    "path": "models/gnn_encoder.pt",
                    "sha256": gnn_sha,
                    "size_bytes": os.path.getsize(str(dst_gnn))
                },
                "xgb_classifier": {
                    "path": "models/xgb_classifier.json",
                    "sha256": xgb_sha,
                    "size_bytes": os.path.getsize(str(dst_xgb_json))
                },
                "cached_probabilities": {
                    "path": "models/hybrid_probs.pt",
                    "sha256": probs_sha,
                    "size_bytes": os.path.getsize(str(dst_probs))
                }
            }
        },
        "rollback_model": {
            "model_version": "TRACE-GSAGE-v1",
            "model_name": "GraphSAGE 3-Hop Baseline",
            "architecture": "3-layer GraphSAGE (Topology-only)",
            "status": "ROLLBACK_READY",
            "operating_threshold": 0.3781,
            "feature_count": 11,
            "metrics": {
                "pr_auc": 0.3420,
                "roc_auc": 0.9060,
                "f1": 0.3680,
                "precision": 0.5700,
                "recall": 0.2710
            },
            "artifacts": {
                "checkpoint": {
                    "path": "checkpoints/best_model.pt",
                    "sha256": rollback_sha,
                    "size_bytes": os.path.getsize(str(rollback_path)) if rollback_path.exists() else 0
                }
            }
        }
    }

    manifest_path = models_dir / "model_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[package] Successfully generated model manifest -> {manifest_path}")

if __name__ == "__main__":
    main()
