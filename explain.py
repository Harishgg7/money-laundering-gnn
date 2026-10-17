"""
Step 6 of the pipeline: Explainability Module.

Given a flagged account (node id), run GNNExplainer to extract the sub-graph
of neighbouring accounts and money flows most responsible for the fraud score,
then render it as a regulator-ready audit trail (figure + printed summary).

    python explain.py                 # explains the highest-scoring account
    python explain.py --node 12345    # explains a specific account
    python explain.py --top 5         # explains the 5 highest-scoring accounts
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

import config
from model import build_model


def load_everything(device):
    data = torch.load(config.GRAPH_PT, weights_only=False)
    ckpt = torch.load(config.CHECKPOINT_DIR / "best_model.pt", weights_only=False)
    model = build_model(in_channels=ckpt["in_channels"], cfg=config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    node_map = pd.read_parquet(config.NODE_MAP_PARQUET)  # account_key, node_id
    id_to_key = dict(zip(node_map["node_id"], node_map["account_key"]))
    return data, model, id_to_key


@torch.no_grad()
def score_all(model, data, device):
    logits = model(data.x.to(device), data.edge_index.to(device))
    return torch.sigmoid(logits).cpu()


def explain_node(model, data, node_id, device, epochs=200, num_hops=None):
    """Run GNNExplainer for a single node.

    The explainer is restricted to the node's `num_hops`-hop subgraph (equal to
    the number of GNN layers by default). For a k-layer GNN a node's prediction
    depends only on its k-hop neighbourhood, so this is mathematically identical
    to explaining on the full graph — but orders of magnitude faster, since each
    optimisation step no longer forwards over all 515k nodes.

    Returns (explanation, subset, sub_edge_index) where `subset` maps local
    subgraph node ids back to original account node ids.
    """
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.utils import k_hop_subgraph

    if num_hops is None:
        num_hops = len(model.convs)

    subset, sub_ei, mapping, _ = k_hop_subgraph(
        int(node_id), num_hops, data.edge_index,
        relabel_nodes=True, num_nodes=data.num_nodes)
    sub_x = data.x[subset].to(device)
    sub_ei = sub_ei.to(device)
    local_index = int(mapping.item())  # new index of the seed node in `subset`

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="binary_classification",
                          task_level="node", return_type="raw"),
    )
    explanation = explainer(sub_x, sub_ei, index=local_index)
    return explanation, subset.cpu().numpy(), sub_ei.cpu().numpy()


def report(explanation, node_id, prob, id_to_key, subset, edge_index, top_k=10):
    """Print + plot the most important accounts/flows behind the flag.

    `subset` maps local subgraph node ids -> original account node ids;
    `edge_index` is the subgraph edge index (local ids).
    """
    edge_mask = explanation.edge_mask.cpu().numpy()
    # Map local subgraph node ids back to original account node ids.
    edge_index = subset[edge_index]

    # Keep only edges the explainer found genuinely important (drop the vast
    # majority at ~0, which lie outside the node's computation subgraph).
    eps = max(1e-3, 0.02 * edge_mask.max())
    keep = np.where(edge_mask > eps)[0]
    keep = keep[np.argsort(edge_mask[keep])[::-1][:top_k]]

    key = id_to_key.get(int(node_id), str(node_id))
    print("\n" + "=" * 70)
    print(f" AUDIT TRAIL - account {key}  (node {node_id})")
    print(f" Fraud probability: {prob:.4f}")
    print("=" * 70)
    if len(keep) == 0:
        print(" GNNExplainer found no individually decisive transaction "
              "(the flag rests on this account's own behavioural features).")
        _plot_subgraph(node_id, key, prob, [], id_to_key)
        return
    print(f" {len(keep)} most important transaction(s) driving this flag:")
    print(f"{'importance':>11} | {'from':>18} -> {'to':<18}")
    print("-" * 70)
    edges_for_plot = []
    for e in keep:
        s, d = int(edge_index[0, e]), int(edge_index[1, e])
        sk, dk = id_to_key.get(s, str(s)), id_to_key.get(d, str(d))
        tag = "  (self-loop)" if s == d else ""
        print(f"{edge_mask[e]:11.4f} | {sk:>18} -> {dk:<18}{tag}")
        edges_for_plot.append((s, d, float(edge_mask[e])))

    _plot_subgraph(node_id, key, prob, edges_for_plot, id_to_key)


def _plot_subgraph(node_id, key, prob, edges, id_to_key):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("[explain] Notice: matplotlib is not installed; skipped PNG plot generation.")
        print("[explain] Tip: Install matplotlib to also generate audit_node_<id>.png graph.")
        return

    G = nx.DiGraph()
    for s, d, w in edges:
        G.add_edge(s, d, weight=w)
    if G.number_of_nodes() == 0:
        print("[explain] no edges to plot for this node.")
        return

    pos = nx.spring_layout(G, seed=config.SEED, k=0.9)
    weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
    widths = 1 + 6 * (weights / (weights.max() + 1e-9))
    node_colors = ["#d62728" if n == node_id else "#1f77b4" for n in G.nodes()]

    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700)
    nx.draw_networkx_edges(G, pos, width=widths, edge_color="#888",
                           arrowsize=18, connectionstyle="arc3,rad=0.08")
    labels = {n: id_to_key.get(n, str(n)).split("-")[-1] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    plt.title(f"Audit trail for account {key} (fraud prob {prob:.3f})\n"
              f"red = flagged account, edge width = importance")
    plt.axis("off")
    out = config.OUTPUT_DIR / f"audit_node_{node_id}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[explain] saved figure -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", type=int, default=None,
                    help="account node id to explain")
    ap.add_argument("--top", type=int, default=1,
                    help="explain the N highest-scoring accounts")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, model, id_to_key = load_everything(device)
    probs = score_all(model, data, device)

    if args.node is not None:
        targets = [args.node]
    else:
        targets = torch.topk(probs, args.top).indices.tolist()

    for node_id in targets:
        explanation, subset, sub_ei = explain_node(model, data, node_id, device)
        report(explanation, node_id, float(probs[node_id]), id_to_key,
               subset, sub_ei)


if __name__ == "__main__":
    main()
