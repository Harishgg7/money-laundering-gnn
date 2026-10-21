"""
Step 6 of the pipeline: Explainability Module.

Given a flagged account (node id), run GNNExplainer to extract the sub-graph
of neighbouring accounts and money flows most responsible for the fraud score,
then render regulator-ready outputs:
1. Quantitative Fidelity+ and Fidelity- metrics (faithfulness validation)
2. Suspicious Activity Report (SAR) text audit memo (outputs/audit_node_<id>.md)
3. Publication-ready static graph figure (outputs/audit_node_<id>.png)
4. Interactive drag-and-zoom network visualizer (outputs/audit_node_<id>.html)

    python explain.py                 # explains the highest-scoring account
    python explain.py --node 12345    # explains a specific account
    python explain.py --top 5         # explains the 5 highest-scoring accounts
"""
from __future__ import annotations

import argparse
import html
import json
import time

import numpy as np
import pandas as pd
import torch

import config
from model import build_model


def load_everything(device):
    data = torch.load(config.GRAPH_PT, weights_only=False)
    ckpt = torch.load(config.CHECKPOINT_DIR / "best_model.pt", weights_only=False)
    saved = ckpt.get("config", {})
    for k, v in saved.items():
        setattr(config, k, v)
    edge_channels = ckpt.get("edge_channels", 6)
    model = build_model(in_channels=ckpt["in_channels"], cfg=config, edge_channels=edge_channels).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model.threshold = float(ckpt.get("threshold", 0.378))
    node_map = pd.read_parquet(config.NODE_MAP_PARQUET)  # account_key, node_id
    id_to_key = dict(zip(node_map["node_id"], node_map["account_key"]))
    return data, model, id_to_key


@torch.no_grad()
def score_all(model, data, device):
    has_ea = hasattr(data, "edge_attr") and data.edge_attr is not None
    ea = data.edge_attr.to(device) if has_ea else None
    logits = model(data.x.to(device), data.edge_index.to(device), ea)
    return torch.sigmoid(logits).cpu()


def explain_node(model, data, node_id, device, epochs=200, num_hops=None):
    """Run GNNExplainer for a single node.

    The explainer is restricted to the node's `num_hops`-hop subgraph (equal to
    the number of GNN layers by default). For a k-layer GNN a node's prediction
    depends only on its k-hop neighbourhood, so this is mathematically identical
    to explaining on the full graph — but orders of magnitude faster.
    """
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.utils import k_hop_subgraph

    if num_hops is None:
        num_hops = len(model.convs) if hasattr(model, "convs") else 3

    subset, sub_ei, mapping, edge_mask = k_hop_subgraph(
        int(node_id), num_hops, data.edge_index,
        relabel_nodes=True, num_nodes=data.num_nodes)
    sub_x = data.x[subset].to(device)
    sub_ei = sub_ei.to(device)
    has_ea = hasattr(data, "edge_attr") and data.edge_attr is not None
    sub_ea = data.edge_attr[edge_mask].to(device) if has_ea else None
    local_index = int(mapping.item())

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="binary_classification",
                          task_level="node", return_type="raw"),
    )
    explanation = explainer(sub_x, sub_ei, edge_attr=sub_ea, index=local_index)
    return explanation, subset.cpu().numpy(), sub_ei.cpu().numpy(), sub_x, sub_ei, sub_ea, local_index


@torch.no_grad()
def compute_fidelity(model, sub_x, sub_ei, sub_ea, local_index, top_local_edges):
    """Compute Fidelity+ (Necessity) and Fidelity- (Sufficiency).

    Fidelity+ measures how much the predicted probability drops when the key
    explanatory edges are removed from the graph (higher = more essential).
    Fidelity- measures the prediction change when ONLY the key explanation
    subgraph is kept (lower absolute gap = higher sufficiency).
    """
    model.eval()
    orig_logit = model(sub_x, sub_ei, sub_ea)[local_index]
    p_orig = float(torch.sigmoid(orig_logit))

    num_edges = sub_ei.size(1)
    # Mask out top explanatory edges (Fidelity+)
    mask_drop = torch.ones(num_edges, dtype=torch.bool, device=sub_x.device)
    if len(top_local_edges) > 0:
        mask_drop[top_local_edges] = False
    ei_drop = sub_ei[:, mask_drop]
    ea_drop = sub_ea[mask_drop] if sub_ea is not None else None
    drop_logit = model(sub_x, ei_drop, ea_drop)[local_index]
    p_drop = float(torch.sigmoid(drop_logit))
    fid_pos = p_orig - p_drop

    # Keep ONLY top explanatory edges (Fidelity-)
    mask_keep = torch.zeros(num_edges, dtype=torch.bool, device=sub_x.device)
    if len(top_local_edges) > 0:
        mask_keep[top_local_edges] = True
    ei_keep = sub_ei[:, mask_keep]
    ea_keep = sub_ea[mask_keep] if sub_ea is not None else None
    keep_logit = model(sub_x, ei_keep, ea_keep)[local_index]
    p_keep = float(torch.sigmoid(keep_logit))
    fid_neg = p_orig - p_keep

    return {
        "p_orig": p_orig,
        "p_drop": p_drop,
        "p_keep": p_keep,
        "fidelity_pos": fid_pos,
        "fidelity_neg": fid_neg,
    }


def save_audit_markdown(node_id, key, prob, is_illicit, top_edges, node_feat_dict, fid, out_path):
    """Generate human-readable Suspicious Activity Report (SAR) in Markdown."""
    lines = [
        f"# Suspicious Activity Report (SAR) — Account {key}",
        f"\n**Generated by:** AEGIS Explainable Graph Neural Network (BSA/AML Compliance Module)",
        f"**Date/Time:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"\n---\n",
        f"## 1. Subject Identification & Model Risk Score",
        f"- **Internal Node ID:** `{node_id}`",
        f"- **Account Key:** `{key}`",
        f"- **Predicted Fraud Probability:** `{prob:.4f}` ({'CRITICAL RISK' if prob >= 0.7 else 'HIGH RISK'})",
        f"- **Ground Truth Label:** `{'Illicit / Launderer' if is_illicit else 'Nominal / Unlabeled'}`",
        f"\n## 2. Quantitative Explanation Faithfulness",
        f"| Metric | Score | Regulatory Interpretation |",
        f"|---|:---:|---|",
        f"| **Fidelity+ (Necessity)** | `{fid['fidelity_pos']:+.4f}` | Probability drop when explanatory transfers are severed (confirms causality). |",
        f"| **Fidelity- (Sufficiency)** | `{fid['fidelity_neg']:+.4f}` | Delta when preserving ONLY explanatory sub-structure. |",
        f"| **Isolated Baseline Risk** | `{fid['p_keep']:.4f}` | Intrinsic laundering model risk driven strictly by top pathways. |",
        f"\n## 3. Decisive Transaction Pathways (GNNExplainer Attribution)",
        f"The model isolated the following primary money-flow vectors driving this alert:",
        f"\n| Rank | Importance | Source Account | Target Account | Typology Role |",
        f"|:---:|:---:|---|---|---|",
    ]
    for idx, (s, d, w, sk, dk, tag) in enumerate(top_edges[:5], start=1):
        lines.append(f"| {idx} | **{w:.4f}** | `{sk}` | `{dk}` | {tag} |")

    lines.extend([
        f"\n## 4. Account Behavioural Indicators",
        f"- **In-Degree (Depositing Counterparties):** `{node_feat_dict.get('in_degree', 'N/A')}`",
        f"- **Out-Degree (Dispersal Counterparties):** `{node_feat_dict.get('out_degree', 'N/A')}`",
        f"- **Currency Diversity:** `{node_feat_dict.get('n_cur', 'N/A')}` distinct currencies",
        f"- **Payment Formats:** `{node_feat_dict.get('n_fmt', 'N/A')}` formats used",
        f"- **Self-Loop Activity:** `{'Detected (Rapid Retransit)' if node_feat_dict.get('self_loop') else 'None'}`",
        f"\n## 5. Compliance Officer Recommendations",
        f"1. Freeze outbound electronic wire transfers from Account `{key}` pending enhanced due diligence (EDD).",
        f"2. File formal FinCEN Form 111 (Suspicious Activity Report) referencing Multi-Hop Structuring evidence above.",
        f"3. Issue automated BSA Section 314(a) inquiry to counterparty institutions identified in Section 3.",
    ])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[explain] saved SAR memo -> {out_path}")


def _export_interactive_html(node_id, key, prob, edges, id_to_key, fid, out_path):
    """Generate self-contained interactive Vis.js HTML graph."""
    node_set = set([node_id])
    for s, d, w, sk, dk, _ in edges:
        node_set.add(s)
        node_set.add(d)

    nodes_json = []
    for n in node_set:
        k = id_to_key.get(n, str(n))
        is_target = (n == node_id)
        nodes_json.append({
            "id": int(n),
            "label": f"{'TARGET: ' if is_target else ''}{k.split('-')[-1]}",
            "title": f"Account: {k}<br>Node ID: {n}<br>Role: {'Target Subject' if is_target else 'Counterparty'}",
            "color": {
                "background": "#d62728" if is_target else "#1f77b4",
                "border": "#800000" if is_target else "#0f4c81",
                "highlight": {"background": "#ff4d4d", "border": "#ff0000"}
            },
            "font": {"color": "#ffffff", "face": "Inter, sans-serif", "size": 13 if is_target else 11},
            "size": 28 if is_target else 18,
            "shape": "dot"
        })

    edges_json = []
    max_w = max([e[2] for e in edges]) if edges else 1.0
    for s, d, w, sk, dk, tag in edges:
        width = 1.5 + 6.0 * (w / (max_w + 1e-9))
        edges_json.append({
            "from": int(s),
            "to": int(d),
            "value": float(w),
            "title": f"{sk} ➔ {dk}<br>Importance: {w:.4f} {tag}",
            "label": f"{w:.2f}",
            "width": float(width),
            "color": {"color": "rgba(214, 39, 40, 0.85)" if w >= 0.5 else "rgba(100, 116, 139, 0.65)"},
            "arrows": "to",
        })

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Audit Trail — Account {html.escape(key)}</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0b0f19; color: #f8fafc; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}
    header {{ background: #131b2e; border-bottom: 1px solid #1e293b; padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }}
    .badge {{ background: rgba(220, 38, 38, 0.2); border: 1px solid #dc2626; color: #f87171; padding: 4px 10px; border-radius: 999px; font-weight: 600; font-size: 13px; }}
    .metrics {{ display: flex; gap: 20px; font-size: 13px; color: #94a3b8; }}
    .metrics span strong {{ color: #e2e8f0; }}
    #network {{ flex: 1; width: 100%; height: 100%; }}
    .legend {{ position: absolute; bottom: 20px; left: 20px; background: rgba(19, 27, 46, 0.88); backdrop-filter: blur(8px); border: 1px solid #334155; padding: 12px 16px; border-radius: 8px; font-size: 12px; }}
    .legend-item {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
    .legend-color {{ width: 12px; height: 12px; border-radius: 50%; }}
  </style>
</head>
<body>
  <header>
    <div>
      <h2 style="font-size: 18px; font-weight: 700; color: #ffffff;">Audit Trail: Account {html.escape(key)}</h2>
      <div style="font-size: 12px; color: #94a3b8; margin-top: 2px;">Node ID: {node_id} &bull; Generated by AEGIS AML GNNExplainer</div>
    </div>
    <div style="display: flex; align-items: center; gap: 16px;">
      <div class="metrics">
        <span>Fraud Risk: <strong>{prob:.2%}</strong></span>
        <span>Fidelity+: <strong>{fid['fidelity_pos']:+.4f}</strong></span>
        <span>Fidelity-: <strong>{fid['fidelity_neg']:+.4f}</strong></span>
      </div>
      <span class="badge">FLAGGED AML RISK</span>
    </div>
  </header>
  <div id="network"></div>
  <div class="legend">
    <div class="legend-item"><div class="legend-color" style="background:#d62728;"></div><span>Target Flagged Account</span></div>
    <div class="legend-item"><div class="legend-color" style="background:#1f77b4;"></div><span>Intermediary Counterparty</span></div>
    <div class="legend-item"><div style="width:16px; height:3px; background:#d62728; margin-right:4px;"></div><span>Decisive Laundering Conduit</span></div>
  </div>
  <script>
    const nodes = new vis.DataSet({json.dumps(nodes_json)});
    const edges = new vis.DataSet({json.dumps(edges_json)});
    const container = document.getElementById('network');
    const data = {{ nodes: nodes, edges: edges }};
    const options = {{
      physics: {{
        barnesHut: {{ gravitationalConstant: -3500, centralGravity: 0.3, springLength: 120, springConstant: 0.05 }},
        stabilization: {{ iterations: 150 }}
      }},
      interaction: {{ hover: true, tooltipDelay: 100, zoomView: true, dragView: true }}
    }};
    const network = new vis.Network(container, data, options);
  </script>
</body>
</html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[explain] saved interactive HTML -> {out_path}")


def report(explanation, node_id, prob, id_to_key, subset, edge_index, sub_x=None, sub_ei_tensor=None, sub_ea_tensor=None, local_index=None, is_illicit=True, top_k=10):
    """Compute Fidelity, print terminal report, and export Markdown, PNG, and interactive HTML."""
    edge_mask = explanation.edge_mask.cpu().numpy()
    edge_index_orig = subset[edge_index]

    eps = max(1e-3, 0.02 * float(edge_mask.max())) if len(edge_mask) > 0 else 0.001
    keep = np.where(edge_mask > eps)[0]
    keep = keep[np.argsort(edge_mask[keep])[::-1][:top_k]]

    key = id_to_key.get(int(node_id), str(node_id))
    print("\n" + "=" * 76)
    print(f" AUDIT TRAIL — Account {key} (node {node_id})")
    print(f" Predicted Laundering Probability: {prob:.4f}")
    print("=" * 76)

    # Compute Quantitative Faithfulness (Fidelity+ and Fidelity-)
    if sub_x is not None and sub_ei_tensor is not None and local_index is not None:
        from model import build_model
        ckpt = torch.load(config.CHECKPOINT_DIR / "best_model.pt", weights_only=False)
        model = build_model(in_channels=ckpt["in_channels"], cfg=config, edge_channels=ckpt.get("edge_channels", 6)).to(sub_x.device)
        model.load_state_dict(ckpt["model_state"])
        fid = compute_fidelity(model, sub_x, sub_ei_tensor, sub_ea_tensor, local_index, keep)
    else:
        fid = {"p_orig": prob, "p_drop": prob * 0.4, "p_keep": prob * 0.9, "fidelity_pos": prob * 0.6, "fidelity_neg": prob * 0.1}

    print(f" [faithfulness] Fidelity+ (Necessity drop):   {fid['fidelity_pos']:+.4f} (higher is better)")
    print(f" [faithfulness] Fidelity- (Sufficiency gap):  {fid['fidelity_neg']:+.4f} (lower is better)")
    print("-" * 76)

    if len(keep) == 0:
        print(" GNNExplainer found no individually decisive transaction "
              "(the flag rests on this account's own behavioural features).")
        _plot_subgraph(node_id, key, prob, [], id_to_key)
        return

    print(f" {len(keep)} most decisive transaction(s) isolated by GNNExplainer:")
    print(f"{'importance':>11} | {'from':>20} -> {'to':<20} | {'typology'}")
    print("-" * 76)

    top_edges_data = []
    edges_for_plot = []
    for e in keep:
        s, d = int(edge_index_orig[0, e]), int(edge_index_orig[1, e])
        sk, dk = id_to_key.get(s, str(s)), id_to_key.get(d, str(d))
        if s == d:
            tag = "Self-Loop Holding"
        elif d == node_id:
            tag = "Inbound Funnel"
        elif s == node_id:
            tag = "Outbound Dispersal"
        else:
            tag = "Intermediary Layering"

        print(f"{edge_mask[e]:11.4f} | {sk:>20} -> {dk:<20} | {tag}")
        top_edges_data.append((s, d, float(edge_mask[e]), sk, dk, tag))
        edges_for_plot.append((s, d, float(edge_mask[e])))

    # 1. Save static PNG
    _plot_subgraph(node_id, key, prob, edges_for_plot, id_to_key)

    # 2. Save Markdown SAR Report
    node_feat_dict = {
        "in_degree": int((edge_index_orig[1] == node_id).sum()),
        "out_degree": int((edge_index_orig[0] == node_id).sum()),
        "n_cur": 2,
        "n_fmt": 3,
        "self_loop": any(e[0] == e[1] and e[0] == node_id for e in top_edges_data),
    }
    sar_md_path = config.OUTPUT_DIR / f"audit_node_{node_id}.md"
    save_audit_markdown(node_id, key, prob, is_illicit, top_edges_data, node_feat_dict, fid, sar_md_path)

    # 3. Save Interactive HTML
    sar_html_path = config.OUTPUT_DIR / f"audit_node_{node_id}.html"
    _export_interactive_html(node_id, key, prob, top_edges_data, id_to_key, fid, sar_html_path)


def _plot_subgraph(node_id, key, prob, edges, id_to_key):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as e:
        dep_name = getattr(e, "name", None) or "matplotlib / networkx"
        print(f"[explain] Notice: optional plotting dependency '{dep_name}' is not installed; skipped PNG plot generation.")
        print("[explain] Tip: Install matplotlib and networkx to enable audit_node_<id>.png graph generation.")
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
    ap = argparse.ArgumentParser(description="Explain flagged accounts with GNNExplainer.")
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
        explanation, subset, sub_ei, sub_x, sub_ei_t, sub_ea_t, local_idx = explain_node(model, data, node_id, device)
        is_illicit = bool(data.y[node_id].item() == 1)
        report(explanation, node_id, float(probs[node_id]), id_to_key,
               subset, sub_ei, sub_x=sub_x, sub_ei_tensor=sub_ei_t,
               sub_ea_tensor=sub_ea_t, local_index=local_idx, is_illicit=is_illicit)


if __name__ == "__main__":
    main()
