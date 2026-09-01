"""
Live demo driver for the pitch.

Runs the trained model over the whole payment network and walks through ONE
flagged account end-to-end: the fraud score, then the GNNExplainer audit trail
naming the exact accounts and transactions behind the flag. Designed to be run
live in front of an audience.

    python demo.py                 # picks a flagged account with a rich sub-graph
    python demo.py --node 496618   # demo a specific account
"""
from __future__ import annotations

import argparse
import time

import torch

import config
from explain import explain_node, load_everything, report, score_all


BANNER = r"""
==============================================================================
   EXPLAINABLE GNN FOR ANTI-MONEY LAUNDERING
   Accounts = nodes | Transactions = edges | GraphSAGE + GNNExplainer
==============================================================================
"""


def pick_demo_node(model, data, id_to_key, device, probs):
    """Prefer a high-scoring account whose flag involves OTHER accounts
    (a multi-hop laundering structure), not just a self-loop — better to show."""
    from torch_geometric.explain import Explainer, GNNExplainer  # noqa: F401
    candidates = torch.topk(probs, 25).indices.tolist()
    best_node, best_incoming = candidates[0], -1
    ei = data.edge_index
    for node in candidates:
        # count distinct in-neighbours (fan-in) within 1 hop
        incoming = int((ei[1] == node).sum())
        if incoming > best_incoming:
            best_incoming, best_node = incoming, node
    return best_node


def _preflight():
    """Fail early with a helpful message if demo artifacts are missing."""
    missing = []
    if not config.GRAPH_PT.exists():
        missing.append(f"  - processed graph: run  python preprocess.py")
    if not (config.CHECKPOINT_DIR / "best_model.pt").exists():
        missing.append(f"  - trained model:    run  python train.py")
    if missing:
        print("\n[!] Cannot start the demo — missing artifacts:")
        print("\n".join(missing))
        print("\nRun the step(s) above once, then re-run the demo.\n")
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description="Live AML demo driver.")
    ap.add_argument("--node", type=int, default=None,
                    help="explain a specific account node id")
    args = ap.parse_args()

    _preflight()

    print(BANNER)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[1/4] Loading trained model and the payment-network graph ...", flush=True)
    data, model, id_to_key = load_everything(device)
    print(f"      network: {data.num_nodes:,} accounts, "
          f"{data.num_edges:,} money-flow edges", flush=True)

    print("[2/4] Scoring every account for laundering risk ...", flush=True)
    t0 = time.time()
    probs = score_all(model, data, device)
    n_flagged = int((probs >= 0.5).sum())
    print(f"      scored {data.num_nodes:,} accounts in {time.time()-t0:.1f}s "
          f"| {n_flagged:,} above 0.5 risk", flush=True)

    node_id = args.node if args.node is not None else \
        pick_demo_node(model, data, id_to_key, device, probs)
    print(f"[3/4] Highest-risk structural case -> account "
          f"{id_to_key.get(node_id, node_id)} "
          f"(risk {float(probs[node_id]):.3f})", flush=True)

    print(f"[4/4] Running GNNExplainer on the account's 3-hop neighbourhood "
          f"for the audit trail ...", flush=True)
    t1 = time.time()
    explanation, subset, sub_ei = explain_node(model, data, node_id, device)
    print(f"      explanation computed in {time.time()-t1:.1f}s", flush=True)
    report(explanation, node_id, float(probs[node_id]), id_to_key,
           subset, sub_ei)

    print("\nDemo complete. Audit figure saved under outputs/.", flush=True)


if __name__ == "__main__":
    main()
