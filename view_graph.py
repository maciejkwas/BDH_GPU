from __future__ import annotations

import argparse
import json
from pathlib import Path


def _build_nx_graph(data):
    try:
        import networkx as nx
    except Exception as e:
        raise SystemExit(
            "NetworkX is required. Install it with: pip install networkx matplotlib"
        ) from e

    G = nx.Graph()
    # Add nodes with type attribute
    for n in data.get("nodes", []):
        G.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
    # Add edges with weight
    for e in data.get("edges", []):
        G.add_edge(e["source"], e["target"], weight=float(e.get("weight", 1.0)))
    return G


def _subset_graph(G, max_nodes: int | None):
    if not max_nodes or G.number_of_nodes() <= max_nodes:
        return G
    # Keep top nodes by degree (simple heuristic)
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    keep = set(n for n, _ in deg[:max_nodes])
    return G.subgraph(keep).copy()


def render_matplotlib(G, out: str | None, layout: str = "spring"):
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except Exception as e:
        raise SystemExit(
            "Matplotlib is required. Install it with: pip install matplotlib"
        ) from e

    # Layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42, k=None)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Colors by type
    node_types = nx.get_node_attributes(G, "type")
    colors = [
        "tab:blue" if node_types.get(n) == "neuron" else "tab:orange" for n in G.nodes
    ]

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color=colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3)
    plt.axis("off")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, bbox_inches="tight", dpi=200)
        print(f"Saved preview to {out}")
    else:
        plt.show()


def render_pyvis(G, out_html: str):
    try:
        from pyvis.network import Network
    except Exception as e:
        raise SystemExit(
            "PyVis is required. Install it with: pip install pyvis jinja2"
        ) from e

    net = Network(height="800px", width="100%", directed=False, notebook=False)
    for n, attrs in G.nodes(data=True):
        color = "#1f77b4" if attrs.get("type") == "neuron" else "#ff7f0e"
        net.add_node(n, label=str(n), color=color, **attrs)
    for u, v, attrs in G.edges(data=True):
        net.add_edge(u, v, value=float(attrs.get("weight", 1.0)))
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    # Avoid notebook mode to prevent template None errors in some envs
    try:
        net.write_html(out_html, open_browser=False, notebook=False)
    except Exception as e:
        raise SystemExit(
            "Failed to write HTML with PyVis. Try: pip install --upgrade pyvis jinja2"
        ) from e
    print(f"Saved interactive HTML to {out_html}")


def main():
    ap = argparse.ArgumentParser(description="Preview dragon graph.json")
    ap.add_argument("--graph", default="graph.json", help="Path to graph JSON")
    ap.add_argument(
        "--max-nodes", type=int, default=800, help="Limit nodes for plotting"
    )
    ap.add_argument(
        "--layout",
        choices=["spring", "kamada_kawai"],
        default="spring",
        help="Layout algorithm for matplotlib",
    )
    ap.add_argument("--out", default=None, help="Save PNG instead of showing window")
    ap.add_argument(
        "--html",
        default=None,
        help="Also export interactive HTML (requires pyvis). Example: graph.html",
    )
    args = ap.parse_args()

    with open(args.graph, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = _build_nx_graph(data)
    Gs = _subset_graph(G, args.max_nodes)

    # Matplotlib render
    render_matplotlib(Gs, out=args.out, layout=args.layout)

    # Optional HTML export
    if args.html:
        render_pyvis(Gs, out_html=args.html)


if __name__ == "__main__":
    main()
