"""
In thống kê của knowledge graph do MiniRAG tạo ở bước index.

Graph được NetworkXStorage lưu tại:
    <workingdir>/graph_chunk_entity_relation.graphml

Cách dùng (chạy từ thư mục gốc repo):
    python3 reproduce/Step_0_A_graph_statistic.py --workingdir ./LiHua-World
"""

import argparse
import os
import sys
from collections import Counter
from statistics import median


def get_args():
    parser = argparse.ArgumentParser(
        description="Print statistics for the MiniRAG knowledge graph"
    )
    parser.add_argument("--workingdir", type=str, default="./LiHua-World")
    parser.add_argument(
        "--graph-file",
        type=str,
        default=None,
        help="Đường dẫn GraphML; mặc định <workingdir>/graph_chunk_entity_relation.graphml",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Số node/type có degree cao nhất cần in (mặc định: 10)",
    )
    return parser.parse_args()


def clean_attr(value, default="UNKNOWN"):
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip().strip('"').strip("'").strip()
        return value or default
    return str(value)


def percent(part, total):
    return 0.0 if total == 0 else part / total * 100


def print_counter(counter, total, top_k, title):
    print(f"\n{title}:")
    if not counter:
        print("  (none)")
        return
    for name, count in counter.most_common(top_k):
        print(f"  {name:<40} {count:>8}  ({percent(count, total):5.2f}%)")


def main():
    args = get_args()
    graph_file = args.graph_file or os.path.join(
        args.workingdir, "graph_chunk_entity_relation.graphml"
    )

    if not os.path.exists(graph_file):
        print(f"[ERROR] Không tìm thấy graph: {graph_file}")
        print("Hãy chạy reproduce/Step_0_index.py trước để tạo graph.")
        sys.exit(1)

    try:
        import networkx as nx
    except ImportError:
        print("[ERROR] Thiếu package networkx. Cài bằng: pip install networkx")
        sys.exit(1)

    try:
        graph = nx.read_graphml(graph_file)
    except Exception as exc:
        print(f"[ERROR] Không đọc được GraphML: {graph_file}")
        print(f"        {exc}")
        sys.exit(1)

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    degrees = [degree for _, degree in graph.degree()]
    total_degree = sum(degrees)

    if graph.is_directed():
        components = list(nx.weakly_connected_components(graph))
        graph_type = "directed"
    else:
        components = list(nx.connected_components(graph))
        graph_type = "undirected"

    largest_component = max((len(c) for c in components), default=0)
    isolated_nodes = sum(1 for d in degrees if d == 0)
    self_loops = nx.number_of_selfloops(graph)

    print("=" * 72)
    print("MiniRAG knowledge graph statistics")
    print("=" * 72)
    print(f"Graph file          : {os.path.abspath(graph_file)}")
    print(f"File size           : {os.path.getsize(graph_file) / 1024:.1f} KB")
    print(f"Graph type          : {graph_type}")
    print()
    print(f"Nodes               : {n_nodes}")
    print(f"Edges               : {n_edges}")
    print(f"Self-loops          : {self_loops}")
    print(f"Density             : {nx.density(graph):.6f}")
    print(f"Average degree      : {total_degree / n_nodes:.2f}" if n_nodes else "Average degree      : 0")
    print(f"Median degree       : {median(degrees):.2f}" if degrees else "Median degree       : 0")
    print(f"Max degree          : {max(degrees) if degrees else 0}")
    print(f"Isolated nodes      : {isolated_nodes}")
    print(f"Connected components: {len(components)}")
    print(f"Largest component   : {largest_component} nodes ({percent(largest_component, n_nodes):5.2f}%)")

    node_types = Counter()
    for _, data in graph.nodes(data=True):
        node_types[clean_attr(data.get("entity_type"))] += 1
    print_counter(node_types, n_nodes, args.top_k, "Node types")

    print(f"\nTop {args.top_k} nodes by degree:")
    top_nodes = sorted(graph.degree(), key=lambda item: (-item[1], item[0]))[: args.top_k]
    if not top_nodes:
        print("  (none)")
    for node, degree in top_nodes:
        print(f"  {node:<52} {degree:>8}")

    print("=" * 72)


if __name__ == "__main__":
    main()
