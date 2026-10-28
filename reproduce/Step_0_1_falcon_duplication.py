"""
FALCON-KG pre-merge (node duplication) cho MiniRAG — training-free.

Áp dụng phần "trước khi thực hiện core Falcon" trong falcon/idea.md: gộp các node
trùng thực thể **trước** khi chạy FALCON core. Điều kiện gộp (bảo thủ):

  * Cùng ``entity_type`` (đã chuẩn hoá).
  * Tên chuẩn hoá (bỏ dấu, viết thường, bỏ ký tự đặc biệt) bằng nhau, HOẶC
  * Tên chuẩn hoá của node này nằm trong tập ``alternative_names`` của node kia
    (cũng chuẩn hoá).

Trường ``alternative_names`` do bước trích xuất mở rộng (minirag/prompt.py +
minirag/operate.py) sinh ra; nếu index cũ không có trường này thì script tự coi tập
bí danh chỉ gồm chính tên node (chỉ gộp khi tên chuẩn hoá bằng nhau).

Script chạy trên bản copy (``--outputdir``) để giữ baseline, gộp graph rồi đồng bộ lại
``entity_name_vdb`` / ``entities_vdb`` / ``relationships_vdb`` (QA mode="mini" đọc chúng).

Cách dùng (từ thư mục gốc repo):
    python3 reproduce/Step_0_1_falcon_duplication.py \
        --workingdir ./outputs/NDAKG-VNH3045-gpt-5-nano-falcon \
        --outputdir ./outputs/NDAKG-VNH3045-gpt-5-nano-falcon-dedup
"""

import argparse
import asyncio
import os
import shutil
import sys
import unicodedata
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reproduce.config import add_llm_args, EndpointConfig  # noqa: E402
from minirag.prompt import GRAPH_FIELD_SEP  # noqa: E402
from minirag.utils import compute_mdhash_id, split_string_by_multi_markers  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser(
        description="FALCON-KG pre-merge (node duplication) cho index MiniRAG"
    )
    add_llm_args(parser)
    parser.add_argument("--workingdir", type=str, required=True)
    parser.add_argument(
        "--outputdir",
        type=str,
        default=None,
        help="Thư mục ghi kết quả đã gộp (mặc định <workingdir>_dedup)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Xoá --outputdir nếu đã tồn tại trước khi copy",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Chuẩn hoá tên (bỏ dấu, viết thường, bỏ ký tự đặc biệt)
# --------------------------------------------------------------------------- #

def _strip_quotes(value: str) -> str:
    value = (value or "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def strong_normalize(name) -> str:
    """Chuẩn hoá mạnh: bóc nháy -> bỏ dấu (NFD) -> viết thường -> chỉ giữ chữ/số."""
    s = _strip_quotes(name)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower()
    return "".join(c for c in s if c.isalnum())


# --------------------------------------------------------------------------- #
# Graph merge helpers — bám theo ngữ nghĩa của minirag/operate.py
# --------------------------------------------------------------------------- #

def _union_join(a, b) -> str:
    """Gộp 2 chuỗi ``<SEP>``-separated thành chuỗi hợp nhất (sort + dedup)."""
    parts = set()
    for val in (a, b):
        if val is None:
            continue
        for p in split_string_by_multi_markers(str(val), [GRAPH_FIELD_SEP]):
            parts.add(p)
    return GRAPH_FIELD_SEP.join(sorted(parts))


def _to_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def merge_node_data(graph, canonical, other) -> None:
    """Gộp attributes của node ``other`` vào ``canonical`` (giữ entity_type của canonical).

    Mọi attribute do schema lịch sử sinh ra (``time_range``, ``geo_location``,
    ``birth_date``, ``start_time``, ...) đều được union tự động, nên attribute
    mới thêm vào schema không cần sửa lại hàm này.
    """
    cd = graph.nodes[canonical]
    od = graph.nodes[other]
    for key in set(cd) | set(od):
        if key == "entity_type":
            continue
        merged = _union_join(cd.get(key), od.get(key))
        if merged:
            cd[key] = merged


def merge_edge_into(graph, a, b, edata) -> None:
    """Thêm/gộp cạnh ``(a, b)`` với dữ liệu ``edata`` (weight cộng dồn, các trường còn lại union)."""
    if a == b:
        return  # bỏ self-loop
    if graph.has_edge(a, b):
        e = graph[a][b]
        e["weight"] = _to_float(e.get("weight")) + _to_float(edata.get("weight"))
        for key in set(e) | set(edata):
            if key == "weight":
                continue
            merged = _union_join(e.get(key), edata.get(key))
            if merged:
                e[key] = merged
    else:
        data = dict(edata)
        data["weight"] = _to_float(data.get("weight"))
        graph.add_edge(a, b, **data)


# --------------------------------------------------------------------------- #
# Vector-DB synchronization
# --------------------------------------------------------------------------- #

async def rebuild_relationships_vdb(rag, graph) -> None:
    """Xoá toàn bộ relationships_vdb rồi upsert lại từ graph đã gộp.

    Format đúng như ``minirag/operate.py::extract_entities``:
        id = md5("rel-" + src + tgt)
        content = keywords + " " + src + " " + tgt + " " + description
    """
    rel_storage = rag.relationships_vdb.client_storage
    existing_ids = [r["__id__"] for r in rel_storage["data"]]
    await rag.relationships_vdb.delete(existing_ids)

    new_data = {}
    for src, tgt, d in graph.edges(data=True):
        a, b = sorted([src, tgt])
        content = f'{d.get("keywords", "")} {a} {b} {d.get("description", "")}'
        new_data[compute_mdhash_id(a + b, prefix="rel-")] = {
            "src_id": a,
            "tgt_id": b,
            "content": content,
        }
    await rag.relationships_vdb.upsert(new_data)
    await rag.relationships_vdb.index_done_callback()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

async def main() -> None:
    args = get_args()
    outputdir = args.outputdir or (args.workingdir.rstrip("/") + "_dedup")

    if os.path.exists(outputdir):
        if not args.force:
            raise SystemExit(
                f"[ERROR] --outputdir đã tồn tại: {outputdir}\n"
                "Dùng --force để xoá và tạo lại, hoặc chọn --outputdir khác."
            )
        shutil.rmtree(outputdir)
    shutil.copytree(args.workingdir, outputdir)
    print(f"Copied baseline   : {args.workingdir} -> {outputdir}")

    # Xoá artefact QA/eval cũ của baseline để QA chạy lại sạch trên bản đã gộp.
    for name in ("qa_output.csv", "qa_eval.csv", "qa_eval.json"):
        path = os.path.join(outputdir, name)
        if os.path.exists(path):
            os.remove(path)

    rag = EndpointConfig.from_args(args).make_rag(outputdir)
    graph = rag.chunk_entity_relation_graph._graph

    n_before = graph.number_of_nodes()
    e_before = graph.number_of_edges()
    isolated_before = sum(1 for n in graph if graph.degree(n) == 0)
    if n_before == 0:
        raise SystemExit(f"[ERROR] Không tìm thấy graph trong: {args.workingdir}")

    nodes = list(graph.nodes(data=True))
    node_ids = [n for n, _ in nodes]
    norm_names = [strong_normalize(n) for n in node_ids]
    norm_types = [strong_normalize(d.get("entity_type", "")) for _, d in nodes]

    def alt_set(node_id, data) -> set:
        s = {strong_normalize(node_id)}
        alt = data.get("alternative_names", "")
        for part in split_string_by_multi_markers(str(alt), [GRAPH_FIELD_SEP]):
            sn = strong_normalize(part)
            if sn:
                s.add(sn)
        return s

    alt_sets = [alt_set(n, d) for n, d in nodes]

    # Blocking theo entity_type.
    type_indices = defaultdict(list)
    for idx, t in enumerate(norm_types):
        type_indices[t].append(idx)

    parent = list(range(len(nodes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for t, idxs in type_indices.items():
        for i in range(len(idxs)):
            gi = idxs[i]
            for j in range(i + 1, len(idxs)):
                gj = idxs[j]
                # cùng tên chuẩn hoá, hoặc tên của node này nằm trong tập bí danh của node kia
                if norm_names[gi] in alt_sets[gj] or norm_names[gj] in alt_sets[gi]:
                    union(gi, gj)

    # Nhóm cluster >= 2 node.
    clusters = defaultdict(list)
    for idx in range(len(nodes)):
        clusters[find(idx)].append(idx)
    merge_clusters = [idxs for idxs in clusters.values() if len(idxs) >= 2]

    # Gộp từng cluster (canonical = degree lớn nhất -> description dài nhất -> tên nhỏ nhất).
    removed = []
    for idxs in merge_clusters:
        order = sorted(
            idxs,
            key=lambda i: (
                -graph.degree(node_ids[i]),
                -len(str(graph.nodes[node_ids[i]].get("description", ""))),
                node_ids[i],
            ),
        )
        canon = node_ids[order[0]]
        for i in order[1:]:
            other = node_ids[i]
            merge_node_data(graph, canon, other)
            for nb in list(graph.neighbors(other)):
                if nb == canon:
                    continue
                edata = dict(graph[other][nb])
                merge_edge_into(graph, canon, nb, edata)
            graph.remove_node(other)
            removed.append(other)

    # Dọn self-loop còn sót (cạnh nội cluster).
    for n in list(graph.nodes):
        if graph.has_edge(n, n):
            graph.remove_edge(n, n)

    n_after = graph.number_of_nodes()
    e_after = graph.number_of_edges()
    isolated_after = sum(1 for n in graph if graph.degree(n) == 0)

    print("=" * 72)
    print("FALCON-KG pre-merge (node duplication) summary")
    print("=" * 72)
    print(f"Duplicate clusters merged : {len(merge_clusters)}")
    print(f"Nodes removed             : {len(removed)}")
    print(f"Nodes  : {n_before} -> {n_after}  ({n_before - n_after} removed)")
    print(f"Edges  : {e_before} -> {e_after}  ({e_before - e_after} removed)")
    print(f"Isolated nodes            : {isolated_before} -> {isolated_after}")
    print(f"Output dir                : {outputdir}")

    # Đồng bộ vector DB (bắt buộc để QA mode='mini' đọc đúng).
    if removed:
        await rag.entity_name_vdb.delete(
            [compute_mdhash_id(n, prefix="Ename-") for n in removed]
        )
        await rag.entities_vdb.delete(
            [compute_mdhash_id(n, prefix="ent-") for n in removed]
        )
        await rag.entity_name_vdb.index_done_callback()
        await rag.entities_vdb.index_done_callback()
    await rebuild_relationships_vdb(rag, graph)

    # Ghi graph đã gộp xuống GraphML.
    await rag.chunk_entity_relation_graph.index_done_callback()
    print("Done. Graph + vector DBs đã được ghi lại vào outputdir.")


if __name__ == "__main__":
    asyncio.run(main())
