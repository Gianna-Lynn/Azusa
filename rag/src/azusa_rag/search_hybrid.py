"""
hybrid / hybrid_rerank 的编排逻辑。

这里不再关心 HTTP 细节或 BM25 细节，只做：
1. dense 和 sparse 的结果怎么合并
2. hybrid_rerank 怎么扩大候选池、再做 rerank
"""

from __future__ import annotations

from .schemas import EmbeddingRecord
from .search_common import embed_query
from .search_hnsw import ensure_hnsw_index_loaded, search_core_hnsw
from .search_sparse import rerank_candidates, search_sparse


def hybrid_merge(dense_ids: list[int], sparse_ids: list[int], top_k: int) -> list[int]:
    """
    最小 hybrid merge。
    先保留 dense 的顺序，再补进 sparse 里还没出现过的候选。
    """
    merged_ids = []
    seen_ids = set()

    for dense_id in dense_ids:
        if dense_id in seen_ids:
            continue
        merged_ids.append(dense_id)
        seen_ids.add(dense_id)

    # dense 结果先保留原顺序，sparse 只负责补充 dense 没召回到的候选。
    for sparse_id in sparse_ids:
        if sparse_id in seen_ids:
            continue
        merged_ids.append(sparse_id)
        seen_ids.add(sparse_id)

    return merged_ids[:top_k]


def build_source_map(
    merged_ids: list[int],
    dense_ids: list[int],
    sparse_ids: list[int],
) -> list[dict]:
    """
    为 hybrid 结果补充来源标记，方便观察每条结果来自 dense、sparse 还是两者都有。
    """
    dense_ids_set = set(dense_ids)
    sparse_ids_set = set(sparse_ids)
    source_items = []

    for merged_id in merged_ids:
        if merged_id in dense_ids_set and merged_id in sparse_ids_set:
            source = "dense + sparse"
        elif merged_id in dense_ids_set:
            source = "dense"
        else:
            source = "sparse"
        source_items.append({"id": merged_id, "source": source})

    return source_items


def search_hybrid_with_sources(
    query: str,
    embedding_records: list[EmbeddingRecord],
    top_k: int,
) -> tuple[list[int], list[dict]]:
    """
    hybrid 专用 helper。
    先分别跑 dense 检索和 sparse 检索，再合并结果，最后返回 id 列表和来源标记。
    """
    query_vec = embed_query(query)  # 把 query 做成向量
    ensure_hnsw_index_loaded(embedding_records)
    dense_ids = search_core_hnsw(query_vec, top_k)  # 用 dense 检索拿一批结果
    sparse_ids = search_sparse(query, embedding_records, top_k)  # 用 sparse 检索再拿一批结果
    merged_ids = hybrid_merge(dense_ids, sparse_ids, top_k)  # 把 dense 和 sparse 结果合并去重
    source_map = build_source_map(merged_ids, dense_ids, sparse_ids)  # 给每个结果补来源标签
    return merged_ids, source_map


def search_hybrid_rerank_with_sources(
    query: str,
    embedding_records: list[EmbeddingRecord],
    top_k: int,
) -> tuple[list[int], list[dict]]:
    """
    最小 hybrid rerank。
    先扩大 dense + sparse 的候选召回，再用 reranker 对候选集做二次重排。
    """
    # 候选集故意取得更宽一点，不然某些本来相关、但被 merge 顺序压后的 chunk
    # 可能还没来得及参与 rerank，就已经被截断掉了。
    candidate_k = min(len(embedding_records), max(20, top_k * 5))

    query_vec = embed_query(query)
    ensure_hnsw_index_loaded(embedding_records)
    dense_ids = search_core_hnsw(query_vec, candidate_k)
    sparse_ids = search_sparse(query, embedding_records, candidate_k)
    merged_ids = hybrid_merge(dense_ids, sparse_ids, candidate_k)
    # 先拿到一个“更宽”的候选集，再交给 reranker 做排序纠偏。
    reranked_ids = rerank_candidates(
        query=query,
        candidate_ids=merged_ids,
        embedding_records=embedding_records,
        top_k=top_k,
    )
    source_map = build_source_map(reranked_ids, dense_ids, sparse_ids)
    return reranked_ids, source_map
