"""
hybrid / hybrid_rerank 的编排逻辑。

这里不再关心 HTTP 细节或 BM25 细节，只做：
1. 用 RRF 把 dense 和 sparse 的排序列表融合成一个
2. hybrid_rerank 怎么扩大候选池、再做 rerank

RRF (Reciprocal Rank Fusion) 是 hybrid fusion 的核心机制。
选择 RRF 而不是分数归一化或加权组合的原因：
- dense (cosine) 和 sparse (BM25) 的分数分布完全不同，无法直接比较；
- RRF 只看 rank，天然跨检索器可比，且不需要调参（除了平滑常数 k）。
"""

from __future__ import annotations

from time import perf_counter

from .schemas import EmbeddingRecord
from .search_common import embed_query
from .search_hnsw import ensure_hnsw_index_loaded, search_core_hnsw
from .search_sparse import rerank_candidates, search_sparse


def rrf_fusion(
    *ranked_lists: list[int],
    top_k: int,
    k: int = 60,
) -> list[int]:
    """
    Reciprocal Rank Fusion: 倒数排名融合. 把多条已排序的 id 列表融合成一个排序列表。

    假设 M 代表所有的排行榜, m 代表第 m 个排行榜,
    rank_m(d) 代表文档 d 在排行榜 m 中的排名(从1开始)
    文档 d 的 RRF 分数定义为：
    Score(d) = ∑_{m∈M} 1/(k + rank_m(d))

    k 默认 60，来自 TREC 等 IR 实验的常用设定。
    k 越大，高 rank 和低 rank 的贡献差距越小（曲线更平）；第 1 名是 1/61 = 0.0164，第 10 名是 1/70 =  0.0142。
    k 越小，高 rank 的权重越突出。 第 1 名的得分是 1/(1+1) = 0.5，第 10 名是 1/(1+10) = 0.09

    如果同一个 id 出现在多路结果中，它会累加各路的贡献，
    因此被多路检索器同时认为相关的文档会自然排到前面。

    rrf_fusion 的输入是多条已排序的 id 列表，例如  [1, 2], [2, 3] 或者 [1, 2, 3], [2, 3, 4], [5, 6]。
    在一个列表中, 列表的(下标+1)代表 rank, 列表的值代表文档 id。
    输出是融合后的 id 列表。
    """
    if not ranked_lists:
        return []

    scores: dict[int, float] = {}

    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # 按 RRF 分数降序，同分时按 id 升序保证确定性
    sorted_ids = sorted(scores.keys(), key=lambda id_: (-scores[id_], id_))
    return sorted_ids[:top_k]


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
    trace: dict | None = None,
) -> tuple[list[int], list[dict]]:
    """
    hybrid 专用 helper。
    先分别跑 dense 检索和 sparse 检索，再合并结果，最后返回 id 列表和来源标记。
    """
    started = perf_counter()
    query_vec = embed_query(query)  # 把 query 做成向量
    if trace is not None:
        trace["embedding_ms"] = (perf_counter() - started) * 1000
    started = perf_counter()
    ensure_hnsw_index_loaded(embedding_records)
    dense_ids = search_core_hnsw(query_vec, top_k)  # 用 dense 检索拿一批结果
    if trace is not None:
        trace["dense_ms"] = (perf_counter() - started) * 1000
        trace["dense_candidate_count"] = len(dense_ids)
    started = perf_counter()
    sparse_ids = search_sparse(query, embedding_records, top_k)  # 用 sparse 检索再拿一批结果
    if trace is not None:
        trace["sparse_ms"] = (perf_counter() - started) * 1000
        trace["sparse_candidate_count"] = len(sparse_ids)
    started = perf_counter()
    merged_ids = rrf_fusion(dense_ids, sparse_ids, top_k=top_k)
    if trace is not None:
        trace["fusion_ms"] = (perf_counter() - started) * 1000
        trace["fused_candidate_count"] = len(merged_ids)
    source_map = build_source_map(merged_ids, dense_ids, sparse_ids)
    return merged_ids, source_map


def search_hybrid_rerank_with_sources(
    query: str,
    embedding_records: list[EmbeddingRecord],
    top_k: int,
    trace: dict | None = None,
) -> tuple[list[int], list[dict]]:
    """
    最小 hybrid rerank。
    先扩大 dense + sparse 的候选召回，再用 reranker 对候选集做二次重排。
    """
    # 候选集故意取得更宽一点，不然某些本来相关、但被 merge 顺序压后的 chunk
    # 可能还没来得及参与 rerank，就已经被截断掉了。
    candidate_k = min(len(embedding_records), max(20, top_k * 5))

    started = perf_counter()
    query_vec = embed_query(query)
    if trace is not None:
        trace["embedding_ms"] = (perf_counter() - started) * 1000
    started = perf_counter()
    ensure_hnsw_index_loaded(embedding_records)
    dense_ids = search_core_hnsw(query_vec, candidate_k)
    if trace is not None:
        trace["dense_ms"] = (perf_counter() - started) * 1000
        trace["dense_candidate_count"] = len(dense_ids)
    started = perf_counter()
    sparse_ids = search_sparse(query, embedding_records, candidate_k)
    if trace is not None:
        trace["sparse_ms"] = (perf_counter() - started) * 1000
        trace["sparse_candidate_count"] = len(sparse_ids)
    started = perf_counter()
    merged_ids = rrf_fusion(dense_ids, sparse_ids, top_k=candidate_k)
    if trace is not None:
        trace["fusion_ms"] = (perf_counter() - started) * 1000
        trace["fused_candidate_count"] = len(merged_ids)
    # 先拿到一个”更宽”的候选集，再交给 reranker 做排序纠偏。
    started = perf_counter()
    reranked_ids = rerank_candidates(
        query=query,
        candidate_ids=merged_ids,
        embedding_records=embedding_records,
        top_k=top_k,
    )
    if trace is not None:
        trace["rerank_ms"] = (perf_counter() - started) * 1000
        trace["rerank_candidate_count"] = len(merged_ids)
    source_map = build_source_map(reranked_ids, dense_ids, sparse_ids)
    return reranked_ids, source_map
