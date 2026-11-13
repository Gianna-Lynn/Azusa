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
    用 Reciprocal Rank Fusion（倒数排名融合）合并多条已排序的记录下标列表。

    假设 M 代表所有的排行榜, m 代表第 m 个排行榜,
    rank_m(d) 代表文档 d 在排行榜 m 中的排名(从1开始)
    文档 d 的 RRF 分数定义为：
    Score(d) = ∑_{m∈M} 1/(k + rank_m(d))

    k 默认 60，来自 TREC 等 IR 实验的常用设定。
    k 越大，高 rank 和低 rank 的贡献差距越小（曲线更平）；第 1 名是 1/61 = 0.0164，第 10 名是 1/70 =  0.0142。
    k 越小，高 rank 的权重越突出。 第 1 名的得分是 1/(1+1) = 0.5，第 10 名是 1/(1+10) = 0.09

    如果同一个记录下标出现在多路结果中，它会累加各路的贡献，
    因此被多路检索器同时认为相关的文档会自然排到前面。

    输入示例为 [1, 2], [2, 3] 或者 [1, 2, 3], [2, 3, 4], [5, 6]。
    每个列表中的位置加 1 是 rank，列表的值是记录在 embedding_records 中的下标。

    # Arguments
    - `ranked_lists`: 多路已排序的记录下标列表；
    - `top_k`: 截取融合结果的数量；
    - `k`: RRF 平滑常数，默认为 60。

    # Returns
    融合后的记录下标列表，按 RRF 分数降序、同分时按下标升序排列；没有输入列表时为空。
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

    # Arguments
    - `merged_ids`: 融合后的记录下标列表；
    - `dense_ids`: dense 检索得到的记录下标列表；
    - `sparse_ids`: sparse 检索得到的记录下标列表。

    # Returns
    与融合结果顺序一致的字典列表，每项包含记录下标 `id` 和来源 `source`。
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
    执行 hybrid 检索并标记结果来源。

    先分别跑 dense 检索和 sparse 检索，再合并结果。

    # Arguments
    - `query`: 查询文本；
    - `embedding_records`: 检索使用的记录列表；
    - `top_k`: 最终结果数量；
    - `trace`: 可选的字典，用于记录各阶段耗时和候选数量。

    # Returns
    融合后的记录下标列表和对应来源标记组成的二元组。
    """
    # 阶段 1：将查询文本编码为 dense 向量
    started = perf_counter()
    query_vec = embed_query(query)  # 把 query 做成向量
    if trace is not None:
        trace["embedding_ms"] = (perf_counter() - started) * 1000
    # 阶段 2：执行 dense 检索(HNSW)，获取候选记录dense_ids, 取 top_k 个
    # 这里的ids是 embedding_records 的下标，后续可以直接用来取出对应的文本
    started = perf_counter()
    fingerprint = ensure_hnsw_index_loaded(embedding_records)
    dense_ids = search_core_hnsw(query_vec, top_k, dataset_fingerprint=fingerprint)
    if trace is not None:
        trace["dense_ms"] = (perf_counter() - started) * 1000
        trace["dense_candidate_count"] = len(dense_ids)
    # 阶段 3：执行 sparse 检索(BM25)，获取候选记录sparse_ids, 取 top_k 个
    started = perf_counter()
    sparse_ids = search_sparse(query, embedding_records, top_k)  # 用 sparse 检索再拿一批结果
    if trace is not None:
        trace["sparse_ms"] = (perf_counter() - started) * 1000
        trace["sparse_candidate_count"] = len(sparse_ids)
    # 阶段 4：使用 RRF 融合 dense 和 sparse 的排序结果merged_ids, 取 top_k 个
    started = perf_counter()
    merged_ids = rrf_fusion(dense_ids, sparse_ids, top_k=top_k)
    if trace is not None:
        trace["fusion_ms"] = (perf_counter() - started) * 1000
        trace["fused_candidate_count"] = len(merged_ids)
    # 阶段 5：为融合结果标记检索来源并返回
    source_map = build_source_map(merged_ids, dense_ids, sparse_ids)
    return merged_ids, source_map


def search_hybrid_rerank_with_sources(
    query: str,
    embedding_records: list[EmbeddingRecord],
    top_k: int,
    trace: dict | None = None,
) -> tuple[list[int], list[dict]]:
    """
    执行最小 hybrid rerank 并标记结果来源。

    先扩大 dense + sparse 的候选召回，再用 reranker 对候选集做二次重排。

    # Arguments
    - `query`: 查询文本；
    - `embedding_records`: 检索使用的记录列表；
    - `top_k`: 最终结果数量；
    - `trace`: 可选的字典，用于记录各阶段耗时、候选数量和重排信息。

    # Returns
    重排后的记录下标列表和对应来源标记组成的二元组。
    """
    # 阶段 0：扩大候选集，给后续 rerank 留余量
    # 候选集故意取得更宽一点，不然某些本来相关、但被 merge 顺序压后的 chunk
    # 可能还没来得及参与 rerank，就已经被截断掉了。
    # top_k * 5: 候选数量设为最终结果的约5倍;
    # max(20, ...):即使 top_k 很小,也至少尝试取20个候选;
    # min(len(embedding_records),.):候选数量不能超过实际记录总数。
    candidate_k = min(len(embedding_records), max(20, top_k * 5))

    # 阶段 1：将查询文本编码为 dense 向量
    started = perf_counter()
    query_vec = embed_query(query)
    if trace is not None:
        trace["embedding_ms"] = (perf_counter() - started) * 1000
    # 阶段 2：执行 dense 检索(HNSW)，获取扩大的候选记录 dense_ids
    # 这里的 ids 是 embedding_records 的下标，后续可以直接用来取出对应的文本。
    started = perf_counter()
    fingerprint = ensure_hnsw_index_loaded(embedding_records)
    dense_ids = search_core_hnsw(query_vec, candidate_k, dataset_fingerprint=fingerprint)
    if trace is not None:
        trace["dense_ms"] = (perf_counter() - started) * 1000
        trace["dense_candidate_count"] = len(dense_ids)
    # 阶段 3：执行 sparse 检索(BM25)，获取扩大的候选记录 sparse_ids
    started = perf_counter()
    sparse_ids = search_sparse(query, embedding_records, candidate_k)
    if trace is not None:
        trace["sparse_ms"] = (perf_counter() - started) * 1000
        trace["sparse_candidate_count"] = len(sparse_ids)
    # 阶段 4：使用 RRF 融合 dense 和 sparse 的排序结果，保留更宽的候选集
    started = perf_counter()
    merged_ids = rrf_fusion(dense_ids, sparse_ids, top_k=candidate_k)
    if trace is not None:
        trace["fusion_ms"] = (perf_counter() - started) * 1000
        trace["fused_candidate_count"] = len(merged_ids)
    # 阶段 5：先拿到一个“更宽”的候选集，再交给 reranker 做二次重排。
    # 与普通 hybrid 直接返回 RRF 结果不同，这里不立即截断到 top_k，
    # 而是让 reranker 在扩大后的候选集中进一步判断查询与文档的相关性，
    # 以纠正仅依赖检索排名可能造成的顺序偏差。
    started = perf_counter()
    reranked_ids = rerank_candidates(
        query=query,
        candidate_ids=merged_ids,
        embedding_records=embedding_records,
        top_k=top_k,
        trace=trace,
    )
    if trace is not None:
        trace["rerank_ms"] = (perf_counter() - started) * 1000
        trace["rerank_candidate_count"] = len(merged_ids)
    source_map = build_source_map(reranked_ids, dense_ids, sparse_ids)
    return reranked_ids, source_map
