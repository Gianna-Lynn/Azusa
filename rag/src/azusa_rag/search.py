"""
这个文件现在是检索层的“门面”。

前面把原来那个很长的 search.py 拆成了多个子模块：
1. search_common.py：公共 helper
2. search_hnsw.py：和 Rust azusa_server 交互
3. search_sparse.py：BM25 sparse + rerank
4. search_hybrid.py：hybrid / hybrid_rerank 编排

但对外部调用者来说，仍然优先 import 这个文件，避免 run_search.py / run_eval.py
以及其他调用点都跟着一起改。
"""

from __future__ import annotations

from time import perf_counter

from .config import EMBEDDING_OUTPUT_PATH
from .loaders import load_embedding_records
from .schemas import EmbeddingRecord
from .filters import AccessContext, MetadataFilter, record_is_visible
from .search_common import (
    build_search_vecs,
    dot_product,
    embed_query,
    get_query,
    ids_to_records,
)
from .search_hnsw import ensure_hnsw_index_loaded, search_core_hnsw
from .search_hybrid import search_hybrid_rerank_with_sources, search_hybrid_with_sources
from .search_sparse import search_sparse


def search_core_brute(
    query_vec: list[float],
    search_vecs: list[tuple[int, list[float]]],
    top_k: int = 3,
) -> list[int]:
    """
    暴力搜索内核。
    接受 query 向量和全部候选向量，返回最优的 top-k 个 id。
    """
    score_id_list = []
    for record_id, vector in search_vecs:
        score = dot_product(query_vec, vector)
        score_id_list.append((score, record_id))

    # sort 包含 tie-break 逻辑：分数相同的时候让 id 小的在前面，避免结果抖动。
    score_id_list.sort(key=lambda item: (-item[0], item[1]))
    return [record_id for _, record_id in score_id_list[:top_k]]


def search_records_with_sources(
    query: str,
    embedding_records: list[EmbeddingRecord],
    top_k: int,
    mode: str = "brute",
    metadata_filter: MetadataFilter | None = None,
    access_context: AccessContext | None = None,
    filter_overfetch: int = 5,
    trace: dict | None = None,
) -> tuple[list[EmbeddingRecord], list[dict] | None]:
    """
    完整的搜索流程。
    按 mode 选择 brute / hnsw / hybrid / hybrid_rerank / bm25，
    最终返回 EmbeddingRecord 列表，以及可选的来源标记。
    """
    total_started = perf_counter()
    if trace is not None:
        trace.update({"mode": mode, "requested_top_k": top_k, "corpus_count": len(embedding_records)})
    filtered_request = metadata_filter is not None or access_context is not None
    requested_top_k = top_k
    if filtered_request:
        if filter_overfetch < 1:
            raise ValueError("filter_overfetch must be at least 1")
        top_k = min(len(embedding_records), max(top_k, top_k * filter_overfetch))

    if mode == "brute":
        # brute 模式才需要 query vector 和全部候选向量。
        started = perf_counter()
        query_vec = embed_query(query=query)  # query -> query vector
        if trace is not None:
            trace["embedding_ms"] = (perf_counter() - started) * 1000
        search_vecs = build_search_vecs(embedding_records=embedding_records)  # embedding_records -> search vectors
        started = perf_counter()
        top_k_list = search_core_brute(query_vec=query_vec, search_vecs=search_vecs, top_k=top_k)
        if trace is not None:
            trace["dense_ms"] = (perf_counter() - started) * 1000
            trace["dense_candidate_count"] = len(top_k_list)
        source_map = None
    elif mode == "hnsw":
        # hnsw 需要先确保 azusa_server 里已经装好当前这批 embedding。
        started = perf_counter()
        query_vec = embed_query(query=query)
        if trace is not None:
            trace["embedding_ms"] = (perf_counter() - started) * 1000
        started = perf_counter()
        ensure_hnsw_index_loaded(embedding_records)
        top_k_list = search_core_hnsw(query_vec=query_vec, top_k=top_k)
        if trace is not None:
            trace["dense_ms"] = (perf_counter() - started) * 1000
            trace["dense_candidate_count"] = len(top_k_list)
        source_map = None
    elif mode == "hybrid":
        top_k_list, source_map = search_hybrid_with_sources(query, embedding_records, top_k, trace=trace)
    elif mode == "hybrid_rerank":
        # hybrid_rerank 在 hybrid 召回的基础上再走一次 rerank 重排。
        top_k_list, source_map = search_hybrid_rerank_with_sources(query, embedding_records, top_k, trace=trace)
    elif mode == "bm25":
        started = perf_counter()
        top_k_list = search_sparse(query, embedding_records, top_k)
        if trace is not None:
            trace["sparse_ms"] = (perf_counter() - started) * 1000
            trace["sparse_candidate_count"] = len(top_k_list)
        source_map = [{"id": record_id, "source": "bm25 sparse"} for record_id in top_k_list]
    else:
        raise ValueError("Search mode must be bm25, brute, hnsw, hybrid,or hybrid_rerank")

    result_records = ids_to_records(top_k_list, embedding_records)
    if filtered_request:
        if trace is not None:
            trace["pre_filter_count"] = len(result_records)
        kept_records = []
        kept_sources = [] if source_map is not None else None
        for index, record in enumerate(result_records):
            if record_is_visible(record, metadata_filter, access_context):
                kept_records.append(record)
                if kept_sources is not None:
                    kept_sources.append(source_map[index])
            if len(kept_records) == requested_top_k:
                break
        result_records = kept_records
        source_map = kept_sources
    result_records = result_records[:requested_top_k]
    if trace is not None:
        trace["post_filter_count"] = len(result_records)
        trace["filter_strategy"] = "overfetch-then-filter" if filtered_request else "none"
        trace["total_ms"] = (perf_counter() - total_started) * 1000
    return result_records, source_map


def search(
    query: str,
    embedding_records: list[EmbeddingRecord],
    top_k: int,
    mode: str = "brute",
) -> list[EmbeddingRecord]:
    """
    对外保留的最小门面。
    如果调用方只关心最终命中的记录，而不关心每条记录来自 dense / sparse 哪一路，
    就继续用这个函数。
    """
    result_records, _ = search_records_with_sources(
        query=query,
        embedding_records=embedding_records,
        top_k=top_k,
        mode=mode,
    )
    return result_records


def main() -> None:
    query_str = get_query()
    embedding_records_list = load_embedding_records(EMBEDDING_OUTPUT_PATH)
    result_list = search(
        query=query_str,
        embedding_records=embedding_records_list,
        top_k=3,
        mode="brute",
    )

    for result in result_list:
        print(result.text)  # dataclass 不是字典，需要用 . 访问，而不是 [] 访问


if __name__ == "__main__":
    main()
