"""
检索层对外的统一门面。

单次检索的完整流程（mode 分派、来源标记、trace）都从
``search_records_with_sources`` 进入；CLI 入口（run_search.py / run_eval.py）、
Agent（agent.py）和 Kaiji gateway（kaiji_gateway.py）统一通过本文件调用，
不直接依赖下面的具体后端模块。

支持的模式以 ``modes.SEARCH_MODES`` 为准：
brute（本地暴力 dense）、hnsw（Rust HNSW 服务）、bm25（sparse）、
hybrid（dense + sparse 经 RRF 融合）、hybrid_rerank（hybrid 候选 + rerank）。

后端细节由子模块承担：
- search_common.py：query 向量化、内积、下标转记录等公共 helper
- search_hnsw.py：和 Rust azusa_server 交互
- search_sparse.py：BM25 sparse 检索与 rerank 调度
- search_hybrid.py：hybrid / hybrid_rerank 编排
"""

from __future__ import annotations

from time import perf_counter

from .config import EMBEDDING_OUTPUT_PATH
from .loaders import load_embedding_records
from .schemas import EmbeddingRecord
from .search_common import (
    build_search_vecs,
    dot_product,
    embed_query,
    get_query,
    indices_to_records,
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
    接受 query 向量和全部候选向量，返回得分最高的 top-k 个下标。
    """
    score_index_list = []
    for record_index, vector in search_vecs:
        score = dot_product(query_vec, vector)
        score_index_list.append((score, record_index))

    # sort 包含 tie-break 逻辑：分数相同的时候让下标小的在前面，避免结果抖动。
    score_index_list.sort(key=lambda item: (-item[0], item[1]))
    return [record_index for _, record_index in score_index_list[:top_k]]


def search_records_with_sources(
    query: str,
    embedding_records: list[EmbeddingRecord],
    top_k: int,
    mode: str = "brute",
    trace: dict | None = None,
) -> tuple[list[EmbeddingRecord], list[dict] | None]:
    """
    单条 query 的完整检索流程。

    按 mode 选择检索后端：

    - ``brute``：本地暴力 dense，作为 baseline
    - ``hnsw``：经 HTTP 调用 Rust HNSW 服务
    - ``bm25``：sparse 关键词检索
    - ``hybrid``：dense + sparse 分别召回后经 RRF 融合
    - ``hybrid_rerank``：hybrid 扩大候选后再用 reranker 重排

    # Arguments
    - `query`: 查询文本；
    - `embedding_records`: 全部可检索记录；
    - `top_k`: 期望返回的结果数量；
    - `mode`: 检索模式，合法值见 ``modes.SEARCH_MODES``；
    - `trace`: 可选字典，用于记录实际后端、各阶段耗时与候选数量。

    # Returns
    ``(result_records, source_map)``：
    截断后的 EmbeddingRecord 列表，以及对应的来源标记；
    brute / hnsw 模式下 source_map 为 ``None``，其余模式为
    ``{"id": 记录下标, "source": 后端说明}`` 字典的列表。
    """
    total_started = perf_counter()
    if trace is not None:
        trace.update({"mode": mode, "requested_top_k": top_k, "corpus_count": len(embedding_records)})
    requested_top_k = top_k



    # 下面根据 mode 选择具体的检索后端，并统一得到候选记录下标和来源信息.
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
        # 1. query 编码为 query vector
        query_vec = embed_query(query=query)
        
        if trace is not None:
            trace["embedding_ms"] = (perf_counter() - started) * 1000
        
        started = perf_counter()
        # 2. 确保 azusa_server 已经加载了当前 embedding_records 的 HNSW 索引，并返回索引指纹.
        fingerprint = ensure_hnsw_index_loaded(embedding_records)
        # 3. 调用 azusa_server 的 HNSW 搜索接口，返回 top_k_list.
        top_k_list = search_core_hnsw(query_vec=query_vec, top_k=top_k, dataset_fingerprint=fingerprint)
        
        if trace is not None:
            trace["dense_ms"] = (perf_counter() - started) * 1000
            trace["dense_candidate_count"] = len(top_k_list)
        
        # 4. hnsw 模式下，source_map 目前没有来源标记，返回 None.
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
        source_map = [{"id": record_index, "source": "bm25 sparse"} for record_index in top_k_list]
        # 搜索数据流研究:
        # top_k_list = [1, 3] , 
        # source_map = [ {"id": 1, "source": "bm25 sparse"}, {"id": 3, "source": "bm25 sparse"}, ]
    else:
        raise ValueError("Search mode must be bm25, brute, hnsw, hybrid,or hybrid_rerank")

    result_records = indices_to_records(top_k_list, embedding_records) # 搜索数据流研究 result_records = [ record_B,  record_D, ]
    result_records = result_records[:requested_top_k]
    if trace is not None:
        trace["post_filter_count"] = len(result_records)
        trace["filter_strategy"] = "none"
        trace["total_ms"] = (perf_counter() - total_started) * 1000
    return result_records, source_map


def search(
    query: str,
    embedding_records: list[EmbeddingRecord],
    top_k: int,
    mode: str = "brute",
    *,
    trace: dict | None = None,
) -> list[EmbeddingRecord]:
    """
    对外保留的最小门面。
    如果调用方只关心最终命中的记录，而不关心每条记录来自 dense / sparse 哪一路，
    就继续用这个函数。
    评测可传入独立 trace 字典，保存实际后端与耗时，不改变返回值结构。
    """
    result_records, _ = search_records_with_sources(
        query=query,
        embedding_records=embedding_records,
        top_k=top_k,
        mode=mode,
        trace=trace,
    )
    return result_records


def main() -> None:
    """
    历史遗留的最简演示入口：用默认 query 跑一次 brute 检索并打印命中文本。

    正式入口是 ``python run.py search``（经 run_search.py）；这里只保留给
    直接 ``python -m azusa_rag.search`` 快速验证的场景。
    """
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
