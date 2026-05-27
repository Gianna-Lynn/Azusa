"""
Public search facade that keeps the external API stable while delegating
implementation details to smaller retrieval modules.
"""

from __future__ import annotations

from .config import EMBEDDING_OUTPUT_PATH
from .loaders import load_embedding_records
from .schemas import EmbeddingRecord
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
    Run exact dense retrieval over all vectors in Python memory.
    """
    score_id_list = []
    for record_id, vector in search_vecs:
        score = dot_product(query_vec, vector)
        score_id_list.append((score, record_id))

    score_id_list.sort(key=lambda item: (-item[0], item[1]))
    return [record_id for _, record_id in score_id_list[:top_k]]


def ts030_search_sparse(
    query: str,
    embedding_records: list[EmbeddingRecord],
    top_k: int = 3,
) -> list[int]:
    """
    Compatibility wrapper around the sparse BM25 retrieval helper.
    """
    return search_sparse(query, embedding_records, top_k)


def ts030_search_hybrid_with_sources(
    query: str,
    embedding_records: list[EmbeddingRecord],
    top_k: int,
) -> tuple[list[int], list[dict]]:
    """
    Compatibility wrapper for the hybrid retrieval helper.
    """
    return search_hybrid_with_sources(query, embedding_records, top_k)


def ts030_search_hybrid_rerank_with_sources(
    query: str,
    embedding_records: list[EmbeddingRecord],
    top_k: int,
) -> tuple[list[int], list[dict]]:
    """
    Compatibility wrapper for the hybrid+rereank retrieval helper.
    """
    return search_hybrid_rerank_with_sources(query, embedding_records, top_k)


def search(
    query: str,
    embedding_records: list[EmbeddingRecord],
    top_k: int,
    mode: str = "brute",
) -> list[EmbeddingRecord]:
    """
    Complete retrieval flow that chooses one backend by mode.
    """
    query_vec = embed_query(query=query)
    search_vecs = build_search_vecs(embedding_records=embedding_records)

    if mode == "brute":
        top_k_list = search_core_brute(query_vec=query_vec, search_vecs=search_vecs, top_k=top_k)
    elif mode == "hnsw":
        ensure_hnsw_index_loaded(embedding_records)
        top_k_list = search_core_hnsw(query_vec=query_vec, top_k=top_k)
    elif mode == "hybrid":
        top_k_list, _ = ts030_search_hybrid_with_sources(query, embedding_records, top_k)
    elif mode == "hybrid_rerank":
        top_k_list, _ = ts030_search_hybrid_rerank_with_sources(query, embedding_records, top_k)
    elif mode == "bm25":
        top_k_list = ts030_search_sparse(query, embedding_records, top_k)
    else:
        raise ValueError("Search mode must be bm25, brute, hnsw, hybrid,or hybrid_rerank")

    return ids_to_records(top_k_list, embedding_records)


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
        print(result.text)


if __name__ == "__main__":
    main()
