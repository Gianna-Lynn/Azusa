"""
Hybrid retrieval orchestration built from dense, sparse, and rerank pieces.
"""

from __future__ import annotations

from .schemas import EmbeddingRecord
from .search_common import embed_query
from .search_hnsw import ensure_hnsw_index_loaded, search_core_hnsw
from .search_sparse import rerank_candidates, search_sparse


def hybrid_merge(dense_ids: list[int], sparse_ids: list[int], top_k: int) -> list[int]:
    """
    Keep dense order first, then append unseen sparse candidates.
    """
    merged_ids = []
    seen_ids = set()

    for dense_id in dense_ids:
        if dense_id in seen_ids:
            continue
        merged_ids.append(dense_id)
        seen_ids.add(dense_id)

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
    Annotate whether each merged result came from dense, sparse, or both.
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
    Run dense HNSW recall plus sparse BM25 recall, then merge their candidates.
    """
    query_vec = embed_query(query)
    ensure_hnsw_index_loaded(embedding_records)
    dense_ids = search_core_hnsw(query_vec, top_k)
    sparse_ids = search_sparse(query, embedding_records, top_k)
    merged_ids = hybrid_merge(dense_ids, sparse_ids, top_k)
    source_map = build_source_map(merged_ids, dense_ids, sparse_ids)
    return merged_ids, source_map


def search_hybrid_rerank_with_sources(
    query: str,
    embedding_records: list[EmbeddingRecord],
    top_k: int,
) -> tuple[list[int], list[dict]]:
    """
    Expand the hybrid candidate pool, then re-rank it before truncating to top-k.
    """
    candidate_k = min(len(embedding_records), max(20, top_k * 5))

    query_vec = embed_query(query)
    ensure_hnsw_index_loaded(embedding_records)
    dense_ids = search_core_hnsw(query_vec, candidate_k)
    sparse_ids = search_sparse(query, embedding_records, candidate_k)
    merged_ids = hybrid_merge(dense_ids, sparse_ids, candidate_k)
    reranked_ids = rerank_candidates(
        query=query,
        candidate_ids=merged_ids,
        embedding_records=embedding_records,
        top_k=top_k,
    )
    source_map = build_source_map(reranked_ids, dense_ids, sparse_ids)
    return reranked_ids, source_map
