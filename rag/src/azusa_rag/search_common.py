"""
Shared helpers used by retrieval backends and the public search facade.
"""

from __future__ import annotations

from .config import QUERY_DIR
from .embedder import embed_text_list
from .loaders import load_documents
from .schemas import EmbeddingRecord


def get_query() -> str:
    """
    Return the first query text file under ``data/queries``.
    """
    query_docs = load_documents(QUERY_DIR)
    return query_docs[0]["text"]


def embed_query(query: str) -> list[float]:
    """
    Encode a single query into one dense vector.
    """
    return embed_text_list([query])[0]


def dot_product(vec1: list[float], vec2: list[float]) -> float:
    """
    Compute the dot product between two vectors of the same length.
    """
    if len(vec1) != len(vec2):
        raise ValueError("vec1 and vec2 have different length, cannot caculate dot product")

    return sum(elem1 * elem2 for elem1, elem2 in zip(vec1, vec2))


def build_search_vecs(
    embedding_records: list[EmbeddingRecord],
) -> list[tuple[int, list[float]]]:
    """
    Map embedding records into ``(record_id, vector)`` pairs.
    """
    id_vector_list = []
    for index, embedding_record in enumerate(embedding_records):
        id_vector_list.append((index, embedding_record.embedding))
    return id_vector_list


def ids_to_records(
    record_ids: list[int],
    embedding_records: list[EmbeddingRecord],
) -> list[EmbeddingRecord]:
    """
    Resolve retrieved ids back into ``EmbeddingRecord`` objects.
    """
    result_records = []
    for record_id in record_ids:
        result_records.append(embedding_records[record_id])
    return result_records
