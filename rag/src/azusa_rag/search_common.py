"""
给检索各个子模块共用的基础 helper。

这里放的是几类“谁都可能用到，但不属于某个具体检索后端”的东西：
1. 读默认 query
2. query 做 embedding
3. 向量点积
4. EmbeddingRecord <-> id / vector 的映射
"""

from __future__ import annotations

from .config import QUERY_DIR
from .embedder import embed_text_list
from .loaders import load_documents
from .schemas import EmbeddingRecord


def get_query() -> str:
    """
    最简化的得到 query 字符串的函数。
    当前仅取 data/queries 目录里的第一条文本。
    """
    query_docs = load_documents(QUERY_DIR)
    return query_docs[0]["text"]  # str


def embed_query(query: str) -> list[float]:
    """
    用 embed_text_list 方法得到 query vector。
    """
    # embed_text_list 返回 list[list[float]]，
    # 但这里我们只传一条 query，所以取第一个元素即可。
    return embed_text_list([query])[0]


def dot_product(vec1: list[float], vec2: list[float]) -> float:
    """
    两个向量的点积。最好提前将两个向量归一化。
    """
    if len(vec1) != len(vec2):
        raise ValueError("vec1 and vec2 have different length, cannot caculate dot product")

    return sum(elem1 * elem2 for elem1, elem2 in zip(vec1, vec2))


def build_search_vecs(
    embedding_records: list[EmbeddingRecord],
) -> list[tuple[int, list[float]]]:
    """
    把 EmbeddingRecord 列表变成 `(id, vector)` 列表。
    这样后面 brute / hnsw 这类检索内核就能直接吃。
    """
    id_vector_list = []
    for index, embedding_record in enumerate(embedding_records):
        # 这里的 index 就是后面整个检索流程里复用的 record id。
        id_vector_list.append((index, embedding_record.embedding))
    return id_vector_list


def ids_to_records(
    record_ids: list[int],
    embedding_records: list[EmbeddingRecord],
) -> list[EmbeddingRecord]:
    """
    用检索返回的 id 回查原始 EmbeddingRecord。
    """
    result_records = []
    for record_id in record_ids:
        result_records.append(embedding_records[record_id])
    return result_records
