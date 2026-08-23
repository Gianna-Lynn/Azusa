"""
和 Rust azusa_server 交互的 dense / HNSW 相关 helper。

这里主要负责两件事：
1. 封装和服务端的 HTTP 通信
2. 确保 server 里已经装入当前这批 embedding 数据
"""

from __future__ import annotations

import json
import urllib.request

from .config import HEALTH_URL, INSERT_URL, SEARCH_URL
from .schemas import EmbeddingRecord


def read_json_response(response) -> dict:
    """
    把 HTTP 响应体读出来，并解析成 Python 字典。
    """
    response_body: bytes = response.read()
    response_json = response_body.decode("utf-8")
    # 这里返回的通常就是 azusa_server 那边序列化出来的 JSON 对象。
    return json.loads(response_json)


def get_json(url: str) -> dict:
    """
    对给定 URL 发 GET 请求，返回解析后的 JSON 字典。
    """
    with urllib.request.urlopen(url) as response:
        return read_json_response(response)


def post_json(url: str, request_dict: dict) -> dict:
    """
    对给定 URL 发 JSON POST 请求，返回解析后的 JSON 字典。
    """
    request_json = json.dumps(request_dict, ensure_ascii=False) # 把字典构造成JSON
    request_body = request_json.encode("utf-8")                 # 编码成 UTF-8 bytes
    request = urllib.request.Request(
        url=url,
        data=request_body,
        headers={"Content-Type": "application/json"},           # 设置 Content-Type: application/json
        method="POST",                                          # 发送 POST
    )
    with urllib.request.urlopen(request) as response:
        return read_json_response(response)                     # 解析 JSON


def post_json_no_response(url: str, request_dict: dict) -> None:
    """
    对给定 URL 发 JSON POST 请求，但不读取业务返回值。
    用于 /insert 这类只关心状态码的接口。
    """
    request_json = json.dumps(request_dict, ensure_ascii=False)
    request_body = request_json.encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        response.status


def build_insert_record(
    embedding_record: EmbeddingRecord,
    record_id: int,
) -> dict:
    """
    把 EmbeddingRecord 变成 azusa_server /insert 所需的字典。
    """
    # 这个结构对应 Rust 服务那边的 VectorRecord：
    # {
    #   "id": 0,
    #   "vector": [...],
    #   "metadata": {
    #     "doc_id": "...",
    #     "chunk_id": "...",
    #     "source": "...",
    #     "page": null
    #   }
    # }
    return {
        "id": record_id,
        "vector": embedding_record.embedding,
        "metadata": {
            "doc_id": embedding_record.doc_id,
            "chunk_id": embedding_record.chunk_id,
            "source": embedding_record.source,
            "page": embedding_record.page,
        },
    }


def ensure_hnsw_index_loaded(embedding_records: list[EmbeddingRecord]) -> None:
    """
    确保 Rust HNSW server 里已经装入了当前这批 embedding 数据。
    """
    # 先请求 /health。
    # 按照 azusa_server 的 health_handler，返回形如：
    # {"status":"ok","record_count":0}
    response_data = get_json(HEALTH_URL)
    record_count = response_data["record_count"]

    # 最小安全策略：
    # - record_count == 0：server 还是空索引，继续走后面的逐条插入
    # - record_count == len(embedding_records)：暂时认为 server 里大概率已经是当前这批数据
    # - 其他情况：不静默复用旧索引，直接报错提醒先重启 azusa_server
    if record_count == len(embedding_records):
        return
    if record_count != 0:
        raise ValueError(
            "Azusa server already contains a different in-memory index. "
            "Please restart azusa_server before loading a new embedding dataset."
        )

    for index, embedding_record in enumerate(embedding_records):
        # record_count == 0 时，就逐条 /insert。
        # 这里仍然沿用“Python 列表下标 == server 内部 id”这套约定。
        insert_dict = build_insert_record(embedding_record=embedding_record, record_id=index)
        post_json_no_response(INSERT_URL, insert_dict)


def search_core_hnsw(
    query_vec: list[float],
    top_k: int = 3,
) -> list[int]:
    """
    HNSW 搜索内核。
    把 query vector 送到 azusa_server 的 /search，并返回 top-k id。
    """
    # 这个结构对应到 azusa_server.rs 里的 SearchRequest。
    request_dict = {
        "vector": query_vec,
        "k": top_k,
        "ef_search": 16,
    }
    response_data = post_json(SEARCH_URL, request_dict)
    # SearchResponse 的 JSON 形状大致是：
    # {"results": [{"id": 3, "score": 0.982}, ...]}
    return [hit["id"] for hit in response_data["results"]]
