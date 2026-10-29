"""
和 Rust azusa_server 交互的 dense / HNSW 相关 helper。

这里主要负责两件事：
1. 封装和服务端的 HTTP 通信
2. 确保 server 里已经装入当前这批 embedding 数据
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.request

from .config import HEALTH_URL, SEARCH_URL
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
    对给定 URL 发 GET 请求，最多等待 30 秒，返回解析后的 JSON 字典。
    HTTP 错误和网络异常向上传播，由搜索门面或网关分类处理。
    """
    with urllib.request.urlopen(url, timeout=30) as response:
        return read_json_response(response)


def post_json(url: str, request_dict: dict) -> dict:
    """
    对给定 URL 发 JSON POST 请求，返回解析后的 JSON 字典。
    发送前拒绝 NaN/Infinity；HTTP 与网络错误不在这里隐式重试。
    """
    request_json = json.dumps(request_dict, ensure_ascii=False, allow_nan=False)
    request_body = request_json.encode("utf-8")                 # 编码成 UTF-8 bytes
    request = urllib.request.Request(
        url=url,
        data=request_body,
        headers={"Content-Type": "application/json"},           # 设置 Content-Type: application/json
        method="POST",                                          # 发送 POST
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return read_json_response(response)                     # 解析 JSON


def post_json_no_response(url: str, request_dict: dict) -> None:
    """
    对给定 URL 发 JSON POST 请求，但不读取业务返回值。
    用于 /insert 这类只关心状态码的接口。
    """
    request_json = json.dumps(request_dict, ensure_ascii=False, allow_nan=False)
    request_body = request_json.encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
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


def ensure_hnsw_index_loaded(embedding_records: list[EmbeddingRecord]) -> str:
    """校验完整有序快照身份，仅向空服务原子装载；冲突要求重启，不覆盖旧索引。

    指纹覆盖正文、metadata、向量及顺序，避免等数量语料和列表重排造成 ID 错配。
    它是可信本地客户端声明的数据身份，不是服务端认证或来源真实性证明。
    """
    # 排序字典键但保留记录列表顺序：下标就是 Rust ID，重排同样算新快照。
    snapshot = [record.to_dict() for record in embedding_records]
    encoded = json.dumps(
        snapshot,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()
    response = get_json(HEALTH_URL)
    count = response.get("record_count")
    if type(count) is not int or count < 0 or "dataset_fingerprint" not in response:
        raise ValueError("Azusa server lacks snapshot identity support; restart the updated server.")
    if response["dataset_fingerprint"] == fingerprint and count == len(embedding_records):
        return fingerprint
    if count != 0 or response["dataset_fingerprint"] is not None:
        raise ValueError(
            "Azusa dataset identity mismatch. Please restart azusa_server "
            "before loading a new embedding dataset."
        )
    # 一个请求携带完整批次，服务端验证后发布，避免逐条插入中断留下半个索引。
    loaded = post_json(
        SEARCH_URL.rsplit("/", 1)[0] + "/index/load",
        {
            "dataset_fingerprint": fingerprint,
            "records": [
                build_insert_record(record, index)
                for index, record in enumerate(embedding_records)
            ],
        },
    )
    if (
        loaded.get("dataset_fingerprint") != fingerprint
        or type(loaded.get("record_count")) is not int
        or loaded["record_count"] != len(embedding_records)
    ):
        raise ValueError("Azusa loaded snapshot identity or record count does not match.")
    return fingerprint


def search_core_hnsw(
    query_vec: list[float],
    top_k: int = 3,
    *,
    dataset_fingerprint: str | None = None,
) -> list[int]:
    """
    将查询向量和预期快照身份送到服务端，返回已校验的 top-k 数字 ID。

    top_k 必须是真正的正整数；ef 至少覆盖 top_k。服务端负责维度、数值和身份
    校验，客户端再拒绝负数/布尔/重复 ID 和非有限分数，避免污染证据映射。
    """
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    request_dict = {
        "vector": query_vec,
        "k": top_k,
        "ef_search": max(16, top_k),
    }
    if dataset_fingerprint is not None:
        request_dict["dataset_fingerprint"] = dataset_fingerprint
    response_data = post_json(SEARCH_URL, request_dict)
    # SearchResponse 的 JSON 形状大致是：
    # {"results": [{"id": 3, "score": 0.982}, ...]}
    hits = response_data["results"]
    ids = [hit["id"] for hit in hits]
    if (
        len(ids) > top_k
        or any(type(record_id) is not int or record_id < 0 for record_id in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("Azusa returned invalid or duplicate record IDs")
    if any(
        type(hit["score"]) not in (int, float) or not math.isfinite(hit["score"])
        for hit in hits
    ):
        raise ValueError("Azusa returned a non-finite score")
    return ids
