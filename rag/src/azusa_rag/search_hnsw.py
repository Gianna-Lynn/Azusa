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
    读取 HTTP 响应体，并解析成 Python 字典。
    """
    response_body: bytes = response.read()
    response_json = response_body.decode("utf-8")
    # 这里返回的通常就是 azusa_server 那边序列化出来的 JSON 对象。
    return json.loads(response_json)


def get_json(url: str) -> dict:
    """
    对给定 URL 发 GET 请求，最多等待 30 秒，解析响应体, 返回解析出来的 JSON 字典。
    HTTP 错误和网络异常向上传播，由搜索门面或网关分类处理。
    """
    # 直接传入 URL 且未提供请求体时，urlopen 默认发送 GET 请求。
    with urllib.request.urlopen(url, timeout=30) as response:
        return read_json_response(response)


def post_json(url: str, request_dict: dict) -> dict:
    """
    对给定 URL 发 JSON POST 请求，解析响应体, 返回解析出来的 JSON 字典。
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



def build_insert_record(
    embedding_record: EmbeddingRecord,
    record_index: int,
) -> dict:
    """
    把 EmbeddingRecord 变成 azusa_server 装载/插入所需的字典。
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
    # 注意别把这里的三个"id"搞混：
    # - 顶层 "id"：Rust 索引内的记录 ID，就是装载时的插入下标（record_index），
    #   之后 /search 返回的 result["id"] 也是它，拿回来当 embedding_records 的位置用；
    # - "doc_id"：源文档的稳定身份；
    # - "chunk_id"：块的稳定身份。
    # 后两者是数据身份，和"下标"完全是两回事。
    return {
        "id": record_index,
        "vector": embedding_record.embedding,
        "metadata": {
            "doc_id": embedding_record.doc_id,
            "chunk_id": embedding_record.chunk_id,
            "source": embedding_record.source,
            "page": embedding_record.page,
        },
    }


def ensure_hnsw_index_loaded(embedding_records: list[EmbeddingRecord]) -> str:
    """确保 Rust 服务里装载当前这批 embedding 记录，并返回数据指纹。

    函数会在服务为空时通过 POST /index/load 发送完整快照；
    如果服务已装载相同快照则直接复用，不同快照则拒绝覆盖。

    指纹 = 对整批记录的文本、metadata、向量和顺序做 sha256 得到的"身份照"，
    比"条数相等"更可靠：服务端按插入下标当记录 ID，换了数据或重排列表后，
    旧下标会指向错误的记录。

    根据服务当前状态走三条路：
    - 服务为空：把整批快照原子装载进 /index/load，复核后返回指纹；
    - 指纹和条数都一致：服务已装好这批数据，直接复用，返回指纹；
    - 服务非空且指纹不一致：报错要求重启服务，不覆盖旧索引。

    注意：指纹是本地客户端声明的数据身份，不是服务端认证，也不证明来源真实性。
    """
    # ---- 第一步：给当前这批记录算一个"数据指纹" ----
    # 把每条 EmbeddingRecord 转成 dict，之后整批序列化再哈希。
    # 排序字典键但保留记录列表顺序：下标就是 Rust ID，重排同样算新快照。
    snapshot = [record.to_dict() for record in embedding_records]
    encoded = json.dumps(
        snapshot,
        ensure_ascii=False,          # 中文等字符原样输出，不做 \uXXXX 转义
        sort_keys=True,              # 只排序每条记录内部的字段顺序，不改变记录之间的顺序
        separators=(",", ":"),       # 去掉空格，让序列化结果字节级确定，哈希才不会抖动
        allow_nan=False,             # NaN/Infinity 不是合法 JSON，混进指纹前先报错
    ).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()  # 64 位十六进制字符串指纹

    # ---- 第二步：问服务"你现在装着什么数据" ----
    response = get_json(HEALTH_URL)
    count = response.get("record_count")
    # 老版本服务没有 dataset_fingerprint 字段 / count 不是整数或为负，
    # 说明它不认识快照身份协议，必须先重启到新版服务再做下面的复用判断。
    if type(count) is not int or count < 0 or "dataset_fingerprint" not in response:
        raise ValueError("Azusa server lacks snapshot identity support; restart the updated server.")

    # ---- 第三步：指纹和条数都一致 -> 服务里装的就是这批数据，直接复用 ----
    if response["dataset_fingerprint"] == fingerprint and count == len(embedding_records):
        return fingerprint

    # ---- 第四步：服务非空但指纹不一致 -> 拒绝覆盖 ----
    # 服务里的旧数据不是我们要的：既不能静默复用（ID 会错配），也不能覆盖，
    # 唯一出路是让使用者重启服务、清空旧索引后再装载。
    if count != 0 or response["dataset_fingerprint"] is not None:
        raise ValueError(
            "Azusa dataset identity mismatch. Please restart azusa_server "
            "before loading a new embedding dataset."
        )

    # ---- 第五步：服务是空的 -> 原子装载整批快照 ----
    # 一个请求携带完整批次，服务端验证后发布，避免逐条插入中断留下半个索引。
    loaded = post_json(
        SEARCH_URL.rsplit("/", 1)[0] + "/index/load",  # 去掉 URL 末尾的 "/search"，拼出 ".../index/load"
        {
            "dataset_fingerprint": fingerprint,
            "records": [
                build_insert_record(record, index)
                for index, record in enumerate(embedding_records)  # enumerate 下标就是 Rust 侧的记录 ID
            ],
        },
    )

    # ---- 第六步：装载后复核响应 ----
    # 服务端宣称的指纹和条数必须和刚发送的一致，防止装了一半或装错数据。
    if (
        loaded.get("dataset_fingerprint") != fingerprint
        or type(loaded.get("record_count")) is not int
        or loaded["record_count"] != len(embedding_records)
    ):
        raise ValueError("Azusa loaded snapshot identity or record count does not match.")
    return fingerprint  # 后续 /search 会带上这个指纹，服务端据此拒绝错配查询


def search_core_hnsw(
    query_vec: list[float],
    top_k: int = 3,
    *,
    dataset_fingerprint: str | None = None,
) -> list[int]:
    """
    把 query 向量发给 azusa_server 的 /search，返回校验过的 top-k 个记录下标。

    这里"下标"指记录在调用方 embedding_records 列表里的位置：服务端按装载
    顺序给记录编号，装载时的下标就是它的 ID，所以拿返回值直接
    embedding_records[index] 就能取回对应记录。

    top_k 必须是真正的正整数，否则直接报错。请求体包含向量、k、ef_search：
    ef_search 是 HNSW 搜索时的探索宽度，取 max(16, top_k) 保证至少召回 k 个结果。
    如果传了 dataset_fingerprint，服务端会先核对自身数据是不是这一批，
    防止拿旧索引答新查询。

    服务端负责校验向量维度、数值合法性和快照身份；客户端侧再拒绝负数/布尔/
    重复下标和非有限分数，避免脏结果混进下游的证据映射。
    """
    # ---- 第一步：先验参数 ----
    # top_k 必须是真正的不带小数的正整数。
    # 用 type(...) is int 而不是 isinstance，是为了把 True/False 这类 bool 也挡掉
    # （bool 是 int 的子类，isinstance 会放它进来，而 JSON 里的 true 解码后就是 bool）。
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    request_dict = {
        "vector": query_vec,
        "k": top_k,
        "ef_search": max(16, top_k),  # 搜索时的探索宽度：ef_search 不够大就召回不满 k 个，保底 16
    }
    if dataset_fingerprint is not None:
        # 带上指纹，服务端会先核对自身数据是不是这一批，防止拿旧索引答新查询。
        request_dict["dataset_fingerprint"] = dataset_fingerprint

    # ---- 第二步：发 POST /search，拿到原始结果 ----
    response_data = post_json(SEARCH_URL, request_dict)
    # SearchResponse 的 JSON 形状大致是：
    # {"results": [{"id": 3, "score": 0.982}, ...]}
    # 注意：服务端的 "id" 字段就是装载时的插入下标（见 ensure_hnsw_index_loaded），
    # 拿回来可以直接当 embedding_records 的列表位置用。
    hits = response_data["results"]
    indices = [hit["id"] for hit in hits]

    # ---- 第三步：校验返回的下标 ----
    # 三种脏数据都不收：结果数超过请求的 k、下标是负数/布尔、同一下标重复。
    # 服务端正常不会这样返回，客户端再挡一层，避免脏结果混进下游的证据映射。
    if (
        len(indices) > top_k
        or any(type(record_index) is not int or record_index < 0 for record_index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError("Azusa returned invalid or duplicate record indices")

    # ---- 第四步：校验分数 ----
    # 分数必须是真正的数字且有限；NaN/Inf 混进来会把下游的排序和打分搞乱。
    if any(
        type(hit["score"]) not in (int, float) or not math.isfinite(hit["score"])
        for hit in hits
    ):
        raise ValueError("Azusa returned a non-finite score")
    return indices
