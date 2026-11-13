"""验证 HNSW 客户端的数据身份、候选预算及证据 ID 边界，不加载模型。"""

from dataclasses import replace

import pytest

from azusa_rag import search_hnsw as client
from azusa_rag.schemas import EmbeddingRecord


def records():
    """返回可区分顺序、正文和向量的两条证据。"""
    return [
        EmbeddingRecord("doc", "a", 0, "fixture.txt", [1.0, 0.0], None, "first"),
        EmbeddingRecord("doc", "b", 1, "fixture.txt", [0.0, 1.0], None, "second"),
    ]


def test_empty_server_loads_one_atomic_snapshot_and_reuses_identity(monkeypatch):
    """首次只有一次批量装载；确认身份后复用，不进行逐条插入。"""
    health = {"record_count": 0, "dataset_fingerprint": None}
    requests = []
    monkeypatch.setattr(client, "get_json", lambda url: health.copy())

    def load(url, body):
        """模拟一次原子装载，并让后续 health 返回本次确认的身份。"""
        requests.append((url, body))
        health.update(record_count=2, dataset_fingerprint=body["dataset_fingerprint"])
        return health.copy()

    monkeypatch.setattr(client, "post_json", load)
    fingerprint = client.ensure_hnsw_index_loaded(records())
    assert client.ensure_hnsw_index_loaded(records()) == fingerprint
    assert len(requests) == 1
    assert requests[0][0].endswith("/index/load")
    assert [record["id"] for record in requests[0][1]["records"]] == [0, 1]

    original = records()
    changed = [
        list(reversed(original)),
        [replace(original[0], text="different"), original[1]],
        [replace(original[0], embedding=[0.5, 0.5]), original[1]],
        [replace(original[0], metadata={"tenant": "other"}), original[1]],
        [replace(original[0], chunk_id="new-id"), original[1]],
        original[:1],
    ]
    for dataset in changed:
        with pytest.raises(ValueError, match="identity mismatch"):
            client.ensure_hnsw_index_loaded(dataset)
    assert len(requests) == 1


@pytest.mark.parametrize("health", [
    {"record_count": 2},
    {"record_count": 2, "dataset_fingerprint": None},
    {"record_count": True, "dataset_fingerprint": None},
    {"record_count": -1, "dataset_fingerprint": None},
    {"record_count": 0, "dataset_fingerprint": "a" * 64},
])
def test_legacy_partial_or_unbound_index_is_not_silently_adopted(monkeypatch, health):
    """旧服务、手动插入和不匹配快照必须在写入前被拒绝。"""
    monkeypatch.setattr(client, "get_json", lambda url: health)
    monkeypatch.setattr(client, "post_json", lambda *args: pytest.fail("must not write"))
    with pytest.raises(ValueError):
        client.ensure_hnsw_index_loaded(records())


def test_load_response_must_confirm_snapshot_identity(monkeypatch):
    """装载响应错误时不向调用者交付可用于检索的身份。"""
    monkeypatch.setattr(client, "get_json", lambda url: {"record_count": 0, "dataset_fingerprint": None})
    monkeypatch.setattr(client, "post_json", lambda *args: {"record_count": 2, "dataset_fingerprint": "wrong"})
    with pytest.raises(ValueError, match="does not match"):
        client.ensure_hnsw_index_loaded(records())


@pytest.mark.parametrize("top_k", [1, 16, 20, 50])
def test_search_budget_and_snapshot_identity_are_transmitted(monkeypatch, top_k):
    """候选预算至少覆盖 k，且把前一步确认的身份放进同一次搜索请求。"""
    requests = []

    def search(url, body):
        """记录实际请求体，返回一条合法命中以检查客户端协议。"""
        requests.append(body)
        return {"results": [{"id": 0, "score": 1.0}]}

    monkeypatch.setattr(client, "post_json", search)
    assert client.search_core_hnsw([1.0, 0.0], top_k, dataset_fingerprint="a" * 64) == [0]
    assert requests[0]["ef_search"] >= top_k
    assert requests[0]["dataset_fingerprint"] == "a" * 64


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_budget_fails_before_network(monkeypatch, top_k):
    """非法预算必须在发送请求前失败，布尔值不能冒充整数。"""
    monkeypatch.setattr(client, "post_json", lambda *args: pytest.fail("must not send"))
    with pytest.raises(ValueError):
        client.search_core_hnsw([1.0], top_k)


@pytest.mark.parametrize("hits", [
    [{"id": -1, "score": 1.0}],
    [{"id": True, "score": 1.0}],
    [{"id": 0, "score": 1.0}, {"id": 0, "score": 1.0}],
    [{"id": 0, "score": float("nan")}],
])
def test_invalid_result_cannot_become_evidence(monkeypatch, hits):
    """负 ID、布尔 ID、重复命中和非有限分数不得流入证据映射。"""
    monkeypatch.setattr(client, "post_json", lambda *args: {"results": hits})
    with pytest.raises(ValueError):
        client.search_core_hnsw([1.0], 2)
