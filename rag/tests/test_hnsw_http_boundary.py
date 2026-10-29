"""在临时端口验证 Python、Rust 和语义网关闭环，不访问模型或用户运行中的服务。"""

from dataclasses import replace
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
import threading
from urllib.error import HTTPError

import pytest

from azusa_rag import search_hnsw as client
from azusa_rag.kaiji_gateway import KaijiGatewayService
from azusa_rag.schemas import EmbeddingRecord


@pytest.fixture
def rust_server(monkeypatch):
    """启动本仓库构建的服务并在结束时回收；端口零交给操作系统分配。"""
    executable = Path(__file__).resolve().parents[2] / "target" / "debug" / (
        "azusa_server.exe" if os.name == "nt" else "azusa_server"
    )
    if not executable.is_file():
        pytest.skip("先从仓库根目录执行 cargo build --offline --bin azusa_server")
    process = subprocess.Popen(
        [str(executable)],
        env={**os.environ, "AZUSA_BIND_ADDR": "127.0.0.1:0"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    line_queue = Queue()
    reader = threading.Thread(target=lambda: line_queue.put(process.stdout.readline()), daemon=True)
    reader.start()
    try:
        try:
            line = line_queue.get(timeout=10)
        except Empty:
            pytest.fail("临时 Rust 服务未按时报告监听地址")
        assert line.startswith("Azusa listening on "), line
        base_url = "http://" + line.split(" on ", 1)[1].strip()
        monkeypatch.setattr(client, "HEALTH_URL", base_url + "/health")
        monkeypatch.setattr(client, "SEARCH_URL", base_url + "/search")
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        reader.join(timeout=1)
        process.stdout.close()
        process.stderr.close()


def make_records():
    """使用人工预设向量，只测证据映射，不声称模型检索质量。"""
    return [
        EmbeddingRecord("doc", "a", 0, "fixture.txt", [1.0, 0.0], None, "first evidence"),
        EmbeddingRecord("doc", "b", 1, "fixture.txt", [0.0, 1.0], None, "second evidence"),
    ]


def test_real_snapshot_search_gateway_and_same_size_conflict(rust_server):
    """真实 HTTP 搜索返回无来源信息时，网关保留命中并能按稳定 ID 回读。"""
    records = make_records()

    def backend(query, records, k, mode):
        """省略模型编码，只用固定向量走真实装载和查询，来源标记刻意为 None。"""
        fingerprint = client.ensure_hnsw_index_loaded(records)
        ids = client.search_core_hnsw([1.0, 0.0], k, dataset_fingerprint=fingerprint)
        return [records[index] for index in ids], None

    gateway = KaijiGatewayService(records, search_function=backend)
    for _ in range(2):
        hits = gateway.search_azusa("first", 1, "hnsw")["hits"]
        assert hits[0]["chunk_id"] == "a"
        assert hits[0]["text"] == gateway.get_chunk_by_id("a")["text"]
        assert hits[0]["metadata"]["retrieval_source"] is None
        assert client.get_json(rust_server + "/record/0")["metadata"]["chunk_id"] == "a"
    for altered in [list(reversed(records)), [replace(records[0], text="other"), records[1]]]:
        with pytest.raises(ValueError, match="identity mismatch"):
            client.ensure_hnsw_index_loaded(altered)
    assert gateway.search_azusa("first", 1, "hnsw")["hits"][0]["text"] == "first evidence"


@pytest.mark.parametrize("mode", ["hnsw", "hybrid", "hybrid_rerank"])
def test_real_retrieval_facade_passes_identity_in_each_dense_path(rust_server, monkeypatch, mode):
    """走生产检索门面，只有编码与模型重排使用替身，HTTP 和索引保持真实。"""
    from azusa_rag import search as facade
    from azusa_rag import search_hybrid as hybrid

    monkeypatch.setattr(facade, "embed_query", lambda *args, **kwargs: [1.0, 0.0])
    monkeypatch.setattr(hybrid, "embed_query", lambda *args, **kwargs: [1.0, 0.0])
    monkeypatch.setattr(hybrid, "rerank_candidates", lambda **kwargs: kwargs["candidate_ids"][:kwargs["top_k"]])
    gateway = KaijiGatewayService(make_records())
    result = gateway.search_azusa("first", 1, mode)
    assert result["mode"] == mode
    assert result["hits"][0]["chunk_id"] == "a"
    assert result["hits"][0]["text"] == gateway.get_chunk_by_id("a")["text"]


def test_real_invalid_requests_and_wrong_query_identity_preserve_snapshot(rust_server):
    """错维度和身份的真实 HTTP 请求失败后，健康检查及合法搜索仍成功。"""
    fingerprint = client.ensure_hnsw_index_loaded(make_records())
    for body in [
        {"vector": [], "k": 1, "dataset_fingerprint": fingerprint},
        {"vector": [1.0], "k": 1, "dataset_fingerprint": fingerprint},
        {"vector": [1.0, 0.0], "k": 0, "dataset_fingerprint": fingerprint},
        {"vector": [1.0, 0.0], "k": 1, "ef_search": 0, "dataset_fingerprint": fingerprint},
        {"vector": [1.0, 0.0], "k": 1, "dataset_fingerprint": "b" * 64},
    ]:
        with pytest.raises(HTTPError) as caught:
            client.post_json(rust_server + "/search", body)
        assert caught.value.code in (400, 409)
        caught.value.close()
        assert client.get_json(rust_server + "/health")["record_count"] == 2
        assert client.search_core_hnsw([1.0, 0.0], 1, dataset_fingerprint=fingerprint) == [0]


def test_real_batch_failure_does_not_publish_partial_index(rust_server):
    """批次后段非法不能留下前段已插入记录，随后可以正常装载。"""
    with pytest.raises(HTTPError) as caught:
        client.post_json(rust_server + "/index/load", {
            "dataset_fingerprint": "a" * 64,
            "records": [{"id": 0, "vector": [1.0, 0.0]}, {"id": 1, "vector": [1.0]}],
        })
    assert caught.value.code == 400
    caught.value.close()
    health = client.get_json(rust_server + "/health")
    assert health["record_count"] == 0
    assert health["dataset_fingerprint"] is None
    fingerprint = client.ensure_hnsw_index_loaded(make_records())
    assert client.search_core_hnsw([1.0, 0.0], 1, dataset_fingerprint=fingerprint) == [0]
