"""验证 Kaiji 语义网关的参数、字段、路由、错误与可复现 BM25 闭环。"""

from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

import pytest

from azusa_rag.kaiji_gateway import (
    GatewayError,
    KaijiGatewayService,
    build_http_server,
    validate_search_request,
)
from azusa_rag.loaders import load_embedding_records
from azusa_rag.schemas import EmbeddingRecord


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "kaiji_gateway_embedding.jsonl"


def make_record(chunk_id: str, text: str) -> EmbeddingRecord:
    return EmbeddingRecord(
        doc_id="doc",
        chunk_id=chunk_id,
        chunk_index=0,
        source="fixture.txt",
        embedding=[1.0, 0.0],
        page=None,
        text=text,
    )


def fake_search(query, embedding_records, top_k, mode):
    matches = [
        record
        for record in embedding_records
        if query.lower() in record.text.lower()
    ][:top_k]
    sources = [{"id": index, "source": f"fixture-{mode}"} for index, _ in enumerate(matches)]
    return matches, sources


@pytest.fixture
def service() -> KaijiGatewayService:
    return KaijiGatewayService(
        [
            make_record("kaiji-001", "Kaiji owns the Agent runtime."),
            make_record("azusa-001", "Azusa owns retrieval."),
            make_record("folder/chunk 1", "Encoded chunk id evidence."),
        ],
        search_function=fake_search,
    )


def test_validate_search_request_rejects_bad_boundaries() -> None:
    assert validate_search_request(" Kaiji ", 3, "bm25") == ("Kaiji", 3, "bm25")

    with pytest.raises(GatewayError) as invalid_k:
        validate_search_request("Kaiji", 0, "bm25")
    assert invalid_k.value.status == 400
    assert not invalid_k.value.retryable

    with pytest.raises(GatewayError, match="mode must be one of"):
        validate_search_request("Kaiji", 3, "unknown")


@pytest.mark.parametrize(
    ("query", "k", "mode", "code"),
    [
        ("", 3, "bm25", "invalid_query"),
        ("   ", 3, "bm25", "invalid_query"),
        (None, 3, "bm25", "invalid_query"),
        ("Kaiji", True, "bm25", "invalid_k"),
        ("Kaiji", 21, "bm25", "invalid_k"),
    ],
)
def test_validate_search_request_rejects_adversarial_values(
    query,
    k,
    mode,
    code: str,
) -> None:
    """空查询、布尔数量和越界数量都必须作为不可重试请求错误拒绝。"""

    with pytest.raises(GatewayError) as caught:
        validate_search_request(query, k, mode)

    assert caught.value.status == 400
    assert caught.value.code == code
    assert not caught.value.retryable


def test_service_maps_search_records_to_stable_hits(service) -> None:
    result = service.search_azusa("Kaiji", 2, "bm25")

    assert result == {
        "mode": "bm25",
        "hits": [
            {
                "chunk_id": "kaiji-001",
                "text": "Kaiji owns the Agent runtime.",
                "score": None,
                "source": "fixture.txt",
                "page": None,
                "metadata": {
                    "doc_id": "doc",
                    "chunk_index": 0,
                    "retrieval_source": "fixture-bm25",
                },
            }
        ],
    }


def test_service_get_chunk_and_structured_not_found(service) -> None:
    assert service.get_chunk_by_id("azusa-001") == {
        "chunk_id": "azusa-001",
        "text": "Azusa owns retrieval.",
        "source": "fixture.txt",
        "page": None,
        "metadata": {"doc_id": "doc", "chunk_index": 0},
    }

    with pytest.raises(GatewayError) as missing:
        service.get_chunk_by_id("missing")
    assert missing.value.status == 404
    assert missing.value.code == "chunk_not_found"


@pytest.mark.parametrize("chunk_id", ["", "   ", None, 7])
def test_service_rejects_invalid_chunk_id(service, chunk_id) -> None:
    """非法 chunk_id 是请求错误，不得伪装成普通未命中。"""

    with pytest.raises(GatewayError) as caught:
        service.get_chunk_by_id(chunk_id)

    assert caught.value.status == 400
    assert caught.value.code == "invalid_chunk_id"
    assert not caught.value.retryable


def test_tracked_fixture_supports_reproducible_bm25_contract() -> None:
    """仓库内小型 fixture 必须独立支撑 TODO12 的 stromules 正向闭环。"""

    fixture_service = KaijiGatewayService(load_embedding_records(FIXTURE_PATH))

    search = fixture_service.search_azusa("stromules", 3, "bm25")
    first_hit = search["hits"][0]
    chunk = fixture_service.get_chunk_by_id(first_hit["chunk_id"])

    assert "1962" in first_hit["text"]
    assert chunk["text"] == first_hit["text"]


def test_backend_failure_is_retryable(service) -> None:
    def broken_search(*args):
        raise RuntimeError("temporary backend outage")

    broken = KaijiGatewayService(
        [make_record("c1", "text")],
        search_function=broken_search,
    )

    with pytest.raises(GatewayError) as failure:
        broken.search_azusa("text", 1, "bm25")
    assert failure.value.status == 503
    assert failure.value.retryable


def test_real_http_search_chunk_and_error_contract(service) -> None:
    server = build_http_server("127.0.0.1", 0, service=service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    connection = http.client.HTTPConnection(host, port, timeout=2)
    try:
        body = json.dumps({"query": "Kaiji", "k": 1, "mode": "bm25"})
        connection.request(
            "POST",
            "/search_azusa",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        search_payload = json.loads(response.read())
        assert response.status == 200
        assert search_payload["hits"][0]["chunk_id"] == "kaiji-001"

        connection.request("GET", "/chunks/kaiji-001")
        response = connection.getresponse()
        chunk_payload = json.loads(response.read())
        assert response.status == 200
        assert chunk_payload["text"] == "Kaiji owns the Agent runtime."

        connection.request("GET", "/chunks/folder%2Fchunk%201")
        response = connection.getresponse()
        encoded_chunk_payload = json.loads(response.read())
        assert response.status == 200
        assert encoded_chunk_payload["chunk_id"] == "folder/chunk 1"

        connection.request("GET", "/chunks/missing")
        response = connection.getresponse()
        missing_payload = json.loads(response.read())
        assert response.status == 404
        assert missing_payload["error"]["code"] == "chunk_not_found"
        assert not missing_payload["error"]["retryable"]

        connection.request("GET", "/unknown")
        response = connection.getresponse()
        unknown_get = json.loads(response.read())
        assert response.status == 404
        assert unknown_get["error"]["code"] == "route_not_found"

        connection.request(
            "POST",
            "/unknown",
            body="{}",
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        unknown_post = json.loads(response.read())
        assert response.status == 404
        assert unknown_post["error"]["code"] == "route_not_found"

        bad_body = json.dumps({"query": "Kaiji", "k": 0, "mode": "bm25"})
        connection.request(
            "POST",
            "/search_azusa",
            body=bad_body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        error_payload = json.loads(response.read())
        assert response.status == 400
        assert error_payload["error"]["code"] == "invalid_k"
        assert not error_payload["error"]["retryable"]
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
