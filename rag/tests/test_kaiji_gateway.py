from __future__ import annotations

import http.client
import json
import threading

import pytest

from azusa_rag.kaiji_gateway import (
    GatewayError,
    KaijiGatewayService,
    build_http_server,
    validate_search_request,
)
from azusa_rag.schemas import EmbeddingRecord


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


def test_service_maps_search_records_to_stable_hits(service) -> None:
    result = service.search_azusa("Kaiji", 2, "bm25")

    assert result["mode"] == "bm25"
    assert result["hits"][0]["chunk_id"] == "kaiji-001"
    assert result["hits"][0]["metadata"]["retrieval_source"] == "fixture-bm25"


def test_service_get_chunk_and_structured_not_found(service) -> None:
    assert service.get_chunk_by_id("azusa-001")["text"] == "Azusa owns retrieval."

    with pytest.raises(GatewayError) as missing:
        service.get_chunk_by_id("missing")
    assert missing.value.status == 404
    assert missing.value.code == "chunk_not_found"


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
