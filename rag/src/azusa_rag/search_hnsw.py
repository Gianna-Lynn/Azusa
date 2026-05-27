"""
Dense retrieval helpers that talk to the Rust HNSW server.
"""

from __future__ import annotations

import json
import urllib.request

from .config import HEALTH_URL, INSERT_URL, SEARCH_URL
from .schemas import EmbeddingRecord


def read_json_response(response) -> dict:
    """
    Parse a JSON HTTP response into a Python dict.
    """
    response_body: bytes = response.read()
    response_json = response_body.decode("utf-8")
    return json.loads(response_json)


def get_json(url: str) -> dict:
    """
    Issue an HTTP GET request and parse the JSON body.
    """
    with urllib.request.urlopen(url) as response:
        return read_json_response(response)


def post_json(url: str, request_dict: dict) -> dict:
    """
    Issue an HTTP POST request with a JSON body and parse the JSON response.
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
        return read_json_response(response)


def post_json_no_response(url: str, request_dict: dict) -> None:
    """
    Issue an HTTP POST request when only the status code matters.
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
    Convert one embedding record into the Rust server's insert payload.
    """
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
    Make sure the in-memory Rust index matches the current embedding dataset.
    """
    response_data = get_json(HEALTH_URL)
    record_count = response_data["record_count"]

    if record_count == len(embedding_records):
        return
    if record_count != 0:
        raise ValueError(
            "Azusa server already contains a different in-memory index. "
            "Please restart azusa_server before loading a new embedding dataset."
        )

    for index, embedding_record in enumerate(embedding_records):
        insert_dict = build_insert_record(embedding_record=embedding_record, record_id=index)
        post_json_no_response(INSERT_URL, insert_dict)


def search_core_hnsw(
    query_vec: list[float],
    top_k: int = 3,
) -> list[int]:
    """
    Search the Rust HNSW service and return top-k record ids.
    """
    request_dict = {
        "vector": query_vec,
        "k": top_k,
        "ef_search": 16,
    }
    response_data = post_json(SEARCH_URL, request_dict)
    return [hit["id"] for hit in response_data["results"]]
