from __future__ import annotations

import json
from pathlib import Path

from azusa_rag.knowledge_base import (
    build_stable_chunks,
    content_digest,
    load_document_records,
    sync_knowledge_base,
)
from azusa_rag.schemas import DocumentRecord


def make_document(text: str, document_id: str = "policy-fixed") -> DocumentRecord:
    digest = content_digest(text)
    return DocumentRecord(
        document_id=document_id,
        version=digest[:16],
        source="policy.md",
        title="Policy",
        text=text,
        content_hash=digest,
        media_type="text/markdown",
        metadata={"department": "public", "visibility": "public"},
    )


def fake_embed(texts: list[str]) -> list[list[float]]:
    return [[float(len(text)), 1.0] for text in texts]


def test_multiformat_loader_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("plain text", encoding="utf-8")
    (tmp_path / "guide.md").write_text("# Guide\nmarkdown", encoding="utf-8")
    (tmp_path / "page.html").write_text(
        "<html><style>hidden</style><body><h1>Title</h1><p>Visible</p></body></html>",
        encoding="utf-8",
    )
    (tmp_path / "azusa_metadata.json").write_text(
        json.dumps({"guide.md": {"title": "Handbook", "department": "engineering", "visibility": "department"}}),
        encoding="utf-8",
    )

    first = load_document_records(tmp_path)
    second = load_document_records(tmp_path)

    assert [item.document_id for item in first] == [item.document_id for item in second]
    assert {item.media_type for item in first} == {"text/plain", "text/markdown", "text/html"}
    html_record = next(item for item in first if item.media_type == "text/html")
    assert "Visible" in html_record.text
    assert "hidden" not in html_record.text
    guide = next(item for item in first if item.source == "guide.md")
    assert guide.title == "Handbook"
    assert guide.metadata["department"] == "engineering"


def test_chunk_identity_survives_unrelated_prefix_change() -> None:
    original = make_document("A" * 60 + "。" + "B" * 60)
    changed = make_document("X" * 60 + "。" + "B" * 60)

    original_chunks = build_stable_chunks(original, chunk_size=61, overlap_size=0)
    changed_chunks = build_stable_chunks(changed, chunk_size=61, overlap_size=0)

    original_b = next(chunk for chunk in original_chunks if chunk.text == "B" * 60)
    changed_b = next(chunk for chunk in changed_chunks if chunk.text == "B" * 60)
    assert original_b.chunk_id == changed_b.chunk_id
    assert original_b.start_char == 61


def test_sync_reuses_unchanged_embeddings_and_removes_old_chunks(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    chunks = tmp_path / "chunks.jsonl"
    embeddings = tmp_path / "embeddings.jsonl"
    calls: list[list[str]] = []

    def tracked_embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return fake_embed(texts)

    first = sync_knowledge_base(
        [make_document("A" * 60 + "。" + "B" * 60)],
        chunk_size=61,
        overlap_size=0,
        manifest_path=manifest,
        chunks_path=chunks,
        embeddings_path=embeddings,
        embed_texts=tracked_embed,
    )
    second = sync_knowledge_base(
        [make_document("X" * 60 + "。" + "B" * 60)],
        chunk_size=61,
        overlap_size=0,
        manifest_path=manifest,
        chunks_path=chunks,
        embeddings_path=embeddings,
        embed_texts=tracked_embed,
    )

    assert first["embedded_chunk_count"] == 2
    assert second["embedded_chunk_count"] == 1
    assert second["reused_embedding_count"] == 1
    assert len(second["changes"]["removed_chunk_ids"]) == 1
    assert len(calls[1]) == 1
    stored = [json.loads(line) for line in embeddings.read_text(encoding="utf-8").splitlines()]
    assert len(stored) == 2
    assert all(item["document_version"] != "legacy" for item in stored)


def test_sync_empty_collection_deletes_all_prior_chunks(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    chunks = tmp_path / "chunks.jsonl"
    embeddings = tmp_path / "embeddings.jsonl"
    sync_knowledge_base(
        [make_document("retire this document")],
        chunk_size=50,
        overlap_size=0,
        manifest_path=manifest,
        chunks_path=chunks,
        embeddings_path=embeddings,
        embed_texts=fake_embed,
    )

    result = sync_knowledge_base(
        [],
        chunk_size=50,
        overlap_size=0,
        manifest_path=manifest,
        chunks_path=chunks,
        embeddings_path=embeddings,
        embed_texts=fake_embed,
    )

    assert result["chunk_count"] == 0
    assert result["changes"]["removed_document_ids"] == ["policy-fixed"]
    assert embeddings.read_text(encoding="utf-8") == ""
