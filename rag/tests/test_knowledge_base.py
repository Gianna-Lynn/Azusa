from __future__ import annotations

import json
import hashlib
from pathlib import Path

import azusa_rag.knowledge_base as knowledge_base
from azusa_rag.knowledge_base import (
    ChangeSet,
    build_stable_chunks,
    content_digest,
    load_document_records,
    plan_changes,
    stable_document_id,
    sync_knowledge_base,
)
from azusa_rag.schemas import ChunkRecord, DocumentRecord


def make_document(text: str, document_id: str = "policy-fixed") -> DocumentRecord:
    # L2 窄测试不能被尚未完成的 L1a 遮挡，因此 fixture 自己生成已知合法哈希。
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
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


def stub_chunks(document: DocumentRecord, *, chunk_size: int, overlap_size: int) -> list[ChunkRecord]:
    """为 L2c 隔离已由 L2a 负责的切块实现。"""

    pieces = document.text.split("|")
    return [
        ChunkRecord(
            doc_id=document.document_id,
            chunk_id=f"{document.document_id}:{hashlib.sha256(piece.encode()).hexdigest()[:20]}:0",
            chunk_index=index,
            source=document.source,
            page=None,
            text=piece,
            document_version=document.version,
            content_hash=hashlib.sha256(piece.encode()).hexdigest(),
        )
        for index, piece in enumerate(pieces)
    ]


def stub_plan(
    documents: list[DocumentRecord],
    chunks: list[ChunkRecord],
    previous_manifest: dict | None,
) -> ChangeSet:
    """为 L2c 隔离已由 L2b 负责的 ChangeSet 算法。"""

    previous = {item["chunk_id"]: item for item in (previous_manifest or {}).get("chunks", [])}
    current = {item.chunk_id: item for item in chunks}
    reused = sorted(
        chunk_id for chunk_id, chunk in current.items()
        if previous.get(chunk_id, {}).get("content_hash") == chunk.content_hash
    )
    return ChangeSet(
        added_chunk_ids=sorted(set(current) - set(reused)),
        reused_chunk_ids=reused,
        removed_chunk_ids=sorted(set(previous) - set(current)),
        added_document_ids=[],
        updated_document_ids=[],
        removed_document_ids=["policy-fixed"] if not documents and previous else [],
    )


def test_content_digest_normalizes_newlines_and_whitespace() -> None:
    assert content_digest("  alpha\r\nbeta  ") == content_digest("alpha\nbeta")
    digest = content_digest("alpha")
    assert len(digest) == 64
    assert digest == digest.lower()


def test_stable_document_id_is_readable_and_path_sensitive() -> None:
    first = stable_document_id("Policies/Employee Handbook.md")
    second = stable_document_id("Archive/Employee Handbook.md")
    assert first.startswith("employee-handbook-")
    assert first != second
    assert first == stable_document_id("policies\\EMPLOYEE HANDBOOK.md")


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


def test_plan_changes_requires_matching_id_and_hash() -> None:
    document = make_document("current")
    chunk = ChunkRecord(
        doc_id=document.document_id,
        chunk_id="stable-chunk",
        chunk_index=0,
        source=document.source,
        page=None,
        text="current",
        content_hash="new-hash",
    )
    previous = {
        "documents": [{"document_id": document.document_id, "content_hash": "old-doc"}],
        "chunks": [
            {"chunk_id": "stable-chunk", "content_hash": "old-hash"},
            {"chunk_id": "removed", "content_hash": "removed-hash"},
        ],
    }

    changes = plan_changes([document], [chunk], previous)

    assert changes.reused_chunk_ids == []
    assert changes.added_chunk_ids == ["stable-chunk"]
    assert changes.removed_chunk_ids == ["removed"]
    assert changes.updated_document_ids == [document.document_id]


def test_sync_reuses_unchanged_embeddings_and_removes_old_chunks(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.json"
    chunks = tmp_path / "chunks.jsonl"
    embeddings = tmp_path / "embeddings.jsonl"
    calls: list[list[str]] = []
    monkeypatch.setattr(knowledge_base, "build_stable_chunks", stub_chunks)
    monkeypatch.setattr(knowledge_base, "plan_changes", stub_plan)

    def tracked_embed(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return fake_embed(texts)

    first = sync_knowledge_base(
        [make_document("changed-A|stable-B")],
        chunk_size=61,
        overlap_size=0,
        manifest_path=manifest,
        chunks_path=chunks,
        embeddings_path=embeddings,
        embed_texts=tracked_embed,
    )
    second = sync_knowledge_base(
        [make_document("changed-X|stable-B")],
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


def test_sync_empty_collection_deletes_all_prior_chunks(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.json"
    chunks = tmp_path / "chunks.jsonl"
    embeddings = tmp_path / "embeddings.jsonl"
    monkeypatch.setattr(knowledge_base, "build_stable_chunks", stub_chunks)
    monkeypatch.setattr(knowledge_base, "plan_changes", stub_plan)
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
