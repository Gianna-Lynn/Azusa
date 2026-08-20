"""Azusa Knowledge Base v1: deterministic ingestion and incremental artifacts.

The v1 boundary is intentionally explicit: source and embedding artifacts are durable,
while the Rust HNSW service remains a rebuildable in-memory acceleration layer.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable

from .chunker import chunk_text_with_spans
from .schemas import ChunkRecord, DocumentRecord, EmbeddingRecord

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf"}
SOURCE_METADATA_FILE = "azusa_metadata.json"


def content_digest(value: str) -> str:
    """Return the canonical SHA-256 hex digest for normalized UTF-8 text."""

    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def stable_document_id(relative_source: str) -> str:
    """Create a readable, collision-resistant identity from the source path."""

    normalized = relative_source.replace("\\", "/").strip("/").lower()
    readable = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", Path(normalized).stem).strip("-")
    readable = readable or "document"
    suffix = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:10]
    return f"{readable}-{suffix}"


class _TextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.ignored_depth += 1
        elif tag in {"p", "br", "div", "section", "article", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.ignored_depth:
            self.ignored_depth -= 1
        elif tag in {"p", "div", "section", "article", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        lines = [re.sub(r"\s+", " ", line).strip() for line in "".join(self.parts).splitlines()]
        return "\n".join(line for line in lines if line)


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - environment-specific dependency gate
        raise RuntimeError("PDF ingestion requires the optional 'pypdf' dependency") from exc

    pages = []
    for page_number, page in enumerate(PdfReader(str(path)).pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            pages.append(f"[page {page_number}]\n{page_text}")
    return "\n\n".join(pages)


def read_source_text(path: Path) -> tuple[str, str]:
    """Read one supported source and return normalized text plus its media type."""

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported document type: {suffix}")
    if suffix == ".pdf":
        return _read_pdf(path), "application/pdf"

    raw = path.read_text(encoding="utf-8")
    if suffix in {".html", ".htm"}:
        parser = _TextHTMLParser()
        parser.feed(raw)
        return html.unescape(parser.text()), "text/html"
    if suffix in {".md", ".markdown"}:
        return raw, "text/markdown"
    return raw, "text/plain"


def load_document_records(source_dir: Path) -> list[DocumentRecord]:
    """Recursively load supported files into a deterministic document collection."""

    metadata_path = source_dir / SOURCE_METADATA_FILE
    source_metadata: dict[str, dict[str, Any]] = {}
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not all(
            isinstance(key, str) and isinstance(value, dict) for key, value in payload.items()
        ):
            raise ValueError(f"{SOURCE_METADATA_FILE} must map relative source paths to JSON objects")
        source_metadata = payload

    paths = sorted(
        path for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not paths:
        raise ValueError(f"No supported documents found in directory: {source_dir}")

    records = []
    for path in paths:
        text, media_type = read_source_text(path)
        if not text.strip():
            continue
        relative_source = path.relative_to(source_dir).as_posix()
        metadata = dict(source_metadata.get(relative_source, {}))
        digest = content_digest(text)
        records.append(
            DocumentRecord(
                document_id=stable_document_id(relative_source),
                version=digest[:16],
                source=relative_source,
                title=str(metadata.pop("title", path.stem)),
                text=text,
                content_hash=digest,
                media_type=media_type,
                metadata={"extension": path.suffix.lower(), **metadata},
            )
        )
    if not records:
        raise ValueError(f"All supported documents are blank in directory: {source_dir}")
    return records


def build_stable_chunks(
    document: DocumentRecord,
    *,
    chunk_size: int,
    overlap_size: int,
) -> list[ChunkRecord]:
    """Chunk a document using content-addressed identities stable across unrelated edits."""

    occurrences: Counter[str] = Counter()
    chunks = []
    for index, (text, start, end) in enumerate(
        chunk_text_with_spans(document.text, chunk_size, overlap_size)
    ):
        digest = content_digest(text)
        occurrence = occurrences[digest]
        occurrences[digest] += 1
        chunk_id = f"{document.document_id}:{digest[:20]}:{occurrence}"
        chunks.append(
            ChunkRecord(
                doc_id=document.document_id,
                chunk_id=chunk_id,
                chunk_index=index,
                source=document.source,
                page=document.page,
                text=text,
                document_version=document.version,
                content_hash=digest,
                start_char=start,
                end_char=end,
                section_path=list(document.section_path),
                metadata={**document.metadata, "title": document.title, "media_type": document.media_type},
            )
        )
    return chunks


@dataclass(frozen=True)
class ChangeSet:
    added_chunk_ids: list[str]
    reused_chunk_ids: list[str]
    removed_chunk_ids: list[str]
    added_document_ids: list[str]
    updated_document_ids: list[str]
    removed_document_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_changes(
    documents: list[DocumentRecord],
    chunks: list[ChunkRecord],
    previous_manifest: dict[str, Any] | None,
) -> ChangeSet:
    previous_manifest = previous_manifest or {"documents": [], "chunks": []}
    previous_documents = {item["document_id"]: item for item in previous_manifest.get("documents", [])}
    current_documents = {item.document_id: item for item in documents}
    previous_chunks = {item["chunk_id"]: item for item in previous_manifest.get("chunks", [])}
    current_chunks = {item.chunk_id: item for item in chunks}

    reused = sorted(
        chunk_id for chunk_id, chunk in current_chunks.items()
        if chunk_id in previous_chunks
        and previous_chunks[chunk_id].get("content_hash") == chunk.content_hash
    )
    added = sorted(set(current_chunks) - set(reused))
    removed = sorted(set(previous_chunks) - set(current_chunks))
    added_documents = sorted(set(current_documents) - set(previous_documents))
    removed_documents = sorted(set(previous_documents) - set(current_documents))
    updated_documents = sorted(
        document_id for document_id in set(current_documents) & set(previous_documents)
        if current_documents[document_id].content_hash != previous_documents[document_id].get("content_hash")
    )
    return ChangeSet(added, reused, removed, added_documents, updated_documents, removed_documents)


def load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema at {path}")
    return payload


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, records: Iterable[Any]) -> None:
    lines = [json.dumps(record.to_dict(), ensure_ascii=False) for record in records]
    _atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def sync_knowledge_base(
    documents: list[DocumentRecord],
    *,
    chunk_size: int,
    overlap_size: int,
    manifest_path: Path,
    chunks_path: Path,
    embeddings_path: Path,
    embed_texts: Callable[[list[str]], list[list[float]]],
) -> dict[str, Any]:
    """Synchronize durable artifacts while embedding only new or changed chunks."""

    chunks = [
        chunk
        for document in documents
        for chunk in build_stable_chunks(document, chunk_size=chunk_size, overlap_size=overlap_size)
    ]
    previous_manifest = load_manifest(manifest_path)
    changes = plan_changes(documents, chunks, previous_manifest)

    previous_embeddings: dict[str, EmbeddingRecord] = {}
    if embeddings_path.exists():
        from .loaders import load_embedding_records

        previous_embeddings = {
            item.chunk_id: item for item in load_embedding_records(embeddings_path)
        }

    reusable = {
        chunk_id: record for chunk_id, record in previous_embeddings.items()
        if chunk_id in set(changes.reused_chunk_ids)
    }
    new_chunks = [chunk for chunk in chunks if chunk.chunk_id not in reusable]
    vectors = embed_texts([chunk.text for chunk in new_chunks])
    if len(vectors) != len(new_chunks):
        raise ValueError("Embedding function returned a different number of vectors")

    new_embeddings = {
        chunk.chunk_id: EmbeddingRecord(
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            source=chunk.source,
            embedding=vector,
            page=chunk.page,
            text=chunk.text,
            document_version=chunk.document_version,
            content_hash=chunk.content_hash,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            section_path=list(chunk.section_path),
            metadata=dict(chunk.metadata),
        )
        for chunk, vector in zip(new_chunks, vectors)
    }
    embeddings = [reusable.get(chunk.chunk_id) or new_embeddings[chunk.chunk_id] for chunk in chunks]

    manifest = {
        "schema_version": 1,
        "documents": [document.to_dict() | {"text": None} for document in documents],
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.doc_id,
                "document_version": chunk.document_version,
                "content_hash": chunk.content_hash,
            }
            for chunk in chunks
        ],
    }
    _write_jsonl(chunks_path, chunks)
    _write_jsonl(embeddings_path, embeddings)
    _atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))
    return {
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "embedded_chunk_count": len(new_chunks),
        "reused_embedding_count": len(reusable),
        "changes": changes.to_dict(),
        "index_sync_strategy": "restart-and-rebuild-in-memory-hnsw",
    }
