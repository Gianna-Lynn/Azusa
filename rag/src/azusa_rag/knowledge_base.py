"""提供确定性知识库导入与增量同步；持久化证据，Rust 内存索引另行重建。"""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

from .chunker import chunk_text_with_spans
from .schemas import ChunkRecord, DocumentRecord, EmbeddingRecord

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf"}
SOURCE_METADATA_FILE = "azusa_metadata.json"


def content_digest(value: str) -> str:
    """规范化换行及首尾空白后计算 SHA256，保证跨进程身份稳定。"""
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def stable_document_id(relative_source: str) -> str:
    """由大小写无关的相对路径生成可读 ID；同名不同目录保留不同身份。"""
    normalized = relative_source.replace("\\", "/").strip("/").lower()
    stem = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", Path(normalized).stem).strip("-") or "document"
    return f"{stem}-{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:10]}"


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


def load_document_records(source_dir: Path, *, allow_empty: bool = False) -> list[DocumentRecord]:
    """按路径排序加载文档；缺失目录失败，空集合须显式允许以防误删知识库。"""

    if not source_dir.is_dir():
        raise ValueError(f"Source directory does not exist: {source_dir}")

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
    if not paths and not allow_empty:
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
    if not records and not allow_empty:
        raise ValueError(f"All supported documents are blank in directory: {source_dir}")
    return records


def build_stable_chunks(
    document: DocumentRecord,
    *,
    chunk_size: int,
    overlap_size: int,
) -> list[ChunkRecord]:
    """复用现有切块器，保留原文 span，以内容与出现次数区分同文档的重复块。"""
    occurrences: Counter[str] = Counter()
    chunks = []
    for index, (text, start, end) in enumerate(
        chunk_text_with_spans(document.text, chunk_size, overlap_size)
    ):
        digest = content_digest(text)
        occurrence = occurrences[digest]
        occurrences[digest] += 1
        chunks.append(ChunkRecord(
            doc_id=document.document_id,
            chunk_id=f"{document.document_id}:{digest[:20]}:{occurrence}",
            chunk_index=index, source=document.source, page=document.page, text=text,
            document_version=document.version, content_hash=digest,
            start_char=start, end_char=end, section_path=list(document.section_path),
            metadata={**document.metadata, "title": document.title, "media_type": document.media_type},
        ))
    return chunks


@dataclass(frozen=True)
class ChangeSet:
    """保存排序后的文档和块变更；复用要求 ID 与内容哈希同时相同。"""
    added_chunk_ids: list[str]
    reused_chunk_ids: list[str]
    removed_chunk_ids: list[str]
    added_document_ids: list[str]
    updated_document_ids: list[str]
    removed_document_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        """返回可持久化的 JSON 兼容变更列表。"""
        return asdict(self)


def plan_changes(
    documents: list[DocumentRecord],
    chunks: list[ChunkRecord],
    previous_manifest: dict[str, Any] | None,
) -> ChangeSet:
    """比较当前文档、块与上一代清单，返回各字段已排序的 ChangeSet。

    previous_manifest=None 表示首次同步，当前空集合表示删除旧集合。文档更新按内容
    哈希判断；块复用要求 ID 和哈希同时一致。这里只规划内容变更，不判断向量可否
    复用，编码身份和实际正文还须由同步入口检查。重复身份或缺失哈希直接拒绝。
    """
    previous = previous_manifest or {"documents": [], "chunks": []}
    old_docs = _index_rows(previous["documents"], "document_id")
    old_chunks = _index_rows(previous["chunks"], "chunk_id")
    current_docs = _index_rows([item.to_dict() for item in documents], "document_id")
    current_chunks = _index_rows([item.to_dict() for item in chunks], "chunk_id")
    reused = {key for key, item in current_chunks.items()
              if key in old_chunks and old_chunks[key]["content_hash"] == item["content_hash"]}
    return ChangeSet(
        sorted(set(current_chunks) - reused), sorted(reused),
        sorted(set(old_chunks) - set(current_chunks)),
        sorted(set(current_docs) - set(old_docs)),
        sorted(key for key in set(old_docs) & set(current_docs)
               if old_docs[key]["content_hash"] != current_docs[key]["content_hash"]),
        sorted(set(old_docs) - set(current_docs)),
    )


def _index_rows(rows: list[dict], key: str) -> dict[str, dict]:
    """校验集合基本结构和唯一 ID，再构造索引；不把损坏清单视为空库。"""
    if not isinstance(rows, list):
        raise ValueError("Manifest collections must be lists")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"Invalid {key}")
        if row[key] in indexed or not isinstance(row.get("content_hash"), str) or not row["content_hash"]:
            raise ValueError(f"Duplicate identity or missing content hash: {row[key]}")
        indexed[row[key]] = row
    return indexed


def load_manifest(path: Path) -> dict[str, Any] | None:
    """读取 v1/v2 清单并校验结构；文件不存在返回 None，损坏或未知版本直接失败。

    v1 可迁移，但同步入口不会复用其缺少编码身份的旧向量。本函数仅校验清单本身，
    v2 引用的两个产物是否存在且摘要一致，由同步入口在读取旧向量前核验。
    """
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] not in {1, 2}:
        raise ValueError(f"Unsupported manifest schema at {path}")
    _index_rows(payload.get("documents"), "document_id")
    _index_rows(payload.get("chunks"), "chunk_id")
    if payload["schema_version"] == 2:
        digests = payload.get("artifacts_sha256")
        if not isinstance(payload.get("embedding_identity"), str) or not payload["embedding_identity"].strip():
            raise ValueError("Manifest embedding identity is missing")
        if not isinstance(digests, dict) or any(
            not isinstance(digests.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", digests[key])
            for key in ("chunks", "embeddings")
        ):
            raise ValueError("Manifest artifact digests are missing or invalid")
    return payload


def _atomic_write_text(path: Path, text: str) -> None:
    """以同目录唯一临时文件替换单个产物；不承诺多个文件的事务原子性。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl(path: Path, records: Iterable[Any]) -> None:
    """序列化当前有序集合；空集合写空文件，不保留已删除的旧行。"""
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
    embedding_identity: str = "caller-managed-v1",
) -> dict[str, Any]:
    """验证后增量生成 v2 产物，刷新元数据，仅复用同编码身份下的未变内容。

    默认身份供兼容的回调调用方自行管理；模型/维度/预处理变化必须改身份。命令行显式
    记录模型配置。单写者使用，三个文件不是整体原子事务；清单最后提交，下次同步
    用文件摘要拒绝断电或外部修改造成的混合代际。编码失败发生在任何产物替换之前。

    输入为当前完整文档集合、三条不同的产物路径和批量编码回调；回调返回值必须与
    待编码文本一一对应，向量非空、有限且维度一致。空集合会清除旧证据，不调用编码。
    返回计数与 ChangeSet；返回成功后仍需重启重建 Rust 索引，不能据此宣称在线更新。
    """
    if type(chunk_size) is not int or type(overlap_size) is not int or not 0 <= overlap_size < chunk_size:
        raise ValueError("Invalid chunk parameters")
    if not isinstance(embedding_identity, str) or not embedding_identity.strip():
        raise ValueError("Embedding identity must be nonblank")
    if len({path.resolve() for path in (manifest_path, chunks_path, embeddings_path)}) != 3:
        raise ValueError("Artifact paths must be distinct")
    chunks = [chunk for document in documents for chunk in build_stable_chunks(
        document, chunk_size=chunk_size, overlap_size=overlap_size)]
    previous = load_manifest(manifest_path)
    changes = plan_changes(documents, chunks, previous)
    old_embeddings = {}
    if previous and previous["schema_version"] == 2:
        for name, path in (("chunks", chunks_path), ("embeddings", embeddings_path)):
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != previous["artifacts_sha256"][name]:
                raise ValueError("Artifact generation mismatch; restore a complete generation")
        rows = [json.loads(line) for line in embeddings_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        old_embeddings = _index_rows(rows, "chunk_id")
        old_chunks = _index_rows(previous["chunks"], "chunk_id")
        if set(old_embeddings) != set(old_chunks) or any(
            old_embeddings[key]["content_hash"] != old_chunks[key]["content_hash"] for key in old_chunks
        ):
            raise ValueError("Embedding identities do not match manifest")
    unchanged_ids = set(changes.reused_chunk_ids)
    reusable = {
        chunk.chunk_id: old_embeddings[chunk.chunk_id]["embedding"] for chunk in chunks
        if previous and previous.get("embedding_identity") == embedding_identity
        and chunk.chunk_id in unchanged_ids and chunk.chunk_id in old_embeddings
        and old_embeddings[chunk.chunk_id]["text"] == chunk.text
    }
    new_chunks = [chunk for chunk in chunks if chunk.chunk_id not in reusable]
    vectors = embed_texts([chunk.text for chunk in new_chunks]) if new_chunks else []
    if len(vectors) != len(new_chunks):
        raise ValueError("Embedding function returned a different number of vectors")
    by_id = {**reusable, **{chunk.chunk_id: vector for chunk, vector in zip(new_chunks, vectors)}}
    dimensions = set()
    for vector in by_id.values():
        if not isinstance(vector, (list, tuple)) or not vector or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in vector
        ):
            raise ValueError("Embedding vectors must be nonempty and finite")
        dimensions.add(len(vector))
    if len(dimensions) > 1:
        raise ValueError("Embedding dimensions must match")
    # 只复用向量；版本、标题、span、ACL 等均取当前块，避免旧元数据越权或漂移。
    embeddings = [EmbeddingRecord(**chunk.to_dict(), embedding=list(by_id[chunk.chunk_id])) for chunk in chunks]
    chunk_text = _jsonl(chunks)
    embedding_text = _jsonl(embeddings)
    manifest = {
        "schema_version": 2, "embedding_identity": embedding_identity,
        "chunk_config": {"size": chunk_size, "overlap": overlap_size},
        "documents": [document.to_dict() | {"text": None} for document in documents],
        "chunks": [{"chunk_id": chunk.chunk_id, "document_id": chunk.doc_id,
                    "document_version": chunk.document_version, "content_hash": chunk.content_hash}
                   for chunk in chunks],
        "artifacts_sha256": {
            "chunks": hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
            "embeddings": hashlib.sha256(embedding_text.encode("utf-8")).hexdigest(),
        },
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)
    _atomic_write_text(chunks_path, chunk_text)
    _atomic_write_text(embeddings_path, embedding_text)
    _atomic_write_text(manifest_path, manifest_text)
    return {"schema_version": 2, "document_count": len(documents), "chunk_count": len(chunks),
            "embedded_chunk_count": len(new_chunks), "reused_embedding_count": len(reusable),
            "changes": changes.to_dict(), "index_sync_strategy": "restart-and-rebuild-in-memory-hnsw"}


def _jsonl(records: Iterable[Any]) -> str:
    """在任何写入之前完成有限 JSON 编码，避免部分写入后才发现非法 metadata。"""
    lines = [json.dumps(record.to_dict(), ensure_ascii=False, allow_nan=False) for record in records]
    return "\n".join(lines) + ("\n" if lines else "")
