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

    # TODO AZUSA-V2-L1a：实现规范化内容哈希。
    # 协议：CRLF/CR 统一为 LF，再 strip 首尾空白，按 UTF-8 计算 SHA-256，
    # 返回 64 位小写十六进制字符串。不得使用 Python hash()，它跨进程不稳定。
    raise NotImplementedError("TODO AZUSA-V2-L1a：实现规范化内容哈希")


def stable_document_id(relative_source: str) -> str:
    """Create a readable, collision-resistant identity from the source path."""

    # TODO AZUSA-V2-L1b：实现稳定文档 ID。
    # 1. 路径分隔符统一为 /，去掉两端 / 并转小写；
    # 2. stem 中非字母、数字、汉字的连续片段折叠为 -，空结果回退 document；
    # 3. 追加规范化完整相对路径 SHA-256 的前 10 位，避免同名文件碰撞。
    raise NotImplementedError("TODO AZUSA-V2-L1b：实现稳定文档 ID")


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

    # TODO AZUSA-V2-L2a：组织稳定 ChunkRecord。
    # 黑盒契约：复用 chunk_text_with_spans；chunk_id 由 document_id、chunk 内容哈希
    # 前 20 位和同内容 occurrence 组成；保留 span、版本、section 与 metadata。
    # 文档无关位置发生改变时，内容未变的 chunk_id 必须保持稳定。
    raise NotImplementedError("TODO AZUSA-V2-L2a：实现稳定 chunk 身份与血缘")


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
    # TODO AZUSA-V2-L2b：根据当前集合与上一版 manifest 生成确定性 ChangeSet。
    # reused 必须同时满足 chunk_id 和 content_hash 相同；所有输出列表排序；
    # added 不能包含 reused；removed 表示上一版存在而当前不存在；文档按内容哈希
    # 区分 added/updated/removed。previous_manifest=None 等价于空知识库。
    raise NotImplementedError("TODO AZUSA-V2-L2b：实现增量变更规划")


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

    # TODO AZUSA-V2-L2c：完成 Knowledge Base v1 的增量同步事务。
    # 必须组织 stable chunks、读取/校验 manifest、复用 unchanged embedding、只编码
    # 新增或变化 chunk、拒绝向量数量不匹配，并以当前 chunk 顺序写三份原子产物。
    # 删除通过“不再写入当前 artifacts”实现；Rust HNSW 的明确策略仍是重启重建。
    # 返回报告字段与 docs/knowledge_base_v1.md、黑盒测试保持一致。
    raise NotImplementedError("TODO AZUSA-V2-L2c：实现增量同步与 embedding 复用")
