"""
知识库导入与增量同步：把源文档变成检索所需的三个产物文件。

三个产物（默认位于 rag/output/artifacts/）：
- chunks.jsonl：切块后的文本，每行一条 JSON
- embedding.jsonl：每个块对应的向量，每行一条 JSON
- knowledge_base_manifest.json（清单）：记录上次同步的状态——有哪些文档、
  哪些块、使用什么编码身份（embedding_identity），以及前两个文件的 SHA256
  摘要。清单是"上一代"的凭据，也是下次同步校验的锚点。

核心思路是增量同步：每次同步都重新切块并完整重写三个产物，但内容没变的块
直接复用上一代的向量，只对新增或变化的块重新调用 embedding 模型。块是否
"没变"由内容哈希（chunk_id 的一部分）与清单中记录的编码身份共同决定。

主要函数：
- load_document_records()：读取源目录中的文档，构造 DocumentRecord 列表
- build_stable_chunks()：把一篇文档切成 ChunkRecord 列表
- sync_knowledge_base()：本文件主入口，完成"对比规划 -> 增量编码 -> 写回
  三个产物"的全过程；产物只写到磁盘，Rust 内存索引需另行重启重建。
"""

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
    """
    计算文本内容的 SHA256 摘要，作为"内容是否变化"的判断依据。

    计算前先把换行符统一成 \\n 并去掉首尾空白，这样同一段内容在 Windows 或
    Linux 上、用不同编辑器保存后，算出的摘要仍然一致，不会被误判为变化。

    Args:
        value: 原始文本内容。

    Returns:
        64 个十六进制字符的 SHA256 摘要字符串。
    """
    # 统一换行符并去掉首尾空白，避免平台差异导致同一内容产生不同摘要。
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def stable_document_id(relative_source: str) -> str:
    """
    根据文档在源目录中的相对路径生成稳定的文档 ID。

    "稳定"指同一文件每次导入都得到相同 ID；路径大小写与分隔符先统一再计算，
    因此 ID 不随平台变化。ID 前半是可读的文件名主干，后半是路径 SHA256 的
    前 10 位，因此不同目录下的同名文件也能得到不同 ID，不会互相混淆。

    Args:
        relative_source: 文档相对源目录的路径，如 "notes/a.md"。

    Returns:
        形如 "<文件名主干>-<路径 SHA256 前 10 位>" 的稳定文档 ID。
    """
    # 统一分隔符和大小写，保证同一文件在不同平台上的身份一致。
    normalized = relative_source.replace("\\", "/").strip("/").lower()
    # 文件名主干只保留小写字母、数字和中文，其余字符替换为连字符。
    stem = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", Path(normalized).stem).strip("-") or "document"
    # 主干提供可读性，路径哈希前 10 位区分同名不同目录的文件。
    return f"{stem}-{hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:10]}"


class _TextHTMLParser(HTMLParser):
    """
    把 HTML 转成近似纯文本，供切块与检索使用。

    规则：script/style/noscript 标签内的内容整体丢弃（它们不是正文）；遇到
    块级标签（p、div、h1 等）补一个换行，避免相邻块的文本粘在一起；其余
    标签去掉、只保留标签里的文字。
    """
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []  # 累积的文本片段
        self.ignored_depth = 0  # 当前嵌套在被忽略标签内的深度

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # script/style/noscript 正文没有检索价值，记录进入深度以便整体丢弃。
        if tag in {"script", "style", "noscript"}:
            self.ignored_depth += 1
        # 块级标签起止视作换行，避免相邻块文本粘连。
        elif tag in {"p", "br", "div", "section", "article", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        # 离开被忽略标签时减少深度；未配对结束标签不会把深度减成负数。
        if tag in {"script", "style", "noscript"} and self.ignored_depth:
            self.ignored_depth -= 1
        elif tag in {"p", "div", "section", "article", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        # 被忽略标签内的数据直接丢弃。
        if not self.ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        # 按行压缩空白并丢弃完全空白的行，得到干净的纯文本。
        lines = [re.sub(r"\s+", " ", line).strip() for line in "".join(self.parts).splitlines()]
        return "\n".join(line for line in lines if line)


def _read_pdf(path: Path) -> str:
    """
    用 pypdf 提取 PDF 全文，每页前缀 "[page N]" 标注页码。

    Args:
        path: PDF 文件路径。

    Returns:
        按页拼接的纯文本；空白页被跳过。

    Raises:
        RuntimeError: 未安装可选依赖 pypdf。
    """
    # pypdf 是可选依赖，在函数内延迟导入，缺失时给出明确错误。
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - environment-specific dependency gate
        raise RuntimeError("PDF ingestion requires the optional 'pypdf' dependency") from exc

    pages = []
    # 页码从 1 开始编号，与阅读习惯一致；空白页不参与拼接。
    for page_number, page in enumerate(PdfReader(str(path)).pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            pages.append(f"[page {page_number}]\n{page_text}")
    return "\n\n".join(pages)


def read_source_text(path: Path) -> tuple[str, str]:
    """
    读取一个受支持的源文档，返回规范化文本和媒体类型（media type）。

    PDF 提取全文并按页标注页码；HTML 先去标签再反转义实体；txt/md 原样
    读取，只区分媒体类型。所有文本统一按 UTF-8 读取。

    Args:
        path: 源文档路径。

    Returns:
        (text, media_type) 二元组，media_type 取值如 "application/pdf"、
        "text/html"、"text/markdown"、"text/plain"。

    Raises:
        ValueError: 文件后缀不受支持。
    """
    # 阶段 1：后缀校验，不支持的格式直接拒绝，避免静默漏文档。
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported document type: {suffix}")
    # 阶段 2：PDF 走独立的可选依赖提取。
    if suffix == ".pdf":
        return _read_pdf(path), "application/pdf"

    # 阶段 3：文本类格式统一按 UTF-8 读取。
    raw = path.read_text(encoding="utf-8")
    # 阶段 4：HTML 先去除标签得到纯文本，再反转义 HTML 实体。
    if suffix in {".html", ".htm"}:
        parser = _TextHTMLParser()
        parser.feed(raw)
        return html.unescape(parser.text()), "text/html"
    # 阶段 5：其余格式原样返回，只区分媒体类型。
    if suffix in {".md", ".markdown"}:
        return raw, "text/markdown"
    return raw, "text/plain"


def load_document_records(source_dir: Path, *, allow_empty: bool = False) -> list[DocumentRecord]:
    """
    加载源目录中的全部文档，构造 DocumentRecord 列表（按路径排序）。

    递归发现受支持后缀的文件（见 SUPPORTED_SUFFIXES），读取内容、合并可选的
    逐文件元数据（azusa_metadata.json），并为每篇文档计算稳定的 ID 和内容
    哈希。空白文档被跳过；结果顺序固定，保证后续处理可复现。

    Args:
        source_dir: 源文档目录。
        allow_empty: 是否允许零文档；默认拒绝，防止空目录误清空知识库。

    Returns:
        按路径排序的 DocumentRecord 列表；空白文档被跳过。

    Raises:
        ValueError: 目录不存在、无受支持文档、元数据结构非法或全部文档为空。
    """

    # 阶段 1：校验源目录，避免在错误路径上执行加载或清理逻辑。
    if not source_dir.is_dir():
        raise ValueError(f"Source directory does not exist: {source_dir}")

    # 阶段 2：读取可选的逐文件元数据，并验证其 JSON 结构。
    metadata_path = source_dir / SOURCE_METADATA_FILE
    source_metadata: dict[str, dict[str, Any]] = {}
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not all(
            isinstance(key, str) and isinstance(value, dict) for key, value in payload.items()
        ):
            raise ValueError(f"{SOURCE_METADATA_FILE} must map relative source paths to JSON objects")
        source_metadata = payload

    # 阶段 3：递归发现支持的文件，并按路径排序以保证处理顺序稳定。
    paths = sorted(
        path for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not paths and not allow_empty:
        raise ValueError(f"No supported documents found in directory: {source_dir}")

    # 阶段 4：读取内容、合并元数据，并为每个文档构造稳定记录。
    records = []
    for path in paths:
        text, media_type = read_source_text(path)
        # 空白文档直接跳过，不进入知识库。
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
                # title 从逐文件元数据提取，缺省用文件名主干；其余元数据原样并入。
                title=str(metadata.pop("title", path.stem)),
                text=text,
                content_hash=digest,
                media_type=media_type,
                metadata={"extension": path.suffix.lower(), **metadata},
            )
        )

    # 阶段 5：拒绝全部为空的结果，除非调用方明确允许空集合。
    if not records and not allow_empty:
        raise ValueError(f"All supported documents are blank in directory: {source_dir}")
    return records


def build_stable_chunks(
    document: DocumentRecord,
    *,
    chunk_size: int,
    overlap_size: int,
) -> list[ChunkRecord]:
    """
    把一篇文档切成若干块，生成稳定的 ChunkRecord 列表。

    切块复用 chunker.chunk_text_with_spans，每个块保留它在原文中的字符位置
    （start_char/end_char），供检索侧定位证据。同一段文字在文档中出现多次
    时，按出现次序编号区分，保证 chunk_id 唯一，且每次切块结果一致。

    Args:
        document: 已加载的文档记录。
        chunk_size: 目标块大小（字符数）。
        overlap_size: 相邻块重叠大小（字符数）。

    Returns:
        ChunkRecord 列表，chunk_id 形如 "<doc_id>:<哈希前 20 位>:<出现序号>"。
    """
    # 同一内容多次出现时按出现次序递增序号，保证 chunk_id 唯一。
    occurrences: Counter[str] = Counter()
    chunks = []
    for index, (text, start, end) in enumerate(
        chunk_text_with_spans(document.text, chunk_size, overlap_size)
    ):
        # 内容哈希 + 出现序号构成块身份；序号只依赖本函数的确定性遍历顺序。
        digest = content_digest(text)
        occurrence = occurrences[digest]
        occurrences[digest] += 1
        chunks.append(ChunkRecord(
            doc_id=document.document_id,
            chunk_id=f"{document.document_id}:{digest[:20]}:{occurrence}",
            chunk_index=index, source=document.source, page=document.page, text=text,
            document_version=document.version, content_hash=digest,
            start_char=start, end_char=end, section_path=list(document.section_path),
            # metadata 补充标题和媒体类型，供检索侧展示与过滤。
            metadata={**document.metadata, "title": document.title, "media_type": document.media_type},
        ))
    return chunks


@dataclass(frozen=True)
class ChangeSet:
    """记录本次同步相比上一代清单，文档和块发生了哪些变化。

    六个字段都是排序后的 ID 列表，同一输入下输出顺序固定，方便测试与对比。
    "复用"指块 ID 和内容哈希都没变，即内容确实一字未动的块。
    """
    added_chunk_ids: list[str]  # 新增块（含内容变化的块）
    reused_chunk_ids: list[str]  # ID 与内容哈希均未变的复用块
    removed_chunk_ids: list[str]  # 已从当前集合消失的块
    added_document_ids: list[str]  # 新增文档
    updated_document_ids: list[str]  # 内容哈希变化的文档
    removed_document_ids: list[str]  # 已删除的文档

    def to_dict(self) -> dict[str, Any]:
        """返回可持久化的 JSON 兼容变更列表。"""
        return asdict(self)


def plan_changes(
    documents: list[DocumentRecord],
    chunks: list[ChunkRecord],
    previous_manifest: dict[str, Any] | None,
) -> ChangeSet:
    """对比当前文档/块与上一代清单，得出"哪些新增、哪些可复用、哪些删除"。

    对比规则：文档按内容哈希判断是否更新；块只有在 ID 和内容哈希都一致时才
    视为可复用（内容一字未动）。注意本函数只规划内容层面的变更，不判断旧
    向量能否复用——旧向量的编码身份与原文是否匹配，由同步入口
    sync_knowledge_base 在读取旧向量时核验。

    Args:
        documents: 当前完整文档集合。
        chunks: 当前完整块集合。
        previous_manifest: 上一代清单；None 表示首次同步（一切按新增处理），
            传入空集合则代表删除旧集合。

    Returns:
        各字段已排序的 ChangeSet。
    """
    # 阶段 1：归一化上一代清单并按 ID 建立索引；损坏的清单由 _index_rows 直接拒绝。
    previous = previous_manifest or {"documents": [], "chunks": []}
    old_docs = _index_rows(previous["documents"], "document_id")
    old_chunks = _index_rows(previous["chunks"], "chunk_id")
    current_docs = _index_rows([item.to_dict() for item in documents], "document_id")
    current_chunks = _index_rows([item.to_dict() for item in chunks], "chunk_id")
    # 阶段 2：块只有在 ID 与内容哈希都一致时才视为复用，其余情况按新块处理。
    reused = {key for key, item in current_chunks.items()
              if key in old_chunks and old_chunks[key]["content_hash"] == item["content_hash"]}
    # 阶段 3：集合运算得到各类变更，全部排序以稳定输出顺序。
    return ChangeSet(
        sorted(set(current_chunks) - reused), sorted(reused),
        sorted(set(old_chunks) - set(current_chunks)),
        sorted(set(current_docs) - set(old_docs)),
        sorted(key for key in set(old_docs) & set(current_docs)
               if old_docs[key]["content_hash"] != current_docs[key]["content_hash"]),
        sorted(set(old_docs) - set(current_docs)),
    )


def _index_rows(rows: list[dict], key: str) -> dict[str, dict]:
    """
    把行列表转成 {ID: 行} 的索引字典，转换前校验每一行的结构。

    每行都必须是非空字典，且 ID 唯一、content_hash 非空；发现损坏直接报错，
    而不是当成空库继续跑，避免把半截数据带进同步流程。

    Args:
        rows: 清单中的行列表。
        key: 行内用作身份的字段名，如 "document_id" 或 "chunk_id"。

    Returns:
        以 key 字段值为键、以行为值的索引字典。

    Raises:
        ValueError: rows 不是列表、行结构非法、ID 重复或缺少有效内容哈希。
    """
    if not isinstance(rows, list):
        raise ValueError("Manifest collections must be lists")
    indexed = {}
    for row in rows:
        # 每行必须是字典且 ID 为非空字符串。
        if not isinstance(row, dict) or not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(f"Invalid {key}")
        # ID 重复或缺少非空 content_hash 都是清单损坏，拒绝而非静默跳过。
        if row[key] in indexed or not isinstance(row.get("content_hash"), str) or not row["content_hash"]:
            raise ValueError(f"Duplicate identity or missing content hash: {row[key]}")
        indexed[row[key]] = row
    return indexed


def load_manifest(path: Path) -> dict[str, Any] | None:
    """读取并校验清单文件；文件不存在视为首次同步，返回 None。

    清单（manifest）是记录上次同步状态的 JSON：包含文档与块的 ID/哈希列表，
    v2 还包含编码身份（embedding_identity）和两个产物的 SHA256 摘要。旧版
    v1 清单也能读取，但它没有编码身份，同步时不会复用旧向量（全部重新编码）。
    本函数只校验清单本身的结构；v2 清单引用的两个产物是否与摘要一致，由同步
    入口在读取旧向量前核验。

    Args:
        path: 清单 JSON 文件路径。

    Returns:
        校验通过的清单 dict；文件不存在时返回 None。

    Raises:
        ValueError: JSON 结构非法、schema_version 不受支持、文档/块行损坏，
            或 v2 清单缺少有效的编码身份与产物摘要。
    """
    # 阶段 1：文件不存在视为首次同步，返回 None。
    if not path.exists():
        return None
    # 阶段 2：解析 JSON 并校验 schema_version，只接受 v1/v2。
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or type(payload.get("schema_version")) is not int or payload["schema_version"] not in {1, 2}:
        raise ValueError(f"Unsupported manifest schema at {path}")
    # 阶段 3：文档与块集合必须能通过结构校验（唯一 ID + 有效哈希）。
    _index_rows(payload.get("documents"), "document_id")
    _index_rows(payload.get("chunks"), "chunk_id")
    # 阶段 4：v2 额外要求编码身份非空、两个产物摘要为合法 SHA256。
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
    """
    原子地写一个文件：先写同目录下的唯一临时文件，再 rename 覆盖目标。

    "原子"指任何时刻读者看到的文件要么是完整的旧内容、要么是完整的新内容，
    不会出现写了一半的状态；进程中断也只会留下一个可清理的临时文件。注意
    这只对单个文件成立，跨文件的顺序保证见 sync_knowledge_base。

    Args:
        path: 目标产物路径。
        text: 要写入的完整文本。
    """
    # 确保父目录存在，首次同步时产物目录可能尚未创建。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 唯一临时名避免多进程冲突；与目标同目录保证 replace 是原子 rename。
    temporary = path.with_name(path.name + f".{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        # replace 原子覆盖目标文件，读取方只会看到旧版或新版。
        temporary.replace(path)
    finally:
        # 无论成功与否都清理临时文件，防止目录里残留垃圾。
        temporary.unlink(missing_ok=True)


def _write_jsonl(path: Path, records: Iterable[Any]) -> None:
    """
    把记录集合逐行转成 JSON（jsonl 格式）并原子写回。

    序列化在写盘前整体完成，任何一条转 JSON 失败都不会写文件。空集合写空
    文件——这是"删除"的表示方式，旧文件里已删除的行不会残留。

    Args:
        path: 目标 jsonl 产物路径。
        records: 带 to_dict 的记录集合，按给定顺序逐行写入。
    """
    # 先完整序列化再写盘，序列化失败不会产生半截文件。
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
    """切块、编码并完整重写三个产物；未变化的块复用旧向量，避免重复调用模型。

    先对照上一代 manifest 找出内容未变的块：编码身份（embedding_identity）相同
    时直接复用其旧向量，只对新增或变化的块调用编码回调。更换 embedding 模型、
    维度或预处理方式时，必须同步更换编码身份，否则会把不同模型产生的向量混在
    一起复用；默认身份只是兼容占位，由调用方自行管理（run.py kb-sync 会按当前
    模型配置生成身份）。

    三个文件依次写入、manifest 最后提交，整体不是原子事务，假定单写者调用。
    manifest 记录前两个文件的 SHA256 摘要，下次同步会先校验：上次写入被中断
    （断电）或文件被外部改动时直接报错，拒绝把新旧两代内容混用。编码失败发生在
    任何写入之前，不会留下半截产物。

    参数为当前完整文档集合（空集合 = 清空全部旧证据）、三个互不相同的产物路径
    和批量编码回调；回调返回的向量必须与待编码文本一一对应，且非空、有限值、
    维度一致。返回统计计数与 ChangeSet；本函数只更新文件，Rust 索引仍需重启
    重建，不能据此宣称在线更新。

    Args:
        documents: 当前完整文档集合（空集合表示删除全部旧证据）。
        chunk_size: 目标块大小（字符数）。
        overlap_size: 相邻块重叠大小（字符数）。
        manifest_path: 清单产物路径。
        chunks_path: 块产物路径。
        embeddings_path: 向量产物路径。
        embed_texts: 批量编码回调，输入文本列表、输出等长向量列表。
        embedding_identity: 编码身份，模型/维度变化时必须更换。

    Returns:
        统计 dict：schema_version、各类计数、复用向量数、ChangeSet 以及
        index_sync_strategy。

    Raises:
        ValueError: 参数非法、产物路径重复、旧代产物摘要不匹配、向量数量或
            维度非法等。
    """
    # 阶段 1：参数与产物路径校验，防止把两个产物写到同一路径。
    if type(chunk_size) is not int or type(overlap_size) is not int or not 0 <= overlap_size < chunk_size:
        raise ValueError("Invalid chunk parameters")
    if not isinstance(embedding_identity, str) or not embedding_identity.strip():
        raise ValueError("Embedding identity must be nonblank")
    if len({path.resolve() for path in (manifest_path, chunks_path, embeddings_path)}) != 3:
        raise ValueError("Artifact paths must be distinct")
    # 阶段 2：对全部文档重新切块，构造当前代完整块集合。
    chunks = [chunk for document in documents for chunk in build_stable_chunks(
        document, chunk_size=chunk_size, overlap_size=overlap_size)]
    # 阶段 3：读取上一代清单并规划内容变更。
    previous = load_manifest(manifest_path)
    changes = plan_changes(documents, chunks, previous)
    # 阶段 4：v2 旧代校验——两个产物摘要必须与清单一致，向量行与块集合也必须
    # 一致，否则说明上次写入被中断或文件被外部改动，拒绝混合代际复用。
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
    # 阶段 5：收集可复用向量——块未变、编码身份相同且旧向量原文与当前块一致。
    unchanged_ids = set(changes.reused_chunk_ids)
    reusable = {
        chunk.chunk_id: old_embeddings[chunk.chunk_id]["embedding"] for chunk in chunks
        if previous and previous.get("embedding_identity") == embedding_identity
        and chunk.chunk_id in unchanged_ids and chunk.chunk_id in old_embeddings
        and old_embeddings[chunk.chunk_id]["text"] == chunk.text
    }
    # 阶段 6：只为新块调用编码回调；空集合不调用编码。
    new_chunks = [chunk for chunk in chunks if chunk.chunk_id not in reusable]
    vectors = embed_texts([chunk.text for chunk in new_chunks]) if new_chunks else []
    if len(vectors) != len(new_chunks):
        raise ValueError("Embedding function returned a different number of vectors")
    # 阶段 7：合并复用向量与新编码向量，校验数量、取值有限性和维度一致。
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
    # 阶段 8：组装本代 EmbeddingRecord——只复用向量；版本、标题、span、ACL 等
    # 均取当前块，避免旧元数据越权或漂移。
    embeddings = [EmbeddingRecord(**chunk.to_dict(), embedding=list(by_id[chunk.chunk_id])) for chunk in chunks]
    chunk_text = _jsonl(chunks)
    embedding_text = _jsonl(embeddings)
    # 阶段 9：构造清单——文档正文置 None 以减小体积，并记录两个产物的摘要。
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
    # 阶段 10：依次写块、写向量、最后提交清单；清单是下次同步的代际锚点。
    _atomic_write_text(chunks_path, chunk_text)
    _atomic_write_text(embeddings_path, embedding_text)
    _atomic_write_text(manifest_path, manifest_text)
    return {"schema_version": 2, "document_count": len(documents), "chunk_count": len(chunks),
            "embedded_chunk_count": len(new_chunks), "reused_embedding_count": len(reusable),
            "changes": changes.to_dict(), "index_sync_strategy": "restart-and-rebuild-in-memory-hnsw"}


def _jsonl(records: Iterable[Any]) -> str:
    """
    把记录集合转成 jsonl 文本；NaN/Infinity 等非法值在转换时即被拒绝。

    这一步在写盘之前完成，保证要么整体成功、要么一个字节都不写，不会出现
    写完一半才发现某条 metadata 非法的情况。

    Args:
        records: 带 to_dict 的记录集合。

    Returns:
        完整 jsonl 文本（每行一条记录，末尾换行；空集合返回空字符串）。
    """
    lines = [json.dumps(record.to_dict(), ensure_ascii=False, allow_nan=False) for record in records]
    return "\n".join(lines) + ("\n" if lines else "")
