"""
说明 Azusa在内存中把一条chunk / 一条embedding 记录成什么样的结构.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

@dataclass
class ChunkRecord:      # 定义chunk metadata
    doc_id: str         # 这段chunk属于哪一份文档
    chunk_id: str       # 这是这份文档的第几块(放在chunk名中)
    chunk_index: int    # 这个chunk是这份文档的第几个(用于程序检索)
    source: str         # 原始来源, 比如文件名 filename.txt
    page: Optional[int] # 给PDF预留页码, txt下可以是None
    text: str           # 这个chunk的正文内容
    document_version: str = "legacy"
    content_hash: str = ""
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    section_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:  # 本方法可以把ChunkRecord这个对象变成普通字典, 字典适合json.dumps(...)
        return asdict(self)
    
@dataclass
class EmbeddingRecord:
    # 先暂时直接复用上面的ChunkRecord
    # EmbeddingRecord是一个dataclass, 需要用".'访问字段, 不是用"[]"访问字段
    doc_id: str         # 这段chunk属于哪一份文档
    chunk_id: str       # 这是这份文档的第几块(放在chunk名中)
    chunk_index: int    # 这个chunk是这份文档的第几个(用于程序检索)
    source: str         # 原始来源, 比如文件名 filename.txt
    embedding: list[float]   # 使用(专用的小型编码)模型编码出来的向量
    page: Optional[int] # 给PDF预留页码, txt下是None. # 注意带有默认值的可选字段需要放在必填字段的后面
    text: str           # 这个chunk的正文内容
    document_version: str = "legacy"
    content_hash: str = ""
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    section_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:  # 本方法可以把EmbeddingRecord这个对象变成普通字典, 字典适合json.dumps(...)
        return asdict(self)


@dataclass(frozen=True)
class DocumentRecord:
    """知识库中的规范化源文档；version 使用内容哈希而不是可变序号。"""

    document_id: str
    version: str
    source: str
    title: str
    text: str
    content_hash: str
    media_type: str
    page: Optional[int] = None
    section_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
