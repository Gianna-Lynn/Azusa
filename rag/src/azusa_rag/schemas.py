"""
说明 Azusa在内存中把一条chunk / 一条embedding 记录成什么样的结构.
schema的英文含义是“模式、结构、蓝图”，在这里指的是数据的结构定义。
它定义了数据的字段、类型和默认值等信息，确保数据在内存中有一致的表示方式。
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
    document_version: str = "legacy"    # 文档版本标识，用于区分同一文档的不同版本。
    content_hash: str = ""              # 文档或内容的哈希值，用于校验内容和识别变更。
    start_char: Optional[int] = None    # chunk 在原文中的起始字符位置，未知时为 None。
    end_char: Optional[int] = None      # chunk 在原文中的结束字符位置，未知时为 None。
    section_path: list[str] = field(default_factory=list)   # chunk 所属的层级章节路径。
    metadata: dict[str, Any] = field(default_factory=dict)  # 存放额外的业务元数据，如标签和权限信息。

    def to_dict(self) -> dict:  # 本方法可以把ChunkRecord这个对象变成普通字典, 字典适合json.dumps(...)
        return asdict(self)
    
@dataclass
class EmbeddingRecord:
    doc_id: str
    chunk_id: str
    chunk_index: int
    source: str
    embedding: list[float]  # 新增字段: 使用(专用的小型编码)模型编码出来的向量
    page: Optional[int]
    text: str
    document_version: str = "legacy"
    content_hash: str = ""
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    section_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)  # frozen=True 表示这个 dataclass 创建后，字段不能再被重新赋值，相当于“只读对象”。但是如果一个字段是字典, 那字典内部还是可以修改的. 所以是浅层只读.
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
