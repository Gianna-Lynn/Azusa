"""
schemas.py: 定义数据结构
"""

from dataclasses import asdict, dataclass
from typing import Optional

@dataclass
class ChunkRecord:      # 定义chunk metadata
    doc_id: str         # 这段chunk属于哪一份文档
    chunk_id: str       # 这是这份文档的第几块(放在chunk名中)
    chunk_index: int    # 这个chunk是这份文档的第几个(用于程序检索)
    source: str         # 原始来源, 比如文件名 filename.txt
    text: str           # 这个chunk的正文内容
    page: Optional[int] = None  # 给PDF预留页码, txt下可以是None
    # TODO:
    # 补充start_char, end_char, 用于更精确的原文定位

    def to_dict(self) -> dict:  # 本方法可以把ChunkRecord这个对象变成普通字典, 字典适合json.dumps(...)
        return asdict(self)
    
@dataclass
class EmbeddingRecord:
    # 先暂时直接复用上面的ChunkRecord
    doc_id: str         # 这段chunk属于哪一份文档
    chunk_id: str       # 这是这份文档的第几块(放在chunk名中)
    chunk_index: int    # 这个chunk是这份文档的第几个(用于程序检索)
    source: str         # 原始来源, 比如文件名 filename.txt
    text: str           # 这个chunk的正文内容
    embedding: list[float]  
    page: Optional[int] = None  # 给PDF预留页码, txt下可以是None. # 注意带有默认值的可选字段需要放在必填字段的后面

    
    def to_dict(self) -> dict:  # 本方法可以把EmbeddingRecord这个对象变成普通字典, 字典适合json.dumps(...)
        return asdict(self)
    