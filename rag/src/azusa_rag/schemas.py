from dataclasses import asdict, dataclass
from typing import Optional

@dataclass
class ChunkRecord:
    doc_id: str         # 这段chunk属于哪一份文档
    chunk_id: str       # 这是这份文档的第几块
    source: str         # 原始来源, 比如文件名 filename.txt
    text: str           # 这个chunk的正文内容
    page: Optional[int] = None  # 给PDF预留页码, txt下可以是None

    def to_dict(self) -> dict:  # 本方法可以把ChunkRecord这个对象变成普通字典, 字典适合json.dumps(...)
        return asdict(self)
    