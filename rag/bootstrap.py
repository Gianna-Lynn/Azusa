"""为 rag 命令行启动器配置统一的包搜索路径。"""

from __future__ import annotations

import sys
from pathlib import Path

RAG_ROOT = Path(__file__).resolve().parent
SRC_DIR = RAG_ROOT / "src"


def ensure_src_path() -> None:
    """将 rag/src 加入包搜索路径，避免启动器重复配置。"""
    src_path = str(SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
