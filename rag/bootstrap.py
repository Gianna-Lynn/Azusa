"""
给 rag 根目录下的启动脚本共用的 bootstrap helper。

现在 run_main.py / run_embed.py / run_search.py / run_eval.py / run_workflow.py
都只是很薄的一层启动器，所以把“把 src 加入 import 路径”这件事集中到这里，
避免每个脚本都重复写一遍。
"""

from __future__ import annotations

import sys
from pathlib import Path

RAG_ROOT = Path(__file__).resolve().parent
SRC_DIR = RAG_ROOT / "src"


def ensure_src_path() -> None:
    """
    把 ``rag/src`` 加入 sys.path。

    这样根目录脚本就能直接 import `azusa_rag.*`，
    不需要每个文件自己重复处理路径问题。
    """
    src_path = str(SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
