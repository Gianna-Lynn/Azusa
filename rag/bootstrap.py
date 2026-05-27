"""
Shared bootstrap helpers for root-level rag scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAG_ROOT = Path(__file__).resolve().parent
SRC_DIR = RAG_ROOT / "src"


def ensure_src_path() -> None:
    """
    Make ``rag/src`` importable for root-level helper scripts.
    """
    src_path = str(SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
