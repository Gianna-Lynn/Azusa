"""
启动 src/azusa_rag/hf_dataset_import.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent # 这个路径指向 rag/，因为这个启动脚本本身就位于 rag/ 下.
SRC_DIR = PROJECT_ROOT / "src" # 这个路径指向 rag/src，下面的 azusa_rag 包就在这里.
if str(SRC_DIR) not in sys.path:
    # 让这个启动器可以直接 import src/ 下的 azusa_rag 包.
    sys.path.insert(0, str(SRC_DIR))

from azusa_rag.hf_dataset_import import main

if __name__ == "__main__":
    # 真正的业务逻辑在包内模块里, 这里仅负责做轻量启动转发.
    main()
