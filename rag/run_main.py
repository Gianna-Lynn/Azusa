"""
本文件不是包 azusa_rag 内部业务模块，而是一个很薄的启动器：
1. 调 bootstrap，把 src 加到 Python 搜索路径里
2. 再调用真正的 azusa_rag.main.main()
"""

from bootstrap import ensure_src_path

ensure_src_path()

from azusa_rag.main import main

if __name__ == "__main__":
    main()
