"""
启动 src/azusa_rag/du_retrieval_download.py
"""

from bootstrap import ensure_src_path

ensure_src_path()

from azusa_rag.du_retrieval_download import main

if __name__ == "__main__":
    main()
