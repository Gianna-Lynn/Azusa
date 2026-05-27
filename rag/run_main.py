"""
Root launcher for the chunk-building pipeline.
"""

from bootstrap import ensure_src_path

ensure_src_path()

from azusa_rag.main import main

if __name__ == "__main__":
    main()
