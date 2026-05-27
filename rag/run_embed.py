"""
Root launcher for chunk embedding generation.
"""

from bootstrap import ensure_src_path

ensure_src_path()

from azusa_rag.embed_main import main

if __name__ == "__main__":
    main()
