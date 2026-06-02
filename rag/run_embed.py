"""
本文件是 embedding 流程的启动器：
1. 调 bootstrap，把 src 加到 Python 搜索路径里
2. 再调用真正的 azusa_rag.embed_main.main()
"""

import argparse
from pathlib import Path

from bootstrap import ensure_src_path

ensure_src_path()

from azusa_rag.config import CHUNK_OUTPUT_PATH, EMBEDDING_OUTPUT_PATH
from azusa_rag.embed_main import build_embedding_artifacts, main


def build_parser() -> argparse.ArgumentParser:
    """
    允许在保留默认行为的同时，按需把别的 chunk 文件接进 embedding 流程。
    """
    parser = argparse.ArgumentParser(description="Build embedding artifacts")
    parser.add_argument(
        "--chunks-path",
        default=str(CHUNK_OUTPUT_PATH),
        help="Path to the input chunks jsonl. Defaults to output/artifacts/chunks.jsonl.",
    )
    parser.add_argument(
        "--embedding-output-path",
        default=str(EMBEDDING_OUTPUT_PATH),
        help="Path to the output embedding jsonl. Defaults to output/artifacts/embedding.jsonl.",
    )
    return parser

if __name__ == "__main__":
    args = build_parser().parse_args()

    if (
        Path(args.chunks_path) == CHUNK_OUTPUT_PATH
        and Path(args.embedding_output_path) == EMBEDDING_OUTPUT_PATH
    ):
        main()
    else:
        # 这里专门给 DuRetrieval 这类“旁路数据源”留入口，
        # 这样我们不用覆盖项目默认的 artifacts，也能单独跑一条新链路。
        build_embedding_artifacts(
            chunk_input_path=Path(args.chunks_path),
            embedding_output_path=Path(args.embedding_output_path),
        )
