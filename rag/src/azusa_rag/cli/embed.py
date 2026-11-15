"""
解析 embedding 命令参数，转交原有产物生成流程。
"""

import argparse
from pathlib import Path

from azusa_rag.config import CHUNK_OUTPUT_PATH, EMBEDDING_OUTPUT_PATH
from azusa_rag.embed_main import build_embedding_artifacts


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

def main(argv: list[str] | None = None) -> None:
    """解析默认或自定义产物路径；统一入口与模块入口使用同一条执行路径。"""
    args = build_parser().parse_args(argv)
    build_embedding_artifacts(
        chunk_input_path=Path(args.chunks_path),
        embedding_output_path=Path(args.embedding_output_path),
    )


if __name__ == "__main__":
    main()
