"""按显式模型身份同步知识库；清空现有证据须用 --allow-empty 明确选择。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bootstrap import ensure_src_path

ensure_src_path()

from azusa_rag.config import (
    CHUNK_OUTPUT_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_OUTPUT_PATH,
    KNOWLEDGE_BASE_MANIFEST_PATH,
    RAW_DATA_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
)
from azusa_rag.embedder import embed_text_list
from azusa_rag.knowledge_base import load_document_records, sync_knowledge_base


def build_parser() -> argparse.ArgumentParser:
    """定义源目录、独立产物路径、切块与编码身份参数，默认拒绝空输入。"""
    parser = argparse.ArgumentParser(description="Incrementally synchronize the Azusa knowledge base")
    parser.add_argument("--source-dir", default=str(RAW_DATA_DIR))
    parser.add_argument("--manifest-path", default=str(KNOWLEDGE_BASE_MANIFEST_PATH))
    parser.add_argument("--chunks-path", default=str(CHUNK_OUTPUT_PATH))
    parser.add_argument("--embeddings-path", default=str(EMBEDDING_OUTPUT_PATH))
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--embedding-identity", default=f"{EMBEDDING_MODEL_NAME}:dim={EMBEDDING_DIM}")
    return parser


def main() -> None:
    """运行同步并输出变更计数；非法输入或损坏旧产物直接以非零退出。"""
    args = build_parser().parse_args()
    documents = load_document_records(Path(args.source_dir), allow_empty=args.allow_empty)
    report = sync_knowledge_base(
        documents,
        chunk_size=args.chunk_size,
        overlap_size=args.overlap,
        manifest_path=Path(args.manifest_path),
        chunks_path=Path(args.chunks_path),
        embeddings_path=Path(args.embeddings_path),
        embed_texts=embed_text_list,
        embedding_identity=args.embedding_identity,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
