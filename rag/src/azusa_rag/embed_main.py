"""
把已经切好的chunk, 变成embedding输出文件
"""
from pathlib import Path

from .config import CHUNK_OUTPUT_PATH, EMBEDDING_OUTPUT_PATH
from .loaders import load_chunk_records
from .embedder import chunk_to_embedding_batch
from .main import write_jsonl

def build_embedding_artifacts(
    chunk_input_path: Path,
    embedding_output_path: Path,
) -> None:
    """
    把指定的 chunks jsonl 编码成 embedding jsonl。
    """
    chunk_record_list = load_chunk_records(jsonl_path=chunk_input_path) # list[ChunkRecord]
    embedding_record_list = chunk_to_embedding_batch(chunk_record_list= chunk_record_list) # list[EmbeddingRecord]
    write_jsonl(embedding_output_path, embedding_record_list)


def main() -> None:
    """
    chunks.jsonl -> list[EmbeddingRecord] -> embedding.jsonl   
    """
    build_embedding_artifacts(
        chunk_input_path=CHUNK_OUTPUT_PATH,
        embedding_output_path=EMBEDDING_OUTPUT_PATH,
    )

if __name__ == "__main__":
    main()
