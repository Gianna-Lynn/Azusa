"""
把已经切好的chunk, 变成embedding输出文件
"""
from .config import CHUNK_OUTPUT_PATH, EMBEDDING_OUTPUT_PATH
from .loaders import load_chunk_records
from .embedder import chunk_to_embedding_batch
from .main import write_jsonl

def main() -> None:
    """
    chunks.jsonl -> list[EmbeddingRecord] -> embedding.jsonl   
    """
    chunk_record_list = load_chunk_records(jsonl_path= CHUNK_OUTPUT_PATH) # list[ChunkRecord]
    embedding_record_list = chunk_to_embedding_batch(chunk_record_list= chunk_record_list) # list[EmbeddingRecord]
    write_jsonl(EMBEDDING_OUTPUT_PATH, embedding_record_list)

if __name__ == "__main__":
    main()