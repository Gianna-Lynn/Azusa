"""基础切块入口：读取源文档，切块并写入配置指定的 chunks JSONL。"""

from pathlib import Path
from .schemas import ChunkRecord
from .chunker import chunk_document
from .config import RAW_DATA_DIR, CHUNK_OUTPUT_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from .loaders import load_documents
import json

def write_jsonl(output_path: Path, records: list[ChunkRecord]) -> None:
    """逐行序列化 ChunkRecord，写入 UTF-8 JSONL；覆盖指定输出文件。"""


    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:

        for record in records:
            record_dict = record.to_dict()
            json_line = json.dumps(record_dict, ensure_ascii=False)
            output_file.write(json_line)
            output_file.write("\n")


def main() -> None:
    """按配置加载文档、切块并写入 chunks 产物。"""


    raw_data_dir = RAW_DATA_DIR
    chunk_output_path = CHUNK_OUTPUT_PATH
    chunk_size = CHUNK_SIZE
    chunk_overlap = CHUNK_OVERLAP

    documents = load_documents(raw_data_dir)

    chunk_list = []

    for document in documents:
        chunkrec_list = chunk_document(
            doc_dict=document,
            chunk_size=chunk_size,
            overlap_size=chunk_overlap
        )


        chunk_list.extend(chunkrec_list)

    write_jsonl(chunk_output_path, chunk_list)

if __name__ == "__main__":
    main()
