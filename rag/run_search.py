"""
最小搜索脚本:
1. 从命令行读取搜索模式和 top-k
2. 调用 azusa_rag.search 里的主搜索流程
3. 在终端打印简明结果
4. 把结构化结果写进 output 文件
"""

import argparse
import json
from pathlib import Path

from azusa_rag.config import EMBEDDING_OUTPUT_PATH, OUTPUT_DIR
from azusa_rag.loaders import load_embedding_records
from azusa_rag.schemas import EmbeddingRecord
from azusa_rag.search import get_query, search


def build_parser() -> argparse.ArgumentParser:
    """
    统一定义命令行参数, 避免把 mode / top_k 写死在代码里.
    """
    parser = argparse.ArgumentParser(description="Run Azusa RAG search")
    parser.add_argument("--mode", choices=["brute", "hnsw"], default="hnsw")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-name", default="search_result")
    return parser


def make_preview(text: str, limit: int = 100) -> str:
    """
    生成短预览, 方便在终端和结果文件里快速扫读.
    """
    clean_text = text.replace("\n", " ").strip()
    if len(clean_text) <= limit:
        return clean_text
    return clean_text[:limit] + "..."


def build_result_dict(
    query: str,
    mode: str,
    top_k: int,
    result_records: list[EmbeddingRecord],
) -> dict:
    """
    把搜索结果整理成结构化字典, 方便后续写入 JSON 或做模式对比.
    """
    result_items = []

    for rank, record in enumerate(result_records, start=1):
        result_items.append(
            {
                "rank": rank,
                "mode": mode,
                "doc_id": record.doc_id,
                "chunk_id": record.chunk_id,
                "source": record.source,
                "chunk_index": record.chunk_index,
                "page": record.page,
                "preview": make_preview(record.text),
                "full_text": record.text,
            }
        )

    return {
        "query": query,
        "mode": mode,
        "top_k": top_k,
        "results": result_items,
    }


def print_results(result_dict: dict) -> None:
    """
    终端只打印摘要, 方便快速看 top-k 是否合理.
    """
    print(f"mode: {result_dict['mode']}")
    print(f"top_k: {result_dict['top_k']}")
    print(f"query: {make_preview(result_dict['query'], limit=120)}")
    print()

    for item in result_dict["results"]:
        print(f"[{item['rank']}] source={item['source']} chunk_id={item['chunk_id']}")
        print(f"    preview={item['preview']}")
        print()


def write_result_json(output_path: Path, result_dict: dict) -> None:
    """
    把结果写成 JSON, 方便后续人工检查和 brute / hnsw 对比.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = json.dumps(result_dict, ensure_ascii=False, indent=2)
    output_path.write_text(output_text, encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    query = get_query()
    embedding_records = load_embedding_records(EMBEDDING_OUTPUT_PATH)

    result_records = search(
        query=query,
        embedding_records=embedding_records,
        top_k=args.top_k,
        mode=args.mode,
    )

    result_dict = build_result_dict(
        query=query,
        mode=args.mode,
        top_k=args.top_k,
        result_records=result_records,
    )

    print_results(result_dict)

    output_path = OUTPUT_DIR / f"{args.output_name}_{args.mode}.json"
    write_result_json(output_path, result_dict)

    print(f"saved to: {output_path}")


if __name__ == "__main__":
    main()
