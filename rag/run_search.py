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
import sys

RAG_ROOT = Path(__file__).resolve().parent
SRC_DIR = RAG_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from azusa_rag.config import EMBEDDING_OUTPUT_PATH, SEARCH_RUNS_DIR
from azusa_rag.loaders import load_embedding_records
from azusa_rag.schemas import EmbeddingRecord
from azusa_rag.search import (
    get_query,
    search,
    ts030_search_hybrid_rerank_with_sources,
    ts030_search_hybrid_with_sources,
)
from azusa_rag.answerer import answer_with_citations

SEARCH_MODES = ["brute", "hnsw", "hybrid", "hybrid_rerank", "bm25"]

def build_parser() -> argparse.ArgumentParser:
    """
    统一定义命令行参数, 避免把 mode / top_k 写死在代码里.
    """
    parser = argparse.ArgumentParser(description="Run Azusa RAG search")
    parser.add_argument("--mode", choices=SEARCH_MODES, default="hnsw")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional custom output basename. Omit it to overwrite the default fixed output file for the chosen mode.",
    )
    parser.add_argument("--with-answer", action="store_true", help="Whether to run the answerer on top of search results.")
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument("--query-text", help="Use an explicit query string instead of the default query file.")
    query_group.add_argument(
        "--query-file",
        help="Read query text from a specific file path instead of the default query file.",
    )
    return parser


def load_query_text(args: argparse.Namespace) -> str:
    """
    统一处理 query 输入来源:
    - 优先用命令行显式传入的 query_text
    - 其次读指定 query_file
    - 最后回退到默认 queries 目录中的第一条 query
    """
    if args.query_text is not None:
        query_text = args.query_text.strip()
        if not query_text:
            raise ValueError("--query-text cannot be blank")
        return query_text

    if args.query_file is not None:
        query_path = Path(args.query_file)
        query_text = query_path.read_text(encoding="utf-8").strip()
        if not query_text:
            raise ValueError(f"Query file is blank: {query_path}")
        return query_text

    return get_query()


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
    source_dict_list: list[dict] | None = None,
    answer_dict: dict | None = None,
) -> dict:
    """
    把搜索结果整理成结构化字典, 方便后续写入 JSON 或做模式对比.
    """
    result_items = []

    for rank, record in enumerate(result_records, start=1):
        result_item = {
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
        if source_dict_list is not None:
            result_item["retrieval_source"] = source_dict_list[rank - 1]["source"]
        
        result_items.append(result_item)

    result_dict = {
        "query": query,
        "mode": mode,
        "top_k": top_k,
        "results": result_items,
    }

    if answer_dict is not None:  # 如果提供了 answer_dict，就把其中的答案文本、原始输出、引用的chunk_id列表和上下文信息添加到结果字典中.
        result_dict["answer"] = answer_dict["answer_text"]
        result_dict["answer_raw"] = answer_dict["answer_raw"]
        result_dict["answer_citations"] = answer_dict["citation_ids"]
        result_dict["answer_context"] = answer_dict["context"]

    return result_dict

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

    if "answer" in result_dict:
        print("answer:")
        print(result_dict["answer"])
        print()

def write_result_json(output_path: Path, result_dict: dict) -> None:
    """
    把结果写成 JSON, 方便后续人工检查和 brute / hnsw 对比.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = json.dumps(result_dict, ensure_ascii=False, indent=2)
    output_path.write_text(output_text, encoding="utf-8")


def run_search_result(
    query: str,
    embedding_records: list[EmbeddingRecord],
    mode: str,
    top_k: int,
    with_answer: bool = False,
) -> dict:
    """
    对单条 query 执行一次完整搜索，并返回统一结果字典。
    """
    if mode == "hybrid":
        merged_ids, source_map = ts030_search_hybrid_with_sources(
            query=query,
            embedding_records=embedding_records,
            top_k=top_k,
        )
        result_records = []
        for merged_id in merged_ids:
            result_records.append(embedding_records[merged_id])
    elif mode == "hybrid_rerank":
        merged_ids, source_map = ts030_search_hybrid_rerank_with_sources(
            query=query,
            embedding_records=embedding_records,
            top_k=top_k,
        )
        result_records = []
        for merged_id in merged_ids:
            result_records.append(embedding_records[merged_id])
    else:
        source_map = None
        result_records = search(
            query=query,
            embedding_records=embedding_records,
            top_k=top_k,
            mode=mode,
        )

    if with_answer:
        answer_dict = answer_with_citations(
            query=query,
            result_records=result_records,
        )
    else:
        answer_dict = None

    return build_result_dict(
        query=query,
        mode=mode,
        top_k=top_k,
        result_records=result_records,
        source_dict_list=source_map,
        answer_dict=answer_dict,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    query = load_query_text(args)
    embedding_records = load_embedding_records(EMBEDDING_OUTPUT_PATH)
    result_dict = run_search_result(
        query=query,
        embedding_records=embedding_records,
        mode=args.mode,
        top_k=args.top_k,
        with_answer=args.with_answer,
    )

    print_results(result_dict)

    output_name = args.output_name or "search_result"
    output_path = SEARCH_RUNS_DIR / f"{output_name}_{args.mode}.json"
    write_result_json(output_path, result_dict)

    print(f"saved to: {output_path}")


if __name__ == "__main__":
    main()
