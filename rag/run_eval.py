"""
最小评测脚本:
1. 读取一份 query -> relevant_chunk_ids 的 jsonl 金标(标准答案, groundtruth)
2. 调用现有检索流程拿回 top-k 结果
3. 计算 Hit@k / Recall@k / MRR
4. 在终端打印摘要, 同时把结构化结果写入 output
备注:
MRR = (1 / N) * Σ (1 / rank_i), 其中 N 是查询总数, rank_i 是第 i 个查询的第一个正确答案出现的位置. 
MRR 越接近 1.0 越好.
"""

import argparse
import json
from pathlib import Path
import sys

RAG_ROOT = Path(__file__).resolve().parent
SRC_DIR = RAG_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from azusa_rag.config import EMBEDDING_OUTPUT_PATH, OUTPUT_DIR, PROJECT_ROOT
from azusa_rag.loaders import load_embedding_records
from azusa_rag.schemas import EmbeddingRecord
from azusa_rag.search import search


def build_parser() -> argparse.ArgumentParser:
    """
    定义评测脚本所需命令行参数.
    """
    parser = argparse.ArgumentParser(description="Run Azusa RAG min eval")
    parser.add_argument(
        "--mode",
        choices=["brute", "hnsw", "hybrid", "hybrid_rerank", "all"],
        default="brute",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--eval-path",
        default=str(PROJECT_ROOT / "data" / "eval" / "min_eval.jsonl"),
    )
    parser.add_argument("--output-name", default="min_eval_result")
    return parser


def load_eval_cases(eval_path: Path) -> list[dict]:
    """
    从 jsonl 文件读取最小评测样本.
    """
    eval_cases = []
    with eval_path.open("r", encoding="utf-8") as eval_file:
        for line in eval_file:
            line = line.strip()
            if not line:
                continue
            eval_cases.append(json.loads(line))

    if not eval_cases:
        raise ValueError(f"No eval cases found in {eval_path}")

    return eval_cases


def extract_chunk_ids(result_records: list[EmbeddingRecord]) -> list[str]:
    """
    把检索返回的 EmbeddingRecord 列表映射成 chunk_id 列表.
    """
    chunk_ids = []
    for record in result_records:
        chunk_ids.append(record.chunk_id)
    return chunk_ids


def reciprocal_rank(retrieved_chunk_ids: list[str], relevant_chunk_ids: set[str]) -> float:
    """
    返回第一个相关结果的倒数排名.
    第一个相关结果在标准答案列表中出现的位置越靠前, 分数越高
    """
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant_chunk_ids:
            return 1.0 / rank
    return 0.0


def evaluate_case(query_case: dict, retrieved_chunk_ids: list[str], top_k: int) -> dict:
    """
    计算单条 query 的最小评测指标.
    """
    relevant_chunk_ids = set(query_case["relevant_chunk_ids"]) # 这条query的标准答案chunk id集合
    retrieved_top_k = retrieved_chunk_ids[:top_k]
    matched_chunk_ids = []
    for chunk_id in retrieved_top_k:
        if chunk_id in relevant_chunk_ids:
            matched_chunk_ids.append(chunk_id)

    hit_at_k = 1.0 if matched_chunk_ids else 0.0
    recall_at_k = len(matched_chunk_ids) / len(relevant_chunk_ids)
    mrr = reciprocal_rank(retrieved_top_k, relevant_chunk_ids) # mrr是mean reciprocal rank,平均倒数排名.但是"平均"这个动作目前不在这里, 这里现在算的是"单条query的分数", 在后面的summarize_case_results中会求平均.

    return {
        "query_id": query_case["query_id"],
        "query": query_case["query"],
        "top_k": top_k,
        "relevant_chunk_ids": query_case["relevant_chunk_ids"],
        "retrieved_chunk_ids": retrieved_top_k,
        "matched_chunk_ids": matched_chunk_ids,
        "hit_at_k": hit_at_k,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "notes": query_case.get("notes"), # 有些样本可能没有备注"notes", 因此这里选用了get, 保证如果没有"notes"也不会报错.
    }


def summarize_case_results(case_results: list[dict], mode: str, top_k: int) -> dict:
    """
    汇总所有 query 的均值指标.
    """
    query_count = len(case_results)
    total_hit_at_k = 0.0
    total_recall_at_k = 0.0
    total_mrr = 0.0

    for case_result in case_results:
        total_hit_at_k += case_result["hit_at_k"]
        total_recall_at_k += case_result["recall_at_k"]
        total_mrr += case_result["mrr"]

    return {
        "mode": mode,
        "top_k": top_k,
        "query_count": query_count,
        "avg_hit_at_k": total_hit_at_k / query_count,
        "avg_recall_at_k": total_recall_at_k / query_count,
        "avg_mrr": total_mrr / query_count,
        "cases": case_results,
    }


def run_eval_for_mode(
    mode: str,
    eval_cases: list[dict],
    embedding_records: list[EmbeddingRecord],
    top_k: int,
) -> dict:
    """
    对单个 mode 执行整套最小评测.
    """
    case_results = []
    for query_case in eval_cases:
        result_records = search(
            query=query_case["query"],
            embedding_records=embedding_records,
            top_k=top_k,
            mode=mode,
        )
        retrieved_chunk_ids = extract_chunk_ids(result_records)
        case_result = evaluate_case(query_case, retrieved_chunk_ids, top_k)
        case_results.append(case_result)

    return summarize_case_results(case_results, mode, top_k)


def print_eval_summary(summary: dict) -> None:
    """
    终端打印精简版评测摘要.
    """
    print(f"mode: {summary['mode']}")
    print(f"top_k: {summary['top_k']}")
    print(f"query_count: {summary['query_count']}")
    print(f"avg_hit@k: {summary['avg_hit_at_k']:.4f}")
    print(f"avg_recall@k: {summary['avg_recall_at_k']:.4f}")
    print(f"avg_mrr: {summary['avg_mrr']:.4f}")
    print()

    for case_result in summary["cases"]:
        print(f"[{case_result['query_id']}]")
        print(f"  hit@k={case_result['hit_at_k']:.1f} recall@k={case_result['recall_at_k']:.2f} mrr={case_result['mrr']:.2f}")
        print(f"  gold={case_result['relevant_chunk_ids']}")
        print(f"  pred={case_result['retrieved_chunk_ids']}")
        print()


def write_eval_json(output_path: Path, summary_dict: dict) -> None:
    """
    把评测结果写成 JSON, 方便后续做模式对比.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = json.dumps(summary_dict, ensure_ascii=False, indent=2)
    output_path.write_text(output_text, encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    eval_path = Path(args.eval_path)
    eval_cases = load_eval_cases(eval_path)
    embedding_records = load_embedding_records(EMBEDDING_OUTPUT_PATH)

    if args.mode == "all":
        modes = ["brute", "hnsw", "hybrid", "hybrid_rerank"]
    else:
        modes = [args.mode]

    failed_modes = []

    for mode in modes:
        try:
            summary = run_eval_for_mode(
                mode=mode,
                eval_cases=eval_cases,
                embedding_records=embedding_records,
                top_k=args.top_k,
            )
        except Exception as error:
            failed_modes.append(
                {
                    "mode": mode,
                    "error": str(error),
                }
            )
            print(f"mode: {mode}")
            print(f"eval failed: {error}")
            print()
            if args.mode != "all":
                raise
            continue

        print_eval_summary(summary)

        output_path = OUTPUT_DIR / f"{args.output_name}_{mode}.json"
        write_eval_json(output_path, summary)
        print(f"saved to: {output_path}")
        print()

    if failed_modes:
        print("failed modes:")
        for failed_mode in failed_modes:
            print(f"  {failed_mode['mode']}: {failed_mode['error']}")


if __name__ == "__main__":
    main()
