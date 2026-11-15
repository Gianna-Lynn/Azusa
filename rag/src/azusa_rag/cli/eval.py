"""
最小评测脚本:
1. 读取一份 query -> relevant_chunk_ids 的 jsonl 金标(标准答案, groundtruth)
2. 调用现有检索流程拿回 top-k 结果
3. 计算 Hit@k / Recall@k / MRR / nDCG@k 和 nearest-rank P50/P95
4. 在终端打印摘要, 同时把结构化结果写入 output
备注:
MRR = (1 / N) * Σ (1 / rank_i), 其中 N 是查询总数, rank_i 是第 i 个查询的第一个正确答案出现的位置. 
MRR 越接近 1.0 越好.
"""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from collections import Counter

from azusa_rag.config import EMBEDDING_OUTPUT_PATH, EVAL_RUNS_DIR, PROJECT_ROOT
from azusa_rag.loaders import load_embedding_records
from azusa_rag.modes import SEARCH_MODES
from azusa_rag.schemas import EmbeddingRecord
from azusa_rag.search import search
from azusa_rag.evaluation_provenance import (
    experiment_identity, file_identity, new_run_directory, runtime_config,
)

EVAL_MODES = SEARCH_MODES


def build_parser() -> argparse.ArgumentParser:
    """
    定义评测脚本所需命令行参数.
    """
    parser = argparse.ArgumentParser(description="Run Azusa RAG min eval")
    parser.add_argument(
        "--mode",
        choices=EVAL_MODES + ["all"],
        default="brute",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, help="新实验输出目录；拒绝覆盖同名结果文件。")
    parser.add_argument(
        "--eval-path",
        default=str(PROJECT_ROOT / "data" / "eval" / "min_eval.jsonl"),
    )
    parser.add_argument(
        "--embedding-path",
        default=str(EMBEDDING_OUTPUT_PATH),
        help="Path to the embedding jsonl used for evaluation. Defaults to output/artifacts/embedding.jsonl.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Optional logical run label. Results use a new directory under output/eval_runs/runs/."
        ),
    )
    parser.add_argument(
        "--keep-run-copy",
        action="store_true",
        help="Also keep a named copy under output/eval_runs/ for this run.",
    )
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

    required_fields = {
        "query_id",
        "query",
        "reference_answer",
        "relevant_chunk_ids",
        "expected_citations",
        "case_type",
    }
    seen_query_ids = set()
    for index, eval_case in enumerate(eval_cases, start=1):
        if not isinstance(eval_case, dict):
            raise ValueError(f"Eval case #{index} must be an object")
        missing_fields = sorted(required_fields - eval_case.keys())
        if missing_fields:
            raise ValueError(
                f"Eval case #{index} is missing required fields: {missing_fields}"
            )
        query_id = eval_case["query_id"]
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("query_id must be a non-blank string")
        if query_id in seen_query_ids:
            raise ValueError(f"Duplicate query_id in eval set: {query_id}")
        seen_query_ids.add(query_id)
        if not isinstance(eval_case["query"], str) or not eval_case["query"].strip():
            raise ValueError(f"Eval case {query_id} has an invalid query")
        relevant = eval_case["relevant_chunk_ids"]
        if not isinstance(relevant, list) or not relevant:
            raise ValueError(f"Eval case {query_id} has no relevant_chunk_ids")
        if any(not isinstance(item, str) or not item.strip() for item in relevant) or len(set(relevant)) != len(relevant):
            raise ValueError(f"Eval case {query_id} has invalid or duplicate relevant IDs")
        citations = eval_case["expected_citations"]
        if (not isinstance(citations, list) or not citations
                or any(not isinstance(item, str) or not item.strip() for item in citations)
                or len(set(citations)) != len(citations)):
            raise ValueError(f"Eval case {query_id} has invalid expected_citations")

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


def normalized_discounted_cumulative_gain(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
    *,
    top_k: int | None = None,
) -> float:
    """计算二值相关性 nDCG，按请求 k 构造理想排序，不因少返回结果而抬高成绩。

    旧的两参数调用把列表长度当作 k；正式评测始终显式传入请求 k。
    重复 ID 属于检索协议错误，拒绝用重复证据累积收益；无相关项返回零。
    """
    if top_k is None:
        top_k = len(retrieved_chunk_ids)
    if type(top_k) is not int or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    if len(set(retrieved_chunk_ids)) != len(retrieved_chunk_ids):
        raise ValueError("retrieved chunk IDs must be unique")
    dcg = sum(1.0 / math.log2(rank + 1)
              for rank, chunk_id in enumerate(retrieved_chunk_ids[:top_k], start=1)
              if chunk_id in relevant_chunk_ids)
    ideal_count = min(top_k, len(relevant_chunk_ids))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_case(
    query_case: dict,
    retrieved_chunk_ids: list[str],
    top_k: int,
    retrieval_latency_ms: float,
) -> dict:
    """
    对一条金标问题的有序 chunk ID 列表计算检索指标，并保留排名供复核。

    top_k 必须为真正的正整数；重复命中和空相关集以 ValueError 拒绝。
    未返回的位置不会缩小 nDCG 理想分母；传入耗时仅作为观测值记录，不执行检索。
    """
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if len(set(retrieved_chunk_ids)) != len(retrieved_chunk_ids):
        raise ValueError("retrieved chunk IDs must be unique")
    relevant_chunk_ids = set(query_case["relevant_chunk_ids"]) # 这条query的标准答案chunk id集合
    if not relevant_chunk_ids:
        raise ValueError("relevant_chunk_ids must not be empty")
    retrieved_top_k = retrieved_chunk_ids[:top_k]
    matched_chunk_ids = []
    for chunk_id in retrieved_top_k:
        if chunk_id in relevant_chunk_ids:
            matched_chunk_ids.append(chunk_id)

    hit_at_k = 1.0 if matched_chunk_ids else 0.0
    recall_at_k = len(matched_chunk_ids) / len(relevant_chunk_ids)
    mrr = reciprocal_rank(retrieved_top_k, relevant_chunk_ids) # mrr是mean reciprocal rank,平均倒数排名.但是"平均"这个动作目前不在这里, 这里现在算的是"单条query的分数", 在后面的summarize_case_results中会求平均.
    ndcg = normalized_discounted_cumulative_gain(retrieved_top_k, relevant_chunk_ids, top_k=top_k)

    return {
        "query_id": query_case["query_id"],
        "query": query_case["query"],
        "reference_answer": query_case["reference_answer"],
        "expected_citations": query_case["expected_citations"],
        "case_type": query_case["case_type"],
        "top_k": top_k,
        "relevant_chunk_ids": query_case["relevant_chunk_ids"],
        "retrieved_chunk_ids": retrieved_top_k,
        "matched_chunk_ids": matched_chunk_ids,
        "hit_at_k": hit_at_k,
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "ndcg_at_k": ndcg,
        "retrieval_latency_ms": retrieval_latency_ms,
        "notes": query_case.get("notes"), # 有些样本可能没有备注"notes", 因此这里选用了get, 保证如果没有"notes"也不会报错.
    }


def percentile(values: list[float], ratio: float) -> float:
    """按 nearest-rank 计算分位数；空列表返回零，不插值放大小样本含义。"""
    if not values:
        return 0.0
    ordered_values = sorted(values)
    rank = max(1, math.ceil(len(ordered_values) * ratio))
    return ordered_values[rank - 1]


def summarize_case_results(case_results: list[dict], mode: str, top_k: int) -> dict:
    """
    汇总所有 query 的均值指标.
    """
    query_count = len(case_results)
    if not query_count:
        raise ValueError("Cannot summarize an empty evaluation")
    total_hit_at_k = 0.0
    total_recall_at_k = 0.0
    total_mrr = 0.0
    total_ndcg = 0.0
    latencies_ms = []

    for case_result in case_results:
        total_hit_at_k += case_result["hit_at_k"]
        total_recall_at_k += case_result["recall_at_k"]
        total_mrr += case_result["mrr"]
        total_ndcg += case_result["ndcg_at_k"]
        latencies_ms.append(case_result["retrieval_latency_ms"])

    return {
        "mode": mode,
        "top_k": top_k,
        "query_count": query_count,
        "avg_hit_at_k": total_hit_at_k / query_count,
        "avg_recall_at_k": total_recall_at_k / query_count,
        "avg_mrr": total_mrr / query_count,
        "avg_ndcg_at_k": total_ndcg / query_count,
        "total_retrieval_latency_ms": sum(latencies_ms),
        "avg_retrieval_latency_ms": sum(latencies_ms) / query_count,
        "p50_retrieval_latency_ms": percentile(latencies_ms, 0.50),
        "p95_retrieval_latency_ms": percentile(latencies_ms, 0.95),
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
    先确认语料身份和金标，再独立计时预热及逐题检索；每题保存 trace 和实际重排后端。
    """
    if mode not in EVAL_MODES or type(top_k) is not int or top_k < 1:
        raise ValueError("Unknown mode or invalid top_k")
    if not eval_cases:
        raise ValueError("Evaluation cases must not be empty")
    chunk_ids = [record.chunk_id for record in embedding_records]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Corpus chunk IDs must be unique")
    known_ids = set(chunk_ids)
    for case in eval_cases:
        missing = set(case["relevant_chunk_ids"]) - known_ids
        if missing:
            raise ValueError(f"Gold IDs missing from corpus for {case['query_id']}: {sorted(missing)}")
    # 单格预热显式独立计时，不混入逐题延迟；记录其后端，避免隐藏冷态回退。
    warmup_trace = {}
    warmup_started = perf_counter()
    search(
        query=eval_cases[0]["query"],
        embedding_records=embedding_records,
        top_k=top_k,
        mode=mode,
        trace=warmup_trace,
    )
    warmup_ms = (perf_counter() - warmup_started) * 1000.0

    case_results = []
    for query_case in eval_cases:
        trace = {}
        retrieval_started_at = perf_counter()
        result_records = search(
            query=query_case["query"],
            embedding_records=embedding_records,
            top_k=top_k,
            mode=mode,
            trace=trace,
        )
        retrieval_latency_ms = (perf_counter() - retrieval_started_at) * 1000.0
        retrieved_chunk_ids = extract_chunk_ids(result_records)
        if not set(retrieved_chunk_ids) <= known_ids:
            raise ValueError("Retrieved evidence is outside the evaluated corpus")
        case_result = evaluate_case(
            query_case,
            retrieved_chunk_ids,
            top_k,
            retrieval_latency_ms,
        )
        case_result["trace"] = trace
        case_results.append(case_result)

    summary = summarize_case_results(case_results, mode, top_k)
    summary.update(
        warmup_latency_ms=warmup_ms,
        warmup_trace=warmup_trace,
        runtime_config=runtime_config(),
        rerank_backends=dict(Counter(item["trace"].get("rerank_backend", "not_used") for item in case_results)),
    )
    return summary


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
    print(f"avg_ndcg@k: {summary['avg_ndcg_at_k']:.4f}")
    print(f"avg_retrieval_latency_ms: {summary['avg_retrieval_latency_ms']:.3f}")
    print(f"p95_retrieval_latency_ms: {summary['p95_retrieval_latency_ms']:.3f}")
    print()

    for case_result in summary["cases"]:
        print(f"[{case_result['query_id']}]")
        print(f"  hit@k={case_result['hit_at_k']:.1f} recall@k={case_result['recall_at_k']:.2f} mrr={case_result['mrr']:.2f}")
        print(f"  gold={case_result['relevant_chunk_ids']}")
        print(f"  pred={case_result['retrieved_chunk_ids']}")
        print()


def write_eval_json(output_path: Path, summary_dict: dict) -> None:
    """
    独占创建 JSON 文件；拒绝覆盖旧实验，序列化失败不生成看似完整的结果。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = json.dumps(summary_dict, ensure_ascii=False, indent=2, allow_nan=False)
    with output_path.open("x", encoding="utf-8") as output:
        output.write(output_text)


def write_eval_outputs(
    mode: str,
    summary_dict: dict,
    output_name: str | None,
    keep_run_copy: bool,
    *,
    output_dir: Path | None = None,
) -> tuple[Path, Path | None]:
    """
    默认使用唯一实验目录；保留旧调用签名，但不再隐式写入 latest。
    返回主结果路径与可选命名副本路径；文件存在时传播 FileExistsError，
    不覆盖或删除旧产物。多文件写入不构成事务，失败时必须检查实际已落盘文件。
    """
    output_dir = output_dir or new_run_directory(EVAL_RUNS_DIR / "runs", output_name or "eval")
    result_path = output_dir / f"eval_result_{mode}.json"
    write_eval_json(result_path, summary_dict)

    run_copy_path = None
    if keep_run_copy and output_name:
        safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in output_name)
        run_copy_path = output_dir / f"{safe_name}_{mode}.json"
        write_eval_json(run_copy_path, summary_dict)

    return result_path, run_copy_path


def main(argv: list[str] | None = None) -> None:
    """执行所有指定模式，保留失败清单；任何失败均以非零状态结束。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    eval_path = Path(args.eval_path)
    inputs = {"embedding": file_identity(Path(args.embedding_path)), "eval": file_identity(eval_path)}
    eval_cases = load_eval_cases(eval_path)
    # 这里允许评测直接指向一份单独的 embedding 产物，避免覆盖默认主链路数据。
    embedding_records = load_embedding_records(Path(args.embedding_path))
    if inputs != {"embedding": file_identity(Path(args.embedding_path)), "eval": file_identity(eval_path)}:
        raise ValueError("Evaluation inputs changed while loading")
    provenance = experiment_identity()
    output_dir = args.output_dir or new_run_directory(EVAL_RUNS_DIR / "runs", args.output_name or "eval")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {output_dir}")

    if args.mode == "all":
        modes = EVAL_MODES
    else:
        modes = [args.mode]

    failed_modes = []
    completed = []

    for mode in modes:
        try:
            summary = run_eval_for_mode(
                mode=mode,
                eval_cases=eval_cases,
                embedding_records=embedding_records,
                top_k=args.top_k,
            )
            if inputs != {"embedding": file_identity(Path(args.embedding_path)), "eval": file_identity(eval_path)}:
                raise ValueError("Evaluation inputs changed during execution")
            summary["provenance"] = {**provenance, "inputs": inputs}
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
            continue

        print_eval_summary(summary)

        result_path, run_copy_path = write_eval_outputs(
            mode=mode,
            summary_dict=summary,
            output_name=args.output_name,
            keep_run_copy=args.keep_run_copy,
            output_dir=output_dir,
        )
        completed.append({"mode": mode, "path": str(result_path)})
        print(f"saved to: {result_path}")
        if run_copy_path is not None:
            print(f"named copy: {run_copy_path}")
        print()

    write_eval_json(output_dir / "evaluation.json", {
        "schema_version": 2, "provenance": provenance, "inputs": inputs,
        "completed": completed, "failures": failed_modes,
    })
    if failed_modes:
        print("failed modes:")
        for failed_mode in failed_modes:
            print(f"  {failed_mode['mode']}: {failed_mode['error']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
