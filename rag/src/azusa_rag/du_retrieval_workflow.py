"""
把 DuRetrieval 小样本流程收成一条可重复执行的 workflow：
1. 可选下载本地 sample 快照
2. 从本地 sample 生成 passages / eval
3. 生成 embedding
4. 跑 bm25 和 hybrid_rerank 两种评测
5. 额外产出一条 query 的检索对比 + answer/citation 示例

设计原则:
- 尽量复用现有 importer / embed / eval / search 核心逻辑
- 结果优先写到 data/processed，避开当前环境里 output/eval_runs 的偶发写入问题
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from urllib.error import URLError

from .config import (
    DU_RETRIEVAL_LATEST_DIR,
    DU_RETRIEVAL_RUNS_DIR,
    HEALTH_URL,
    PROJECT_ROOT,
)
from .du_retrieval_download import (
    DATASET_DU_RETRIEVAL,
    DATASET_DU_RETRIEVAL_QRELS,
    build_local_sample_snapshot,
    fetch_dataset_rows,
    write_snapshot,
)
from .du_retrieval_import import (
    build_eval_records,
    build_passage_records,
    limit_du_retrieval_rows,
    write_jsonl,
)
from .embed_main import build_embedding_artifacts
from .loaders import load_embedding_records
from .search_hnsw import get_json

# 这里显式把 rag 项目根目录补进 sys.path，
# 是为了兼容当前仍然存在于项目根目录下的 run_eval.py / run_search.py。
# 这样 workflow 既能通过 run_du_retrieval_workflow.py 运行，
# 也不容易被“只配置了 src 到 PYTHONPATH”这件事绊住。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from run_eval import load_eval_cases, run_eval_for_mode
from run_search import run_search_result


def build_parser() -> argparse.ArgumentParser:
    """
    定义 DuRetrieval sample workflow 的命令行参数。
    """
    processed_dir = PROJECT_ROOT / "data" / "processed"
    parser = argparse.ArgumentParser(description="Run the DuRetrieval sample workflow")
    parser.add_argument(
        "--sample-json-path",
        default=str(processed_dir / "du_retrieval_sample_small.json"),
        help="Local DuRetrieval sample snapshot path.",
    )
    parser.add_argument(
        "--download-sample",
        action="store_true",
        help="Re-download the local sample snapshot before running the workflow.",
    )
    parser.add_argument("--query-limit", type=int, default=5)
    parser.add_argument("--corpus-limit", type=int, default=200)
    parser.add_argument("--qrels-scan-length", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output-prefix",
        default="du_retrieval_sample_small",
        help=(
            "Logical workflow name. By default generated files are overwritten "
            "under data/processed/du_retrieval_workflow/latest/."
        ),
    )
    parser.add_argument(
        "--keep-run-copy",
        action="store_true",
        help="Write this run under data/processed/du_retrieval_workflow/runs/<output-prefix>/.",
    )
    parser.add_argument(
        "--compare-query",
        default=None,
        help="Optional explicit query used for side-by-side search comparison.",
    )
    return parser


def download_sample_snapshot(
    sample_json_path: Path,
    query_limit: int,
    corpus_limit: int,
    qrels_scan_length: int,
) -> dict:
    """
    从 Hugging Face 拉取一份 DuRetrieval 本地 sample 快照。
    """
    corpus_rows = fetch_dataset_rows(
        dataset=DATASET_DU_RETRIEVAL,
        split="corpus",
        offset=0,
        length=corpus_limit,
    )
    query_rows = fetch_dataset_rows(
        dataset=DATASET_DU_RETRIEVAL,
        split="queries",
        offset=0,
        length=query_limit,
    )
    qrel_rows = fetch_dataset_rows(
        dataset=DATASET_DU_RETRIEVAL_QRELS,
        split="dev",
        offset=0,
        length=qrels_scan_length,
    )
    snapshot = build_local_sample_snapshot(
        corpus_rows=corpus_rows,
        query_rows=query_rows,
        qrel_rows=qrel_rows,
    )
    write_snapshot(sample_json_path, snapshot)
    return {
        "sample_json_path": str(sample_json_path),
        "corpus_rows": len(snapshot["corpus"]),
        "query_rows": len(snapshot["queries"]),
        "qrel_rows": len(snapshot["qrels"]),
    }


def import_local_sample(
    sample_json_path: Path,
    passages_output_path: Path,
    eval_output_path: Path,
    notes_text: str,
    corpus_limit: int | None,
    query_limit: int | None,
) -> dict:
    """
    把本地 sample 快照转成 passages / eval 两份下游产物。
    """
    data = json.loads(sample_json_path.read_text(encoding="utf-8"))
    corpus, queries, qrels = limit_du_retrieval_rows(
        corpus=data["corpus"],
        queries=data["queries"],
        qrels=data["qrels"],
        corpus_limit=corpus_limit,
        query_limit=query_limit,
    )
    eval_records = build_eval_records(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        notes_text=notes_text,
    )
    passage_records = build_passage_records(corpus)
    write_jsonl(eval_output_path, eval_records)
    write_jsonl(passages_output_path, passage_records)
    return {
        "passages_output_path": str(passages_output_path),
        "eval_output_path": str(eval_output_path),
        "passage_count": len(passage_records),
        "eval_case_count": len(eval_records),
    }


def build_report_markdown(summary: dict) -> str:
    """
    生成一份面向人工扫读的 Markdown 汇总。
    """
    lines = []
    lines.append("# DuRetrieval Sample Workflow")
    lines.append("")
    lines.append(f"- generated_at: `{summary['generated_at']}`")
    lines.append(f"- workflow_name: `{summary['workflow_name']}`")
    lines.append(f"- sample_json: `{summary['paths']['sample_json_path']}`")
    lines.append(f"- passages: `{summary['paths']['passages_output_path']}`")
    lines.append(f"- embedding: `{summary['paths']['embedding_output_path']}`")
    lines.append(f"- eval: `{summary['paths']['eval_output_path']}`")
    lines.append(f"- top_k: `{summary['top_k']}`")
    lines.append("")

    lines.append("## Steps")
    lines.append("")
    for step_name, step_result in summary["steps"].items():
        lines.append(
            f"- `{step_name}`: status=`{step_result['status']}`"
        )
    lines.append("")

    lines.append("## Eval Metrics")
    lines.append("")
    lines.append("| mode | hit@k | recall@k | mrr | nDCG@k | p50 ms | p95 ms | status |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for mode_item in summary["eval_modes"]:
        if mode_item["status"] == "completed":
            metrics = mode_item["metrics"]
            lines.append(
                "| "
                + mode_item["mode"]
                + " | "
                + f"{metrics['avg_hit_at_k']:.4f}"
                + " | "
                + f"{metrics['avg_recall_at_k']:.4f}"
                + " | "
                + f"{metrics['avg_mrr']:.4f}"
                + " | " + f"{metrics['avg_ndcg_at_k']:.4f}"
                + " | " + f"{metrics['p50_retrieval_latency_ms']:.3f}"
                + " | " + f"{metrics['p95_retrieval_latency_ms']:.3f}"
                + " | completed |"
            )
        else:
            lines.append("| " + mode_item["mode"] + " | - | - | - | - | - | - | failed |")
    lines.append("")

    lines.append("## Metric Ranking")
    lines.append("")
    ranking = summary.get("mode_ranking_by_mrr", [])
    if not ranking:
        lines.append("- 没有成功完成任何 mode 的评测。")
    else:
        lines.append("| rank | mode | hit@k | recall@k | mrr |")
        lines.append("| --- | --- | --- | --- | --- |")
        for index, mode_item in enumerate(ranking, start=1):
            lines.append(
                "| "
                + str(index)
                + " | "
                + mode_item["mode"]
                + " | "
                + f"{mode_item['avg_hit_at_k']:.4f}"
                + " | "
                + f"{mode_item['avg_recall_at_k']:.4f}"
                + " | "
                + f"{mode_item['avg_mrr']:.4f}"
                + " |"
            )
    lines.append("")

    lines.append("## Delta Vs Bm25")
    lines.append("")
    delta_vs_bm25 = summary.get("delta_vs_bm25", {})
    if not delta_vs_bm25:
        lines.append("- 当前没有可用的 baseline 对比。")
    else:
        lines.append("| mode | hit delta | recall delta | mrr delta |")
        lines.append("| --- | --- | --- | --- |")
        for mode_name in ["bm25", "hybrid_rerank"]:
            delta_item = delta_vs_bm25.get(mode_name)
            if delta_item is None:
                continue
            lines.append(
                "| "
                + mode_name
                + " | "
                + f"{delta_item['avg_hit_at_k_delta']:.4f}"
                + " | "
                + f"{delta_item['avg_recall_at_k_delta']:.4f}"
                + " | "
                + f"{delta_item['avg_mrr_delta']:.4f}"
                + " |"
            )
    lines.append("")

    lines.append("## Compare Query")
    lines.append("")
    lines.append(f"- query: {summary['compare_query']}")
    lines.append("")
    for mode_name, result_path in summary["paths"]["search_result_paths"].items():
        lines.append(f"### {mode_name}")
        lines.append("")
        lines.append(f"- result_path: `{result_path}`")
        result_dict = summary["search_results"].get(mode_name)
        if result_dict is None:
            lines.append("- result unavailable")
            lines.append("")
            continue
        for item in result_dict["results"]:
            lines.append(f"- rank {item['rank']}: {item['chunk_id']}")
        if "answer" in result_dict:
            lines.append(f"- answer: {result_dict['answer']}")
            lines.append(f"- citations: {result_dict.get('answer_citations', [])}")
        lines.append("")

    return "\n".join(lines)


def build_mode_ranking(eval_summaries: dict[str, dict]) -> list[dict]:
    """
    把完成的 mode 指标压成一个可排序的小表，方便报告先看总体胜负。
    """
    ranking_items = []
    for mode_name, summary in eval_summaries.items():
        ranking_items.append(
            {
                "mode": mode_name,
                "avg_hit_at_k": summary["avg_hit_at_k"],
                "avg_recall_at_k": summary["avg_recall_at_k"],
                "avg_mrr": summary["avg_mrr"],
            }
        )
    ranking_items.sort(
        key=lambda item: (
            -item["avg_mrr"],
            -item["avg_recall_at_k"],
            -item["avg_hit_at_k"],
            item["mode"],
        )
    )
    return ranking_items


def build_delta_vs_bm25(eval_summaries: dict[str, dict]) -> dict[str, dict]:
    """
    统一把各个 mode 相对 bm25 的增量算出来，便于快速判断 rerank 有没有拉开。
    """
    if "bm25" not in eval_summaries:
        return {}

    baseline = eval_summaries["bm25"]
    delta_dict = {}
    for mode_name, summary in eval_summaries.items():
        delta_dict[mode_name] = {
            "avg_hit_at_k_delta": summary["avg_hit_at_k"] - baseline["avg_hit_at_k"],
            "avg_recall_at_k_delta": summary["avg_recall_at_k"] - baseline["avg_recall_at_k"],
            "avg_mrr_delta": summary["avg_mrr"] - baseline["avg_mrr"],
        }
    return delta_dict


def print_terminal_summary(summary: dict, summary_path: Path, report_path: Path) -> None:
    """
    在终端打印一个短摘要，减少每次都要手翻 Markdown 的负担。
    """
    print(f"workflow_name: {summary['workflow_name']}")
    print(f"summary_json: {summary_path}")
    print(f"report_md: {report_path}")

    ranking = summary.get("mode_ranking_by_mrr", [])
    if ranking:
        best_mode = ranking[0]
        print("best_mode_by_mrr:")
        print(
            f"  mode={best_mode['mode']} "
            f"hit@k={best_mode['avg_hit_at_k']:.4f} "
            f"recall@k={best_mode['avg_recall_at_k']:.4f} "
            f"mrr={best_mode['avg_mrr']:.4f}"
        )

    delta_item = summary.get("delta_vs_bm25", {}).get("hybrid_rerank")
    if delta_item is not None:
        print("hybrid_rerank_vs_bm25:")
        print(
            f"  hit_delta={delta_item['avg_hit_at_k_delta']:.4f} "
            f"recall_delta={delta_item['avg_recall_at_k_delta']:.4f} "
            f"mrr_delta={delta_item['avg_mrr_delta']:.4f}"
        )

    hybrid_result = summary.get("search_results", {}).get("hybrid_rerank", {})
    if hybrid_result.get("answer"):
        print("sample_answer:")
        print(f"  {hybrid_result['answer']}")


def main() -> None:
    args = build_parser().parse_args()

    output_dir = (
        DU_RETRIEVAL_RUNS_DIR / args.output_prefix
        if args.keep_run_copy
        else DU_RETRIEVAL_LATEST_DIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_json_path = Path(args.sample_json_path)
    passages_output_path = output_dir / "passages.jsonl"
    eval_output_path = output_dir / "eval.jsonl"
    embedding_output_path = output_dir / "embedding.jsonl"
    bm25_eval_path = output_dir / "bm25.json"
    hybrid_eval_path = output_dir / "hybrid_rerank.json"
    bm25_search_path = output_dir / "search_bm25.json"
    hybrid_search_path = output_dir / "search_hybrid_rerank_answer.json"
    summary_path = output_dir / "workflow_summary.json"
    report_path = output_dir / "workflow_report.md"

    steps = {
        "download_sample": {"status": "skipped"},
        "import_local_sample": {"status": "pending"},
        "build_embedding": {"status": "pending"},
        "run_eval_modes": {"status": "pending"},
        "run_compare_search": {"status": "pending"},
    }

    download_result = None
    if args.download_sample or not sample_json_path.exists():
        # 这里允许 workflow 自己把 sample 快照拉到本地，避免每次手动切步骤。
        download_result = download_sample_snapshot(
            sample_json_path=sample_json_path,
            query_limit=args.query_limit,
            corpus_limit=args.corpus_limit,
            qrels_scan_length=args.qrels_scan_length,
        )
        steps["download_sample"]["status"] = "completed"
    else:
        steps["download_sample"]["status"] = "reused_local_snapshot"

    import_result = import_local_sample(
        sample_json_path=sample_json_path,
        passages_output_path=passages_output_path,
        eval_output_path=eval_output_path,
        notes_text="dataset=C-MTEB/DuRetrieval; stage=workflow_sample",
        corpus_limit=args.corpus_limit,
        query_limit=args.query_limit,
    )
    steps["import_local_sample"]["status"] = "completed"

    build_embedding_artifacts(
        chunk_input_path=passages_output_path,
        embedding_output_path=embedding_output_path,
    )
    steps["build_embedding"]["status"] = "completed"

    eval_cases = load_eval_cases(eval_output_path)
    embedding_records = load_embedding_records(embedding_output_path)

    eval_modes = []
    eval_summaries = {}
    for mode in ["bm25", "hybrid_rerank"]:
        try:
            summary = run_eval_for_mode(
                mode=mode,
                eval_cases=eval_cases,
                embedding_records=embedding_records,
                top_k=args.top_k,
            )
            eval_summaries[mode] = summary
            eval_modes.append(
                {
                    "mode": mode,
                    "status": "completed",
                    "metrics": {
                        "avg_hit_at_k": summary["avg_hit_at_k"],
                        "avg_recall_at_k": summary["avg_recall_at_k"],
                        "avg_mrr": summary["avg_mrr"],
                        "avg_ndcg_at_k": summary["avg_ndcg_at_k"],
                        "p50_retrieval_latency_ms": summary["p50_retrieval_latency_ms"],
                        "p95_retrieval_latency_ms": summary["p95_retrieval_latency_ms"],
                    },
                }
            )
        except Exception as error:
            eval_modes.append(
                {
                    "mode": mode,
                    "status": "failed",
                    "error": str(error),
                }
            )
    steps["run_eval_modes"]["status"] = "completed"

    # 这里直接把 summary 手工写进 data/processed，避开当前环境里 output/eval_runs 的写入问题。
    bm25_eval_path.write_text(
        json.dumps(eval_summaries.get("bm25"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    hybrid_eval_path.write_text(
        json.dumps(eval_summaries.get("hybrid_rerank"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    compare_query = args.compare_query or eval_cases[0]["query"]
    bm25_result = run_search_result(
        query=compare_query,
        embedding_records=embedding_records,
        mode="bm25",
        top_k=args.top_k,
        with_answer=False,
    )
    hybrid_result = run_search_result(
        query=compare_query,
        embedding_records=embedding_records,
        mode="hybrid_rerank",
        top_k=args.top_k,
        with_answer=True,
    )
    steps["run_compare_search"]["status"] = "completed"
    bm25_search_path.write_text(json.dumps(bm25_result, ensure_ascii=False, indent=2), encoding="utf-8")
    hybrid_search_path.write_text(json.dumps(hybrid_result, ensure_ascii=False, indent=2), encoding="utf-8")

    server_status: dict | None
    try:
        server_status = get_json(HEALTH_URL)
    except URLError:
        server_status = None

    mode_ranking_by_mrr = build_mode_ranking(eval_summaries)
    delta_vs_bm25 = build_delta_vs_bm25(eval_summaries)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "workflow_name": args.output_prefix,
        "top_k": args.top_k,
        "steps": steps,
        "download": download_result,
        "import": import_result,
        "server_status": server_status,
        "compare_query": compare_query,
        "eval_modes": eval_modes,
        "mode_ranking_by_mrr": mode_ranking_by_mrr,
        "delta_vs_bm25": delta_vs_bm25,
        "paths": {
            "sample_json_path": str(sample_json_path),
            "passages_output_path": str(passages_output_path),
            "eval_output_path": str(eval_output_path),
            "embedding_output_path": str(embedding_output_path),
            "bm25_eval_path": str(bm25_eval_path),
            "hybrid_rerank_eval_path": str(hybrid_eval_path),
            "search_result_paths": {
                "bm25": str(bm25_search_path),
                "hybrid_rerank": str(hybrid_search_path),
            },
        },
        "search_results": {
            "bm25": {
                "query": bm25_result["query"],
                "results": bm25_result["results"],
            },
            "hybrid_rerank": {
                "query": hybrid_result["query"],
                "results": hybrid_result["results"],
                "answer": hybrid_result.get("answer"),
                "answer_citations": hybrid_result.get("answer_citations", []),
            },
        },
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(build_report_markdown(summary), encoding="utf-8")

    print_terminal_summary(summary, summary_path, report_path)
    print(f"bm25_json: {bm25_eval_path}")
    print(f"hybrid_rerank_json: {hybrid_eval_path}")


if __name__ == "__main__":
    main()
