"""
一键执行 README 中常用的本地验证工作流，并生成收敛后的总结产物。

目标：
1. 尽量复用现有 run_main / run_embed / run_eval / run_search 的核心逻辑
2. 减少人工逐个命令、逐个 JSON 对比的负担
3. 只额外生成两份核心产物：
   - workflow_summary.json
   - workflow_report.md
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

RAG_ROOT = Path(__file__).resolve().parent
SRC_DIR = RAG_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from azusa_rag.config import (
    EMBEDDING_OUTPUT_PATH,
    EVAL_RUNS_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
)
from azusa_rag.embed_main import main as embed_main
from azusa_rag.hf_dataset_import import import_hf_squad_sample
from azusa_rag.loaders import load_embedding_records
from azusa_rag.main import main as chunk_main
from run_eval import EVAL_MODES, load_eval_cases, run_eval_for_mode
from run_search import run_search_result


MODE_METADATA = {
    "brute": {
        "family": "dense-only",
        "label": "Brute Force Dense",
        "notes": "直接用向量相似度暴力搜索, 作为最朴素的 dense baseline.",
    },
    "hnsw": {
        "family": "dense-only",
        "label": "HNSW Dense",
        "notes": "通过 Rust HNSW 服务执行近似向量检索, 用来观察 ANN 近似检索是否保持效果.",
    },
    "bm25": {
        "family": "sparse-only",
        "label": "BM25 Sparse",
        "notes": "只依赖词面匹配的 BM25 baseline, 适合观察专有名词、年份和字面命中收益.",
    },
    "hybrid": {
        "family": "dense+sparse",
        "label": "Hybrid",
        "notes": "先召回 dense 和 BM25 sparse, 再按当前 merge 策略合并去重.",
    },
    "hybrid_rerank": {
        "family": "dense+sparse+rerank",
        "label": "Hybrid Rerank",
        "notes": "在 hybrid 候选集基础上再做 rerank, 用来观察排序纠偏收益.",
    },
}

MODE_DISPLAY_ORDER = ["brute", "hnsw", "bm25", "hybrid", "hybrid_rerank"]
MODE_ORDER_INDEX = {mode: index for index, mode in enumerate(MODE_DISPLAY_ORDER)}


DEFAULT_ANSWER_CHECKS = [
    {
        "case_id": "massachusetts",
        "query": "In what year did Massachusetts first require children to be educated in schools?",
        "notes": "英文事实题，适合观察 hybrid_rerank 的 top-k 和 answer 是否对齐。",
    },
    {
        "case_id": "jr_tokyo_metro",
        "query": "JR常磐缓行线和东京地铁是什么关系？",
        "notes": "中文证据题，适合观察 answer 是否引用真实 chunk。",
    },
    {
        "case_id": "insufficient",
        "query": "这份材料有没有明确提到哈佛大学校长的名字？",
        "notes": "证据不足题，适合观察 answerer 是否保守。",
    },
]


def build_parser() -> argparse.ArgumentParser:
    """
    定义 workflow 入口脚本的命令行参数。

    这里统一收纳三类开关：
    1. 评测本身需要的参数，比如 top-k、eval-path、modes
    2. answer spot check 相关参数
    3. 可选的数据准备步骤，比如重建 artifacts、导入 HF eval
    """
    parser = argparse.ArgumentParser(description="Run the consolidated Azusa RAG workflow")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--eval-path",
        default=str(PROJECT_ROOT / "data" / "eval" / "min_eval.jsonl"),
    )
    parser.add_argument(
        "--output-name",
        default="workflow",
        help="Basename of the consolidated workflow outputs under output/eval_runs/.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=EVAL_MODES,
        default=EVAL_MODES,
        help="Retrieval modes to compare inside the workflow report.",
    )
    parser.add_argument(
        "--answer-mode",
        choices=EVAL_MODES,
        default="hybrid_rerank",
        help="Which mode to use for the lightweight answer + citation spot checks.",
    )
    parser.add_argument(
        "--skip-answer-checks",
        action="store_true",
        help="Skip the representative answer + citation verification section.",
    )
    parser.add_argument(
        "--rebuild-artifacts",
        action="store_true",
        help="Re-run run_main.py and run_embed.py before evaluation.",
    )
    parser.add_argument(
        "--import-hf-eval",
        action="store_true",
        help="Import a small Hugging Face QA sample before evaluation.",
    )
    parser.add_argument("--dataset", default="autoevaluate/squad-sample")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--query-limit", type=int, default=3)
    return parser


def run_optional_steps(args: argparse.Namespace) -> dict:
    """
    执行可选的数据准备步骤，并记录状态。
    """
    step_summary = {
        "rebuild_artifacts": {
            "requested": args.rebuild_artifacts,
            "status": "skipped",
        },
        "import_hf_eval": {
            "requested": args.import_hf_eval,
            "status": "skipped",
        },
    }

    if args.rebuild_artifacts:
        # 这一步等价于 README 里的:
        # python run_main.py
        # python run_embed.py
        chunk_main()
        embed_main()
        step_summary["rebuild_artifacts"]["status"] = "completed"
        step_summary["rebuild_artifacts"]["chunk_output_path"] = str(
            EMBEDDING_OUTPUT_PATH.parent / "chunks.jsonl"
        )
        step_summary["rebuild_artifacts"]["embedding_output_path"] = str(
            EMBEDDING_OUTPUT_PATH
        )

    if args.import_hf_eval:
        # 这一步会把 HF QA 样本写进 data/raw 和 data/eval，
        # 供后面的检索评测和 answer 检查直接复用。
        import_result = import_hf_squad_sample(
            dataset=args.dataset,
            config=args.config,
            split=args.split,
            offset=args.offset,
            length=args.length,
            raw_dir=RAW_DATA_DIR,
            eval_path=Path(PROJECT_ROOT / "data" / "eval" / "hf_squad_sample_eval.jsonl"),
            query_limit=args.query_limit,
        )
        step_summary["import_hf_eval"]["status"] = "completed"
        step_summary["import_hf_eval"]["result"] = import_result

    return step_summary


def get_mode_order_key(mode: str) -> tuple[int, str]:
    """
    返回 mode 的稳定展示顺序.
    未登记的 mode 仍然能安全排到最后.
    """
    return (MODE_ORDER_INDEX.get(mode, len(MODE_DISPLAY_ORDER)), mode)


def build_mode_catalog(modes: list[str]) -> list[dict]:
    """
    为 workflow 报告生成一份 mode 说明表.
    """
    catalog = []
    for mode in modes:
        metadata = MODE_METADATA.get(
            mode,
            {
                "family": "unknown",
                "label": mode,
                "notes": "No mode description available.",
            },
        )
        catalog.append(
            {
                "mode": mode,
                "family": metadata["family"],
                "label": metadata["label"],
                "notes": metadata["notes"],
            }
        )
    catalog.sort(key=lambda item: get_mode_order_key(item["mode"]))
    return catalog


def build_mode_ranking(mode_summaries: dict[str, dict]) -> list[dict]:
    """
    把各个 mode 的总体指标压成一个可排序列表。

    排序优先级：
    1. avg_mrr
    2. avg_recall_at_k
    3. avg_hit_at_k
    4. mode 名称（仅作稳定排序的兜底）
    """
    ranking_items = []
    for mode, summary in mode_summaries.items():
        ranking_items.append(
            {
                "mode": mode,
                "avg_hit_at_k": summary["avg_hit_at_k"],
                "avg_recall_at_k": summary["avg_recall_at_k"],
                "avg_mrr": summary["avg_mrr"],
            }
        )

    # 先看 MRR，再看 recall 和 hit，这样更贴近“正确答案排得更靠前”这个目标。
    ranking_items.sort(
        key=lambda item: (
            -item["avg_mrr"],
            -item["avg_recall_at_k"],
            -item["avg_hit_at_k"],
            get_mode_order_key(item["mode"]),
        )
    )
    return ranking_items


def build_delta_vs_baseline(mode_summaries: dict[str, dict], baseline_mode: str) -> dict[str, dict]:
    """
    计算每个 mode 相对 baseline 的指标增量。

    这样后面的 summary / report 可以直接回答：
    “某个方案比 brute 到底高了多少，而不是只给绝对值。”
    """
    baseline = mode_summaries[baseline_mode]
    delta_dict = {}

    for mode, summary in mode_summaries.items():
        delta_dict[mode] = {
            "avg_hit_at_k_delta": summary["avg_hit_at_k"] - baseline["avg_hit_at_k"],
            "avg_recall_at_k_delta": summary["avg_recall_at_k"] - baseline["avg_recall_at_k"],
            "avg_mrr_delta": summary["avg_mrr"] - baseline["avg_mrr"],
        }

    return delta_dict


def build_compact_mode_summaries(mode_summaries: dict[str, dict]) -> dict[str, dict]:
    """
    把完整 mode summaries 压成更适合放进 workflow_summary.json 的精简版本。

    这里故意不保留每个 mode 下全量的 cases 明细，避免 summary.json
    重新膨胀成一个不适合人看的大文件。
    """
    compact_summaries = {}

    for mode, summary in mode_summaries.items():
        compact_summaries[mode] = {
            "mode": summary["mode"],
            "top_k": summary["top_k"],
            "query_count": summary["query_count"],
            "avg_hit_at_k": summary["avg_hit_at_k"],
            "avg_recall_at_k": summary["avg_recall_at_k"],
            "avg_mrr": summary["avg_mrr"],
        }

    return compact_summaries


def build_case_comparisons(mode_summaries: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """
    汇总跨 mode 的单题差异，便于快速定位“谁拉开了差距”。
    """
    per_query = {}
    for mode, summary in mode_summaries.items():
        for case_result in summary["cases"]:
            query_id = case_result["query_id"]
            if query_id not in per_query:
                # 先按 query_id 聚合，把不同 mode 的单题结果挂到同一个 query 下面。
                per_query[query_id] = {
                    "query_id": query_id,
                    "query": case_result["query"],
                    "per_mode": {},
                }
            per_query[query_id]["per_mode"][mode] = case_result

    disagreement_cases = []
    all_fail_cases = []

    for query_case in per_query.values():
        hit_values = []
        recall_values = []
        mrr_values = []
        per_mode_summary = {}

        for mode, case_result in query_case["per_mode"].items():
            hit_values.append(case_result["hit_at_k"])
            recall_values.append(case_result["recall_at_k"])
            mrr_values.append(case_result["mrr"])
            per_mode_summary[mode] = {
                "hit_at_k": case_result["hit_at_k"],
                "recall_at_k": case_result["recall_at_k"],
                "mrr": case_result["mrr"],
                "matched_chunk_ids": case_result["matched_chunk_ids"],
                "retrieved_chunk_ids": case_result["retrieved_chunk_ids"],
            }

        hit_spread = max(hit_values) - min(hit_values)
        recall_spread = max(recall_values) - min(recall_values)
        mrr_spread = max(mrr_values) - min(mrr_values)

        # spread 越大，说明这条 query 越能拉开不同方案的差距，
        # 后面更值得优先拿出来人工观察。
        comparison_item = {
            "query_id": query_case["query_id"],
            "query": query_case["query"],
            "hit_spread": hit_spread,
            "recall_spread": recall_spread,
            "mrr_spread": mrr_spread,
            "per_mode": per_mode_summary,
        }

        if max(hit_values) == 0.0:
            # 所有 mode 都 hit=0，说明这是“大家一起失败”的 case。
            all_fail_cases.append(comparison_item)

        if hit_spread > 0.0 or recall_spread > 0.0 or mrr_spread > 0.0:
            disagreement_cases.append(comparison_item)

    disagreement_cases.sort(
        key=lambda item: (
            -item["mrr_spread"],
            -item["recall_spread"],
            -item["hit_spread"],
            item["query_id"],
        )
    )
    return disagreement_cases, all_fail_cases


def run_answer_checks(
    embedding_records,
    top_k: int,
    answer_mode: str,
) -> list[dict]:
    """
    用少量代表性 query 验证 answer + citation 是否可工作。

    这里不追求大规模答案层评测，只挑几条最能说明问题的 spot checks：
    - 英文事实题
    - 中文证据题
    - 证据不足题
    """
    answer_results = []
    for answer_case in DEFAULT_ANSWER_CHECKS:
        result_dict = run_search_result(
            query=answer_case["query"],
            embedding_records=embedding_records,
            mode=answer_mode,
            top_k=top_k,
            with_answer=True,
        )
        answer_results.append(
            {
                "case_id": answer_case["case_id"],
                "query": answer_case["query"],
                "notes": answer_case["notes"],
                "result": result_dict,
            }
        )
    return answer_results


def build_compact_answer_checks(answer_checks: list[dict]) -> list[dict]:
    """
    只保留 answer spot checks 中最值得快速扫读的字段。

    目标是让 summary.json 里还能看到：
    - 问了什么
    - 回答了什么
    - 引了哪些 chunk
    - top1 是谁

    但不再把完整 results、full_text、answer_context 全部塞进去。
    """
    compact_checks = []

    for answer_case in answer_checks:
        result_dict = answer_case["result"]
        compact_checks.append(
            {
                "case_id": answer_case["case_id"],
                "query": answer_case["query"],
                "notes": answer_case["notes"],
                "answer": result_dict.get("answer"),
                "answer_raw": result_dict.get("answer_raw"),
                "answer_citations": result_dict.get("answer_citations", []),
                "top1_chunk_id": result_dict["results"][0]["chunk_id"] if result_dict["results"] else None,
            }
        )

    return compact_checks


def normalize_workflow_error(error: Exception) -> str:
    """
    把底层异常整理成更适合 workflow 报告阅读的提示语。
    """
    error_text = str(error)
    if "WinError 10061" in error_text:
        return (
            "无法连接 azusa_server。"
            "请先启动 `cargo run --bin azusa_server`，"
            "再重新运行包含 hnsw / hybrid / hybrid_rerank 的 workflow。"
        )
    return error_text


def build_workflow_summary(args: argparse.Namespace) -> dict:
    """
    执行整个工作流，并返回聚合后的结构化总结。
    """
    # 第一步先处理可选的数据准备逻辑，让 summary 里能明确记录：
    # 这次是否重建了 artifacts，是否导入了 HF 样本。
    step_summary = run_optional_steps(args)

    eval_path = Path(args.eval_path)
    eval_cases = load_eval_cases(eval_path)
    embedding_records = load_embedding_records(EMBEDDING_OUTPUT_PATH)

    mode_summaries = {}
    failed_modes = []
    for mode in args.modes:
        try:
            # 这里直接复用 run_eval.py 的核心评测函数，
            # 避免 workflow 自己再维护一套平行评测实现。
            mode_summaries[mode] = run_eval_for_mode(
                mode=mode,
                eval_cases=eval_cases,
                embedding_records=embedding_records,
                top_k=args.top_k,
            )
        except Exception as error:
            failed_modes.append(
                {
                    "mode": mode,
                    "error": normalize_workflow_error(error),
                }
            )

    answer_checks = []
    answer_check_error = None
    if not args.skip_answer_checks:
        try:
            # answer spot checks 不走大规模批量，只补几条代表 case，
            # 目的是快速看 answer + citation 链路是否仍然健康。
            answer_checks = run_answer_checks(
                embedding_records=embedding_records,
                top_k=args.top_k,
                answer_mode=args.answer_mode,
            )
        except Exception as error:
            answer_check_error = normalize_workflow_error(error)

    ranking = build_mode_ranking(mode_summaries) if mode_summaries else []
    baseline_mode = "brute" if "brute" in mode_summaries else (ranking[0]["mode"] if ranking else None)
    if baseline_mode is not None:
        # 只有存在至少一个成功 mode 时，才有 baseline / disagreement 这些比较型信息。
        delta_vs_baseline = build_delta_vs_baseline(mode_summaries, baseline_mode)
        disagreement_cases, all_fail_cases = build_case_comparisons(mode_summaries)
        compact_mode_summaries = build_compact_mode_summaries(mode_summaries)
    else:
        delta_vs_baseline = {}
        disagreement_cases = []
        all_fail_cases = []
        compact_mode_summaries = {}

    compact_answer_checks = build_compact_answer_checks(answer_checks)

    return {
        "workflow_name": args.output_name,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "steps": step_summary,
        "eval": {
            "eval_path": str(eval_path),
            "top_k": args.top_k,
            "requested_modes": args.modes,
            "completed_modes": list(mode_summaries.keys()),
            "failed_modes": failed_modes,
            "mode_catalog": build_mode_catalog(args.modes),
            "baseline_mode": baseline_mode,
            "mode_ranking_by_mrr": ranking,
            "delta_vs_baseline": delta_vs_baseline,
            "disagreement_cases": disagreement_cases[:10],
            "all_fail_cases": all_fail_cases,
            "mode_summaries": compact_mode_summaries,
        },
        "answer_checks": {
            "requested": not args.skip_answer_checks,
            "mode": args.answer_mode,
            "error": answer_check_error,
            "cases": compact_answer_checks,
        },
    }


def format_ratio(value: float) -> str:
    """
    统一格式化比率类指标，避免 report 中的小数位风格不一致。
    """
    return f"{value:.4f}"


def build_markdown_report(summary: dict) -> str:
    """
    把聚合后的 workflow summary 压成一份人类可快速扫读的 Markdown 报告。
    """
    lines = []
    lines.append("# Workflow Report")
    lines.append("")
    lines.append(f"- generated_at: `{summary['generated_at']}`")
    lines.append(f"- workflow_name: `{summary['workflow_name']}`")
    lines.append(f"- eval_path: `{summary['eval']['eval_path']}`")
    lines.append(f"- top_k: `{summary['eval']['top_k']}`")
    lines.append("")

    lines.append("## Steps")
    lines.append("")
    for step_name, step_result in summary["steps"].items():
        lines.append(
            f"- `{step_name}`: status=`{step_result['status']}` requested=`{step_result['requested']}`"
        )
    lines.append("")

    lines.append("## Mode Guide")
    lines.append("")
    mode_catalog = summary["eval"].get("mode_catalog", [])
    if not mode_catalog:
        lines.append("- 当前没有可展示的 mode 说明。")
    else:
        lines.append("| mode | family | label | notes |")
        lines.append("| --- | --- | --- | --- |")
        for mode_item in mode_catalog:
            lines.append(
                "| "
                + mode_item["mode"]
                + " | "
                + mode_item["family"]
                + " | "
                + mode_item["label"]
                + " | "
                + mode_item["notes"]
                + " |"
            )
    lines.append("")

    lines.append("## Metric Ranking")
    lines.append("")
    if not summary["eval"]["mode_ranking_by_mrr"]:
        lines.append("- 没有成功完成任何 mode 的评测。")
    else:
        # 先给一个最小总表，让读者先从总体指标看“谁更强”。
        lines.append("| mode | family | avg_hit@k | avg_recall@k | avg_mrr |")
        lines.append("| --- | --- | --- | --- | --- |")
        for ranking_item in summary["eval"]["mode_ranking_by_mrr"]:
            family = MODE_METADATA.get(ranking_item["mode"], {}).get("family", "unknown")
            lines.append(
                "| "
                + ranking_item["mode"]
                + " | "
                + family
                + " | "
                + format_ratio(ranking_item["avg_hit_at_k"])
                + " | "
                + format_ratio(ranking_item["avg_recall_at_k"])
                + " | "
                + format_ratio(ranking_item["avg_mrr"])
                + " |"
            )
    lines.append("")

    baseline_mode = summary["eval"]["baseline_mode"]
    lines.append("## Delta Vs Baseline")
    lines.append("")
    if baseline_mode is None:
        lines.append("- 没有可用 baseline。")
    else:
        lines.append(f"- baseline_mode: `{baseline_mode}`")
        lines.append("")
        lines.append("| mode | hit delta | recall delta | mrr delta |")
        lines.append("| --- | --- | --- | --- |")
        ordered_modes = sorted(
            summary["eval"]["delta_vs_baseline"].keys(),
            key=get_mode_order_key,
        )
        for mode in ordered_modes:
            delta_item = summary["eval"]["delta_vs_baseline"][mode]
            lines.append(
                "| "
                + mode
                + " | "
                + format_ratio(delta_item["avg_hit_at_k_delta"])
                + " | "
                + format_ratio(delta_item["avg_recall_at_k_delta"])
                + " | "
                + format_ratio(delta_item["avg_mrr_delta"])
                + " |"
            )
    lines.append("")

    lines.append("## Most Divergent Cases")
    lines.append("")
    disagreement_cases = summary["eval"]["disagreement_cases"]
    if not disagreement_cases:
        lines.append("- 当前没有明显拉开差距的 case。")
    else:
        # 先给一个总表，方便快速扫哪几条 query 最能拉开不同方案的差距。
        lines.append("| query_id | hit spread | recall spread | mrr spread |")
        lines.append("| --- | --- | --- | --- |")
        for case_item in disagreement_cases[:5]:
            lines.append(
                "| "
                + case_item["query_id"]
                + " | "
                + format_ratio(case_item["hit_spread"])
                + " | "
                + format_ratio(case_item["recall_spread"])
                + " | "
                + format_ratio(case_item["mrr_spread"])
                + " |"
            )
        lines.append("")

        # 再逐条展开 mode 对比表，避免报告过长重新变回“要逐条翻”。
        for case_item in disagreement_cases[:5]:
            lines.append(f"### {case_item['query_id']}")
            lines.append("")
            lines.append(f"- query: {case_item['query']}")
            lines.append("")
            lines.append("| mode | hit@k | recall@k | mrr | pred |")
            lines.append("| --- | --- | --- | --- | --- |")
            ordered_modes = sorted(case_item["per_mode"].keys(), key=get_mode_order_key)
            for mode in ordered_modes:
                mode_item = case_item["per_mode"][mode]
                lines.append(
                    "| "
                    + mode
                    + " | "
                    + format_ratio(mode_item["hit_at_k"])
                    + " | "
                    + format_ratio(mode_item["recall_at_k"])
                    + " | "
                    + format_ratio(mode_item["mrr"])
                    + " | "
                    + str(mode_item["retrieved_chunk_ids"])
                    + " |"
                )
            lines.append("")

    lines.append("## All-Fail Cases")
    lines.append("")
    all_fail_cases = summary["eval"]["all_fail_cases"]
    if not all_fail_cases:
        lines.append("- 没有出现所有 mode 全部 miss 的 case。")
    else:
        for case_item in all_fail_cases[:5]:
            lines.append(f"- `{case_item['query_id']}`: {case_item['query']}")
    lines.append("")

    lines.append("## Answer Checks")
    lines.append("")
    answer_summary = summary["answer_checks"]
    if not answer_summary["requested"]:
        lines.append("- 本次工作流跳过了 answer 检查。")
    elif answer_summary["error"] is not None:
        lines.append(f"- answer 检查失败：`{answer_summary['error']}`")
    else:
        # answer 这里只保留 spot checks，不把整个 answer_context 原样塞进报告里，
        # 否则会让报告重新膨胀成一大堆原文。
        lines.append(f"- answer_mode: `{answer_summary['mode']}`")
        lines.append("")
        for answer_case in answer_summary["cases"]:
            lines.append(f"### {answer_case['case_id']}")
            lines.append("")
            lines.append(f"- query: {answer_case['query']}")
            lines.append(f"- notes: {answer_case['notes']}")
            lines.append(f"- answer: {answer_case.get('answer', '')}")
            lines.append(f"- citations: {answer_case.get('answer_citations', [])}")
            lines.append(f"- top1: {answer_case.get('top1_chunk_id', 'N/A')}")
            lines.append("")

    failed_modes = summary["eval"]["failed_modes"]
    if failed_modes:
        lines.append("## Failed Modes")
        lines.append("")
        for failed_mode in failed_modes:
            lines.append(f"- `{failed_mode['mode']}`: `{failed_mode['error']}`")
        lines.append("")

    return "\n".join(lines)


def write_output_files(output_name: str, summary: dict) -> tuple[Path, Path]:
    """
    把 workflow 的两份核心产物写到 output/eval_runs/ 下。

    返回值：
    - summary_path: 结构化 JSON 汇总
    - report_path: 供人快速扫读的 Markdown 报告
    """
    EVAL_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = EVAL_RUNS_DIR / f"{output_name}_summary.json"
    report_path = EVAL_RUNS_DIR / f"{output_name}_report.md"

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(build_markdown_report(summary), encoding="utf-8")
    return summary_path, report_path


def print_terminal_summary(summary: dict, summary_path: Path, report_path: Path) -> None:
    """
    在终端打印一个精简版摘要，避免用户跑完后还不知道先看什么。
    """
    print(f"workflow_name: {summary['workflow_name']}")
    print(f"summary_json: {summary_path}")
    print(f"report_md: {report_path}")
    print()

    ranking = summary["eval"]["mode_ranking_by_mrr"]
    if ranking:
        best_mode = ranking[0]
        # 先把“当前最好的是谁”直接打出来，减少用户第一眼还要自己翻 report 的成本。
        print("best_mode_by_mrr:")
        print(
            f"  mode={best_mode['mode']} "
            f"hit@k={best_mode['avg_hit_at_k']:.4f} "
            f"recall@k={best_mode['avg_recall_at_k']:.4f} "
            f"mrr={best_mode['avg_mrr']:.4f}"
        )
        print()

        dense_sparse_modes = [
            item for item in ranking if item["mode"] in {"brute", "hnsw", "bm25", "hybrid"}
        ]
        if dense_sparse_modes:
            best_pre_rerank = dense_sparse_modes[0]
            print("best_pre_rerank_mode:")
            print(
                f"  mode={best_pre_rerank['mode']} "
                f"family={MODE_METADATA.get(best_pre_rerank['mode'], {}).get('family', 'unknown')} "
                f"hit@k={best_pre_rerank['avg_hit_at_k']:.4f} "
                f"recall@k={best_pre_rerank['avg_recall_at_k']:.4f} "
                f"mrr={best_pre_rerank['avg_mrr']:.4f}"
            )
            print()

    disagreement_cases = summary["eval"]["disagreement_cases"]
    if disagreement_cases:
        # 如果有明显拉开差距的 case，终端里直接提示最值得先看的那一条。
        print("top_divergent_case:")
        print(f"  query_id={disagreement_cases[0]['query_id']}")
        print(f"  query={disagreement_cases[0]['query']}")
        print(
            f"  spreads: hit={disagreement_cases[0]['hit_spread']:.4f} "
            f"recall={disagreement_cases[0]['recall_spread']:.4f} "
            f"mrr={disagreement_cases[0]['mrr_spread']:.4f}"
        )
        print()

    if summary["answer_checks"]["requested"] and summary["answer_checks"]["cases"]:
        first_answer_case = summary["answer_checks"]["cases"][0]
        # answer 部分终端只展示一个样例，完整内容仍以 report / summary 为准。
        print("sample_answer_check:")
        print(f"  case_id={first_answer_case['case_id']}")
        print(f"  answer={first_answer_case.get('answer', '')}")
        print()


def main() -> None:
    """
    workflow 脚本主入口。

    流程顺序：
    1. 解析命令行参数
    2. 执行 workflow 并生成结构化 summary
    3. 写出 JSON / Markdown 两份核心产物
    4. 在终端打印精简摘要
    """
    parser = build_parser()
    args = parser.parse_args()

    summary = build_workflow_summary(args)
    summary_path, report_path = write_output_files(args.output_name, summary)
    print_terminal_summary(summary, summary_path, report_path)


if __name__ == "__main__":
    main()
