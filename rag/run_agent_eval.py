"""
运行 Search Agent v0 的样例评测，并保存每道题的完整执行轨迹。

这份脚本评估三个彼此独立的环节：
1. 查询规划：问题是否被识别为预期的单跳或多跳查询。
2. 证据检索：最终证据中找回了多少人工标注的相关 chunk。
3. 引用质量：启用 ``--with-answer`` 后，模型引用是否准确、完整。

注意：``reference_answer`` 目前只用于人工复核，脚本尚未自动判断答案文本
是否正确。因此，不带 ``--with-answer`` 时，本脚本只评估查询规划和证据检索。
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter

from bootstrap import ensure_src_path

ensure_src_path()

from azusa_rag.agent import run_search_agent
from azusa_rag.config import EMBEDDING_OUTPUT_PATH, LATEST_EVAL_RUN_DIR, PROJECT_ROOT
from azusa_rag.loaders import load_embedding_records
from azusa_rag.modes import SEARCH_MODES


# 每一行评测数据都必须提供这些字段。集中声明可以让数据契约一眼可见。
REQUIRED_CASE_FIELDS = {
    "case_id",
    "category",
    "query",
    "reference_answer",
    "expected_query_type",
    "expected_min_hops",
    "relevant_chunk_ids",
    "expected_citations",
}


def build_parser() -> argparse.ArgumentParser:
    """构造 Agent 样例评测的命令行参数。"""
    parser = argparse.ArgumentParser(description="Run the Stage 3 Search Agent evaluation set")
    parser.add_argument("--mode", choices=SEARCH_MODES, default="bm25")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--with-answer",
        action="store_true",
        help="同时运行 Answerer，并评估 citation；默认只评估规划和检索。",
    )
    parser.add_argument(
        "--eval-path",
        default=str(PROJECT_ROOT / "data" / "eval" / "agent_eval.jsonl"),
        help = "默认使用 data/eval/agent_eval.jsonl 样例集。",
    )
    parser.add_argument("--embedding-path", default=str(EMBEDDING_OUTPUT_PATH))
    parser.add_argument(
        "--output-path",
        default=None,
        help="默认写入 output/eval_runs/latest/agent_eval_<mode>.json。",
    )
    return parser


def load_agent_eval_cases(eval_path: Path) -> list[dict]:
    """
    读取并校验 Agent JSONL 样例集。

    JSONL 文件每个非空行代表一道题。这里仅校验结构是否完整；
    查询内容和金标 chunk 是否合理，仍由样例维护者负责确认。
    """
    # 读取 JSONL 样例集.
    # 其中一个case的示例格式如下：
    # {
    #     "case_id": "case_001",
    #     "category": "single_hop",
    #     "query": "When was the event?",
    #     "reference_answer": "1962",
    #     "expected_query_type": "single_hop",
    #     "expected_min_hops": 1,
    #     "relevant_chunk_ids": ["chunk_001", "chunk_002"],
    #     "expected_citations": ["chunk_001"],
    # }
    cases = []
    with eval_path.open("r", encoding="utf-8") as eval_file:
        for line_number, line in enumerate(eval_file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                case = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Agent eval 第 {line_number} 行不是有效 JSON") from error

            cases.append(case)

    if not cases:
        raise ValueError(f"Agent eval 样例集为空: {eval_path}")

    seen_case_ids = set()
    for case in cases:
        # 检查每条样例是否包含所有必需字段。
        missing_fields = sorted(REQUIRED_CASE_FIELDS - case.keys())
        if missing_fields:
            raise ValueError(f"Agent eval 样例缺少字段 {missing_fields}: {case}")

        # case_id 是结果文件中定位单题的主键，因此不允许重复。
        case_id = case["case_id"]
        if case_id in seen_case_ids:
            raise ValueError(f"Agent eval case_id 重复: {case_id}")
        seen_case_ids.add(case_id)

    return cases


def evaluate_query_plan(case: dict, result: dict) -> dict:
    """评估查询类型和检索 hop 数是否符合样例预期。"""
    plan = result["query_plan"]
    actual_query_type = plan["query_type"]
    actual_hops = len(result["hops"])

    return {
        "expected_query_type": case["expected_query_type"],
        "actual_query_type": actual_query_type,
        "query_type_match": actual_query_type == case["expected_query_type"],
        "expected_min_hops": case["expected_min_hops"], # expected_min_hops 表示最低要求，不表示 Agent 必须刚好执行这么多跳。
        "actual_hops": actual_hops,
        "hop_count_ok": actual_hops >= case["expected_min_hops"],
    }


def evaluate_retrieved_evidence(case: dict, result: dict) -> dict:
    """评估融合后的证据找回了多少人工标注的相关 chunk。"""
    retrieved_ids = [item["chunk_id"] for item in result["fused_evidence"]]
    relevant_ids = set(case["relevant_chunk_ids"])
    matched_ids = [chunk_id for chunk_id in retrieved_ids if chunk_id in relevant_ids]

    # relevant_chunk_ids 为空表示该题预期“证据不足”。
    # 这种题没有可供召回的金标证据，所以普通 evidence recall 不适用。
    evidence_recall = None
    if relevant_ids:
        evidence_recall = len(set(matched_ids)) / len(relevant_ids)

    return {
        "relevant_chunk_ids": case["relevant_chunk_ids"],
        "retrieved_chunk_ids": retrieved_ids,
        "matched_chunk_ids": matched_ids,
        "evidence_recall": evidence_recall,
    }


def evaluate_citations(case: dict, result: dict) -> dict:
    """
    评估 Answerer 输出的 citation。

    返回 ``None`` 表示本次没有运行 Answerer，并不是得分为零。
    对“证据不足”样例，保持空引用就是正确行为，因此 precision 和 recall
    都记为 1；如果模型制造了引用，则两项都记为 0。
    """
    expected_citations = set(case["expected_citations"])

    # 如果没有运行 Answerer，则不评估 citation, 返回 None 表示“不适用”。
    # 见 rag\src\azusa_rag\agent.py里的run_search_agent函数定义,
    # 其中定义了answer和answer_citations只有在with_answer为True时才会返回, 否则不会返回这两个字段.
    if "answer" not in result:
        return {
            "expected_citations": case["expected_citations"],
            "actual_citations": [],
            "citation_precision": None,
            "citation_recall": None,
        }

    actual_citations = set(result.get("answer_citations", []))

    # 如果样例预期没有引用，则模型也不应制造引用。否则，precision 和 recall 都记为 0。
    if not expected_citations:
        citation_score = 1.0 if not actual_citations else 0.0
        citation_precision = citation_score
        citation_recall = citation_score

    # 如果样例预期有引用，而且模型也给出了引用，则计算 precision 和 recall。
    else:
        correct_citations = actual_citations & expected_citations
        citation_precision = (
            len(correct_citations) / len(actual_citations) if actual_citations else 0.0
        )
        citation_recall = len(correct_citations) / len(expected_citations)

    return {
        "expected_citations": case["expected_citations"],
        "actual_citations": sorted(actual_citations),
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
    }


def evaluate_agent_case(case: dict, result: dict, latency_ms: float) -> dict:
    """
    把单题的规划、证据和引用评分组装成一份结果。

    该函数不负责运行 Agent，也不判断答案文本是否等于 reference_answer；
    它只比较已经生成的结构化执行轨迹与样例金标。
    """
    plan_metrics = evaluate_query_plan(case, result)
    evidence_metrics = evaluate_retrieved_evidence(case, result)
    citation_metrics = evaluate_citations(case, result)

    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "query": case["query"],
        # reference_answer 暂时保留给人查看 trace，不参与自动评分。
        "reference_answer": case["reference_answer"],
        **plan_metrics,
        **evidence_metrics,
        **citation_metrics,
        "latency_ms": latency_ms,
        "notes": case.get("notes"),
        "trace": result,
    }


def average_available_scores(case_results: list[dict], field: str) -> float | None:
    """对非 None 分数求平均；所有样例都不适用时返回 None。"""
    scores = [item[field] for item in case_results if item[field] is not None]
    return sum(scores) / len(scores) if scores else None


def summarize_agent_cases(case_results: list[dict], mode: str, top_k: int) -> dict:
    """汇总全部样例的查询规划、证据召回和 citation 指标。"""
    if not case_results:
        raise ValueError("不能汇总空的 Agent eval 结果")

    case_count = len(case_results)
    return {
        "agent_version": "search_agent_v0",
        "mode": mode,
        "top_k": top_k,
        "case_count": case_count,
        "query_type_accuracy": (
            sum(item["query_type_match"] for item in case_results) / case_count
        ),
        "hop_count_pass_rate": (
            sum(item["hop_count_ok"] for item in case_results) / case_count
        ),
        # 证据不足题的 evidence_recall 为 None，因此不会稀释普通检索召回率。
        "avg_evidence_recall": average_available_scores(
            case_results, "evidence_recall"
        ),
        # 未启用 --with-answer 时，两项 citation 均保持 None。
        "avg_citation_precision": average_available_scores(
            case_results, "citation_precision"
        ),
        "avg_citation_recall": average_available_scores(
            case_results, "citation_recall"
        ),
        "cases": case_results,
    }


def print_agent_eval_summary(summary: dict) -> None:
    """在终端打印便于快速检查的 Agent 评测摘要。"""
    print(f"mode: {summary['mode']}")
    print(f"case_count: {summary['case_count']}")
    print(f"query_type_accuracy: {summary['query_type_accuracy']:.4f}")
    print(f"hop_count_pass_rate: {summary['hop_count_pass_rate']:.4f}")

    if summary["avg_evidence_recall"] is not None:
        print(f"avg_evidence_recall: {summary['avg_evidence_recall']:.4f}")

    if summary["avg_citation_precision"] is None:
        print("citation metrics: 未启用 --with-answer，本次不评估回答引用")
    else:
        print(f"avg_citation_precision: {summary['avg_citation_precision']:.4f}")
        print(f"avg_citation_recall: {summary['avg_citation_recall']:.4f}")


def write_agent_eval_summary(output_path: Path, summary: dict) -> Path:
    """写入评测结果；目标文件被占用时回退到系统临时目录。"""
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
        return output_path
    except PermissionError:
        # Windows 下编辑器或旧进程可能占用结果文件。回退路径可以保住本次结果，
        # main() 最后会打印真实保存位置，避免使用者误以为写入了原路径。
        fallback_dir = Path(tempfile.gettempdir()) / "azusa_agent_eval"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_path = fallback_dir / output_path.name
        fallback_path.write_text(payload, encoding="utf-8")
        return fallback_path


def main() -> None:
    """加载样例和 embedding，逐题运行 Agent，然后汇总并保存结果。"""
    args = build_parser().parse_args()
    cases = load_agent_eval_cases(Path(args.eval_path))
    embedding_records = load_embedding_records(Path(args.embedding_path))

    case_results = []
    for case in cases:
        # 计时范围只包含 Agent 执行，不包含样例读取和最终写文件。
        started_at = perf_counter()
        result = run_search_agent(
            query=case["query"],
            embedding_records=embedding_records,
            mode=args.mode,
            top_k=args.top_k,
            with_answer=args.with_answer,
        )
        latency_ms = (perf_counter() - started_at) * 1000.0
        case_results.append(evaluate_agent_case(case, result, latency_ms))

    summary = summarize_agent_cases(case_results, args.mode, args.top_k)
    output_path = (
        Path(args.output_path)
        if args.output_path
        else LATEST_EVAL_RUN_DIR / f"agent_eval_{args.mode}.json"
    )
    actual_output_path = write_agent_eval_summary(output_path, summary)

    print_agent_eval_summary(summary)
    print(f"saved to: {actual_output_path}")


if __name__ == "__main__":
    main()
