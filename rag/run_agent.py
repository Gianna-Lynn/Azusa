"""
Run Search Agent v0 from the command line.
"""

import argparse
import json
from pathlib import Path

from bootstrap import ensure_src_path

ensure_src_path()

from azusa_rag.agent import run_search_agent
from azusa_rag.config import EMBEDDING_OUTPUT_PATH, LATEST_SEARCH_RUN_DIR, SEARCH_RUNS_DIR
from azusa_rag.loaders import load_embedding_records
from azusa_rag.modes import SEARCH_MODES
from azusa_rag.search import get_query


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Azusa Search Agent v0")
    parser.add_argument("--mode", choices=SEARCH_MODES, default="bm25")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--with-answer", action="store_true")
    parser.add_argument(
        "--embedding-path",
        default=str(EMBEDDING_OUTPUT_PATH),
        help="Path to embedding jsonl. Defaults to output/artifacts/embedding.jsonl.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Optional logical run name. By default results are overwritten under "
            "output/search_runs/latest/."
        ),
    )
    parser.add_argument(
        "--keep-run-copy",
        action="store_true",
        help="Also keep a named copy under output/search_runs/ for this run.",
    )
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument("--query-text")
    query_group.add_argument("--query-file")
    return parser


def load_query_text(args: argparse.Namespace) -> str:
    if args.query_text is not None:
        query = args.query_text.strip()
        if not query:
            raise ValueError("--query-text cannot be blank")
        return query

    if args.query_file is not None:
        query_path = Path(args.query_file)
        query = query_path.read_text(encoding="utf-8").strip()
        if not query:
            raise ValueError(f"Query file is blank: {query_path}")
        return query

    return get_query()


def print_agent_summary(result: dict, output_path: Path) -> None:
    plan = result["query_plan"]
    print(f"agent_version: {result['agent_version']}")
    print(f"query_type: {plan['query_type']}")
    print(f"rewritten_query: {plan['rewritten_query']}")
    print(f"retrieval_queries: {plan['retrieval_queries']}")
    print(f"mode: {result['mode']}")
    print(f"top_k: {result['top_k']}")
    print()
    print("fused_evidence:")
    for item in result["fused_evidence"]:
        fusion = item["fusion"]
        print(
            f"  [{item['rank']}] {item['chunk_id']} "
            f"hits={fusion['hit_count']} best_rank={fusion['best_rank']}"
        )
        print(f"      {item['preview']}")
    if "answer" in result:
        print()
        print("answer:")
        print(result["answer"])
    print()
    print(f"saved to: {output_path}")


def write_agent_outputs(
    mode: str,
    result: dict,
    output_name: str | None,
    keep_run_copy: bool,
) -> tuple[Path, Path | None]:
    """
    默认把 agent 结果写到 latest 目录；只有显式要求时才额外保留命名副本。
    """
    LATEST_SEARCH_RUN_DIR.mkdir(parents=True, exist_ok=True)
    latest_path = LATEST_SEARCH_RUN_DIR / f"search_agent_v0_{mode}.json"
    latest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    run_copy_path = None
    if keep_run_copy and output_name:
        SEARCH_RUNS_DIR.mkdir(parents=True, exist_ok=True)
        run_copy_path = SEARCH_RUNS_DIR / f"{output_name}_{mode}.json"
        run_copy_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return latest_path, run_copy_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    query = load_query_text(args)
    embedding_records = load_embedding_records(Path(args.embedding_path))
    result = run_search_agent(
        query=query,
        embedding_records=embedding_records,
        mode=args.mode,
        top_k=args.top_k,
        with_answer=args.with_answer,
    )

    latest_path, run_copy_path = write_agent_outputs(
        mode=args.mode,
        result=result,
        output_name=args.output_name,
        keep_run_copy=args.keep_run_copy,
    )
    print_agent_summary(result, latest_path)
    if run_copy_path is not None:
        print(f"named copy: {run_copy_path}")


if __name__ == "__main__":
    main()
