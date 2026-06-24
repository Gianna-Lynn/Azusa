"""
Run Search Agent v0 from the command line.
"""

import argparse
import json
from pathlib import Path

from bootstrap import ensure_src_path

ensure_src_path()

from azusa_rag.agent import run_search_agent
from azusa_rag.config import EMBEDDING_OUTPUT_PATH, SEARCH_RUNS_DIR
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
        default="search_agent_v0",
        help="Output basename under output/search_runs.",
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

    SEARCH_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SEARCH_RUNS_DIR / f"{args.output_name}_{args.mode}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print_agent_summary(result, output_path)


if __name__ == "__main__":
    main()
