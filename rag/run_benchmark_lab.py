"""Run a reproducible retrieval experiment matrix over prebuilt artifact variants."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from bootstrap import ensure_src_path

ensure_src_path()

from azusa_rag.config import EMBEDDING_OUTPUT_PATH, EVAL_RUNS_DIR, PROJECT_ROOT
from azusa_rag.loaders import load_embedding_records
from azusa_rag.modes import MODE_DISPLAY_ORDER
from run_eval import load_eval_cases, run_eval_for_mode


def parse_variant(value: str) -> tuple[str, Path, Path]:
    """Parse NAME=EMBEDDING_JSONL,EVAL_JSONL without guessing missing paths."""

    try:
        name, paths = value.split("=", 1)
        embedding_path, eval_path = paths.split(",", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("variant must be NAME=EMBEDDING_JSONL,EVAL_JSONL") from exc
    if not name.strip():
        raise argparse.ArgumentTypeError("variant name cannot be blank")
    return name.strip(), Path(embedding_path), Path(eval_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Azusa Retrieval Benchmark Lab")
    parser.add_argument(
        "--variant",
        action="append",
        type=parse_variant,
        help="Repeatable NAME=EMBEDDING_JSONL,EVAL_JSONL; use kb-sync to build chunk variants.",
    )
    parser.add_argument("--modes", default=",".join(MODE_DISPLAY_ORDER))
    parser.add_argument("--top-k", default="3,5,10")
    parser.add_argument("--output-dir", default=str(EVAL_RUNS_DIR / "benchmark_lab"))
    return parser


def run_matrix(
    variants: list[tuple[str, Path, Path]],
    modes: list[str],
    top_ks: list[int],
) -> dict:
    rows = []
    failures = []
    for variant_name, embedding_path, eval_path in variants:
        records = load_embedding_records(embedding_path)
        cases = load_eval_cases(eval_path)
        for top_k in top_ks:
            for mode in modes:
                try:
                    summary = run_eval_for_mode(mode, cases, records, top_k)
                except Exception as exc:
                    failures.append({"variant": variant_name, "mode": mode, "top_k": top_k, "error": str(exc)})
                    continue
                rows.append(
                    {
                        "variant": variant_name,
                        "mode": mode,
                        "top_k": top_k,
                        "query_count": summary["query_count"],
                        "hit_at_k": summary["avg_hit_at_k"],
                        "recall_at_k": summary["avg_recall_at_k"],
                        "mrr": summary["avg_mrr"],
                        "ndcg_at_k": summary["avg_ndcg_at_k"],
                        "p50_ms": summary["p50_retrieval_latency_ms"],
                        "p95_ms": summary["p95_retrieval_latency_ms"],
                    }
                )
    return {"schema_version": 1, "rows": rows, "failures": failures}


def write_outputs(output_dir: Path, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = ["variant", "mode", "top_k", "query_count", "hit_at_k", "recall_at_k", "mrr", "ndcg_at_k", "p50_ms", "p95_ms"]
    with (output_dir / "benchmark.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["rows"])
    lines = [
        "# Azusa Retrieval Benchmark Lab",
        "",
        "> Latency excludes the explicit warm-up request in each mode; compare on the same machine and artifact set.",
        "",
        "| variant | mode | k | Hit@k | Recall@k | MRR | nDCG@k | P50 ms | P95 ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['variant']} | {row['mode']} | {row['top_k']} | {row['hit_at_k']:.4f} | "
            f"{row['recall_at_k']:.4f} | {row['mrr']:.4f} | {row['ndcg_at_k']:.4f} | "
            f"{row['p50_ms']:.3f} | {row['p95_ms']:.3f} |"
        )
    if report["failures"]:
        lines.extend(["", "## Failed cells", ""])
        lines.extend(f"- `{item['variant']} / {item['mode']} / k={item['top_k']}`: {item['error']}" for item in report["failures"])
    (output_dir / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    variants = args.variant or [
        ("current", Path(EMBEDDING_OUTPUT_PATH), PROJECT_ROOT / "data" / "eval" / "min_eval.jsonl")
    ]
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    unknown_modes = sorted(set(modes) - set(MODE_DISPLAY_ORDER))
    if unknown_modes:
        raise ValueError(f"Unknown modes: {unknown_modes}")
    top_ks = [int(item) for item in args.top_k.split(",")]
    if not top_ks or any(item < 1 for item in top_ks):
        raise ValueError("top-k values must be positive integers")
    report = run_matrix(variants, modes, top_ks)
    write_outputs(Path(args.output_dir), report)
    print(json.dumps({"row_count": len(report["rows"]), "failure_count": len(report["failures"])}, indent=2))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
