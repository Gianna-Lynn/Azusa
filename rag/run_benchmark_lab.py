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
    # TODO AZUSA-V2-L3a：交付完整实验矩阵执行器。
    # 每个 variant 只加载一次 artifacts；遍历 top_k × modes；单元格失败必须进入
    # failures 而不能抹掉成功格；成功行字段与 assignment 表格完全一致。
    raise NotImplementedError("TODO AZUSA-V2-L3a：实现 Retrieval Benchmark 矩阵")


def write_outputs(output_dir: Path, report: dict) -> None:
    # TODO AZUSA-V2-L3b：交付 benchmark.json / benchmark.csv / benchmark.md。
    # JSON 保留完整 report；CSV 使用固定字段与 utf-8-sig；Markdown 含指标表、
    # latency 口径声明和失败单元格。三份产物不得只写成功结果而隐藏 failures。
    raise NotImplementedError("TODO AZUSA-V2-L3b：实现三格式实验报告")


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
