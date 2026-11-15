"""运行可追溯的检索实验矩阵，保留失败单元格并独占写入三种报告。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from azusa_rag.config import EMBEDDING_OUTPUT_PATH, EVAL_RUNS_DIR, PROJECT_ROOT
from azusa_rag.evaluation_provenance import experiment_identity, file_identity, new_run_directory
from azusa_rag.loaders import load_embedding_records
from azusa_rag.modes import MODE_DISPLAY_ORDER
from azusa_rag.cli.eval import load_eval_cases, run_eval_for_mode

CSV_FIELDS = [
    "variant", "mode", "top_k", "query_count", "hit_at_k", "recall_at_k", "mrr",
    "ndcg_at_k", "p50_ms", "p95_ms", "rerank_backends", "status", "error",
]


def parse_variant(value: str) -> tuple[str, Path, Path]:
    """解析 NAME=EMBEDDING_JSONL,EVAL_JSONL；名称和两个路径均不得为空。"""
    try:
        name, paths = value.split("=", 1)
        embedding_path, eval_path = paths.split(",", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("variant must be NAME=EMBEDDING_JSONL,EVAL_JSONL") from exc
    if not all(item.strip() for item in (name, embedding_path, eval_path)):
        raise argparse.ArgumentTypeError("variant name and paths cannot be blank")
    return name.strip(), Path(embedding_path.strip()), Path(eval_path.strip())


def build_parser() -> argparse.ArgumentParser:
    """构造矩阵参数；smoke 使用仓库 fixture，其余路径显式指定或沿用本机产物。"""
    parser = argparse.ArgumentParser(description="Azusa Retrieval Benchmark Lab")
    parser.add_argument("--variant", action="append", type=parse_variant,
                        help="NAME=EMBEDDING_JSONL,EVAL_JSONL，可重复指定预构建语料版本。")
    parser.add_argument("--smoke", action="store_true", help="使用受跟踪的小型合成语料，只跑 BM25 协议回归。")
    parser.add_argument("--modes", default=None)
    parser.add_argument("--top-k", default="3,5,10")
    parser.add_argument("--output-dir", type=Path, help="新输出目录，拒绝覆盖已有结果。")
    return parser


def _inputs(embedding_path: Path, eval_path: Path) -> dict:
    """记录两个输入文件的精确字节身份，不接受 loader 的隐式路径回退。"""
    return {"embedding": file_identity(embedding_path), "eval": file_identity(eval_path)}


def run_matrix(
    variants: list[tuple[str, Path, Path]],
    modes: list[str],
    top_ks: list[int],
) -> dict:
    """每个 variant 加载一次，遍历完整笛卡尔积；数据或单格失败均保留坐标。

    同名 variant、重复坐标和空矩阵直接拒绝。HNSW 仍遵守 P0-A 的不可变快照契约，
    不为切换 variant 自动清空用户服务；冲突作为失败单元格记录。
    """
    if not variants or not modes or not top_ks:
        raise ValueError("Matrix axes must not be empty")
    names = [item[0] for item in variants]
    if any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("Variant names must be non-blank and unique")
    if len(set(modes)) != len(modes) or len(set(top_ks)) != len(top_ks):
        raise ValueError("Matrix coordinates must be unique")
    if any(type(k) is not int or k < 1 for k in top_ks):
        raise ValueError("top-k values must be positive integers")

    report = {
        "schema_version": 2, "provenance": experiment_identity(),
        "expected_cell_count": len(variants) * len(modes) * len(top_ks),
        "variants": {}, "rows": [], "failures": [], "runs": [],
    }
    for name, embedding_path, eval_path in variants:
        # 即使文件不存在，也保留请求路径；没有哈希表示该输入尚未核验。
        report["variants"][name] = {
            "embedding": {"path": str(embedding_path.resolve()), "sha256": None},
            "eval": {"path": str(eval_path.resolve()), "sha256": None},
        }
        try:
            identity = _inputs(embedding_path, eval_path)
            records = load_embedding_records(embedding_path)
            cases = load_eval_cases(eval_path)
            if _inputs(embedding_path, eval_path) != identity:
                raise ValueError("Input files changed while loading")
            report["variants"][name] = identity
        except Exception as exc:
            for k in top_ks:
                for mode in modes:
                    report["failures"].append({
                        "variant": name, "mode": mode, "top_k": k,
                        "stage": "load", "error": str(exc),
                    })
            continue

        for k in top_ks:
            for mode in modes:
                try:
                    if _inputs(embedding_path, eval_path) != identity:
                        raise ValueError("Input files changed during matrix execution")
                    summary = run_eval_for_mode(mode, cases, records, k)
                    if _inputs(embedding_path, eval_path) != identity:
                        raise ValueError("Input files changed during matrix execution")
                    row = {
                        "variant": name, "mode": mode, "top_k": k,
                        "query_count": summary["query_count"],
                        "hit_at_k": summary["avg_hit_at_k"], "recall_at_k": summary["avg_recall_at_k"],
                        "mrr": summary["avg_mrr"], "ndcg_at_k": summary["avg_ndcg_at_k"],
                        "p50_ms": summary["p50_retrieval_latency_ms"], "p95_ms": summary["p95_retrieval_latency_ms"],
                        "rerank_backends": summary.get("rerank_backends", {}),
                    }
                    # 逐题排名、实际后端、预热和运行配置保留在 JSON，不能只交付均值。
                    json.dumps(summary, allow_nan=False)
                    report["rows"].append(row)
                    report["runs"].append({"variant": name, "mode": mode, "top_k": k, "summary": summary})
                except Exception as exc:
                    report["failures"].append({
                        "variant": name, "mode": mode, "top_k": k,
                        "stage": "evaluate", "error": str(exc),
                    })
    assert len(report["rows"]) + len(report["failures"]) == report["expected_cell_count"]
    return report


def _markdown_cell(value: object) -> str:
    """转义表格分隔符和换行，避免错误文本破坏失败清单的结构。"""
    return str(value).replace("|", r"\|").replace("\r", "").replace("\n", "<br>")


def write_outputs(output_dir: Path, report: dict) -> None:
    """独占创建 JSON/CSV/Markdown；每种格式均保留失败，不覆盖同名文件。

    不声称三个文件具备跨文件事务；任何写入异常都向上传播，命令不得报告成功。
    """
    paths = [output_dir / f"benchmark.{suffix}" for suffix in ("json", "csv", "md")]
    if any(path.exists() for path in paths):
        raise FileExistsError(f"Benchmark results already exist: {output_dir}")
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    with paths[0].open("x", encoding="utf-8") as output:
        output.write(encoded)
    with paths[1].open("x", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow({**row, "rerank_backends": json.dumps(row.get("rerank_backends", {})),
                             "status": "success", "error": ""})
        for failure in report["failures"]:
            writer.writerow({**failure, "status": "failed"})
    provenance = report.get("provenance", {})
    lines = [
        "# Azusa Retrieval Benchmark Lab", "",
        f"- run_id: {provenance.get('run_id', 'legacy')}",
        f"- commit: {provenance.get('source', {}).get('git_commit')}",
        f"- dirty: {provenance.get('source', {}).get('git_dirty')}",
        "", "> Latency：一次显式预热不计入逐题延迟；包含编码、身份检查、HTTP、检索与重排。"
        "BM25 包含每次建索引；P50/P95 为 nearest-rank，跨机器不可直接比较。", "",
        "| variant | mode | k | Hit@k | Recall@k | MRR | nDCG@k | P50 ms | P95 ms | 实际重排后端 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {_markdown_cell(row['variant'])} | {_markdown_cell(row['mode'])} | {row['top_k']} | "
            f"{row['hit_at_k']:.4f} | {row['recall_at_k']:.4f} | {row['mrr']:.4f} | "
            f"{row['ndcg_at_k']:.4f} | {row['p50_ms']:.3f} | {row['p95_ms']:.3f} | "
            f"{_markdown_cell(row.get('rerank_backends', {}))} |"
        )
    lines.extend(["", "## 输入身份", ""])
    for name, inputs in report.get("variants", {}).items():
        for kind, identity in inputs.items():
            lines.append(f"- {_markdown_cell(name)} / {kind}: {_markdown_cell(identity['path'])}, SHA-256 {identity['sha256']}")
    lines.extend(["", "## Failed cells", ""])
    lines.extend(
        f"- {_markdown_cell(item['variant'])} / {_markdown_cell(item['mode'])} / k={item['top_k']}: {_markdown_cell(item['error'])}"
        for item in report["failures"]
    )
    if not report["failures"]:
        lines.append("- 无。")
    with paths[2].open("x", encoding="utf-8") as output:
        output.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    """执行矩阵并输出实际目录；任何失败单元格导致非零退出，仍保存完整失败报告。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.smoke and args.variant:
        parser.error("--smoke and --variant cannot be combined")
    variants = args.variant or [
        ("current", Path(EMBEDDING_OUTPUT_PATH), PROJECT_ROOT / "data" / "eval" / "min_eval.jsonl")
    ]
    if args.smoke:
        fixture = PROJECT_ROOT / "tests" / "fixtures"
        variants = [("synthetic-bm25", fixture / "retrieval_smoke_embedding.jsonl", fixture / "retrieval_smoke_eval.jsonl")]
    modes = [item.strip() for item in (args.modes or ("bm25" if args.smoke else ",".join(MODE_DISPLAY_ORDER))).split(",") if item.strip()]
    if set(modes) - set(MODE_DISPLAY_ORDER):
        parser.error("Unknown retrieval modes")
    if args.smoke and modes != ["bm25"]:
        parser.error("Synthetic smoke embeddings are placeholders; only bm25 is meaningful")
    top_ks = [int(item) for item in args.top_k.split(",")]
    output_dir = args.output_dir or new_run_directory(EVAL_RUNS_DIR / "runs", "benchmark")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    report = run_matrix(variants, modes, top_ks)
    report["scope"] = "synthetic BM25 protocol regression, not retrieval-quality benchmark" if args.smoke else "user-supplied evaluation"
    write_outputs(output_dir, report)
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(report["rows"]),
                      "failure_count": len(report["failures"])}, ensure_ascii=False, indent=2))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
