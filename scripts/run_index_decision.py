"""比较同数据、同 L2 度量的 Flat/HNSW；小型自有合成数据只验证实验方法。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
from uuid import uuid4


def percentile(values: list[float], quantile: float) -> float:
    """返回 nearest-rank 观测值，调用方固定使用 0.5/0.95；空列表拒绝评分。"""
    if not values:
        raise ValueError("No query observations")
    return sorted(values)[max(0, math.ceil(len(values)*quantile)-1)]


def execute(worker: Path, config: dict, target: Path, cell: str) -> dict:
    """运行一个已冻结输入的 worker，返回 success/observation 或 failure/error。

    每格单独启动进程，校验查询数、排名唯一性、ID 范围和有限延迟；没有内存观测
    也按失败保留，不能写成零。输入落盘发生在启动之前，写入失败直接传播；进程和
    协议失败转成单格结果，交给矩阵报告保留坐标，而非省略失败后聚合成功格。
    """
    path = target/f"{cell}.input.json"
    path.write_text(json.dumps(config, allow_nan=False), encoding="utf-8")
    try:
        process = subprocess.run([str(worker), str(path)], capture_output=True, timeout=120)
        (target/f"{cell}.stdout").write_bytes(process.stdout)
        (target/f"{cell}.stderr").write_bytes(process.stderr)
        if process.returncode != 0:
            raise RuntimeError(f"Worker exit code {process.returncode}")
        result = json.loads(process.stdout)
        if len(result["queries"]) != len(config["queries"]):
            raise ValueError("Incomplete query observations")
        for observation in result["queries"]:
            ids = observation["ids"]
            if len(ids) != config["k"] or len(set(ids)) != len(ids) or any(
                type(value) is not int or not 0 <= value < len(config["base"]) for value in ids
            ):
                raise ValueError("Invalid ranking")
            if not math.isfinite(observation["latency_ms"]) or observation["latency_ms"] < 0:
                raise ValueError("Invalid latency")
        if result["peak_rss_bytes"] is None:
            raise ValueError("Peak RSS unavailable on this platform")
        return {"status": "success", "observation": result}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return {"status": "failure", "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    """固定数据和插入排列种子，在独立进程重复建图并输出三格式报告。

    同规模的 Flat 是当前数据与 L2 度量下的精确金标；它失败时同组 HNSW 也不能
    发布 Recall。查询分位数来自逐查询观测，RSS 是整个进程峰值。运行前后核对
    源码和 worker 散列，避免把不同版本合成一个实验；任意格失败则命令非零退出。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="128,512,2048")
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    if not sizes or len(set(sizes)) != len(sizes) or min(sizes) < 5 or min(args.queries, args.dimension, args.repeats) < 1:
        parser.error("需要唯一规模 >=5，查询数、维度、重复数为正整数")
    root = Path(__file__).resolve().parents[1]
    worker = (args.worker or root/"target/release"/(
        "index_decision_benchmark.exe" if platform.system() == "Windows" else "index_decision_benchmark"
    )).resolve()
    if not worker.is_file():
        parser.error("请先离线构建 release 工作进程，或显式指定 --worker")
    target = root/"rag/.cache/index-decision"/uuid4().hex
    target.mkdir(parents=True, exist_ok=False)
    source_paths = [root/"Cargo.toml", root/"Cargo.lock", Path(__file__).resolve()]
    source_paths += list((root/"src").rglob("*.rs"))
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    worker_hash = hashlib.sha256(worker.read_bytes()).hexdigest()
    rng = random.Random(args.seed)
    base = [[rng.uniform(-1, 1) for _ in range(args.dimension)] for _ in range(max(sizes))]
    query_rng = random.Random(args.seed+1)
    queries = [[query_rng.uniform(-1, 1) for _ in range(args.dimension)] for _ in range(args.queries)]
    rows = []
    for size in sizes:
        data_hash = hashlib.sha256(json.dumps([base[:size], queries]).encode()).hexdigest()
        for repeat in range(args.repeats):
            order = list(range(size))
            random.Random(args.seed+repeat+100).shuffle(order)
            algorithms = [("flat", 16), ("hnsw", 16), ("hnsw", 64)]
            random.Random(args.seed+repeat+size).shuffle(algorithms)
            group = []
            for algorithm, ef in algorithms:
                config = {"base": base[:size], "queries": queries, "order": order,
                          "algorithm": algorithm, "k": 5, "ef_search": ef, "m": 8, "ef_construction": 80}
                cell = f"n{size}-r{repeat}-{algorithm}-ef{ef}"
                group.append({"size": size, "repeat": repeat, "algorithm": algorithm, "ef_search": ef,
                              "dataset_sha256": data_hash, "cell": cell, **execute(worker, config, target, cell)})
            flat = next(row for row in group if row["algorithm"] == "flat")
            for row in group:
                if row["status"] == "success" and flat["status"] == "success":
                    observed = row["observation"]
                    gold = flat["observation"]["queries"]
                    observations = observed["queries"]
                    row["recall_at_5"] = sum(len(set(a["ids"]) & set(b["ids"]))/5 for a,b in zip(observations,gold))/len(gold)
                    row["p50_ms"] = percentile([q["latency_ms"] for q in observations], 0.5)
                    row["p95_ms"] = percentile([q["latency_ms"] for q in observations], 0.95)
                    row["build_ms"] = observed["build_ms"]
                    row["peak_rss_bytes"] = observed["peak_rss_bytes"]
                elif row["status"] == "success":
                    row.update(status="failure", error="Flat reference unavailable")
            rows.extend(group)
    if any(hashlib.sha256((root/name).read_bytes()).hexdigest() != digest for name,digest in hashes.items()) or hashlib.sha256(worker.read_bytes()).hexdigest() != worker_hash:
        raise RuntimeError("实验期间源码或二进制变化，拒绝发布报告")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    report = {"kind": "synthetic-index-decision", "base_commit": commit, "source_sha256": hashes,
              "worker_sha256": worker_hash, "platform": platform.platform(), "processor": platform.processor(),
              "python": platform.python_version(), "config": vars(args) | {"worker": str(worker)},
              "metric": "L2", "k": 5, "graph_rng_seed": 42, "m": 8, "ef_construction": 80,
              "warmup": "one full query pass per fresh process", "memory_scope": "whole worker peak RSS / Windows peak working set",
              "failure_count": sum(row["status"] != "success" for row in rows), "rows": rows}
    (target/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["size","repeat","algorithm","ef_search","status","recall_at_5","p50_ms","p95_ms","build_ms","peak_rss_bytes","error"]
    with (target/"report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 合成数据 Flat/HNSW 对照", "", "仅描述本机小规模实验，不能外推真实 embedding、SIFT1M 或生产阈值。", "",
             "|规模|重复|算法|ef|状态|Recall@5|P95 ms|构建 ms|进程峰值字节|", "|---|---|---|---|---|---|---|---|---|"]
    lines += ["|"+"|".join(str(row.get(key,"")) for key in ("size","repeat","algorithm","ef_search","status","recall_at_5","p95_ms","build_ms","peak_rss_bytes"))+"|" for row in rows]
    lines += ["", "失败单元："] + [f'- {row["cell"]}: {row["error"]}' for row in rows if row["status"] != "success"]
    (target/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"report": str(target/"report.json"), "failures": report["failure_count"]}))
    raise SystemExit(1 if report["failure_count"] else 0)


if __name__ == "__main__":
    main()
