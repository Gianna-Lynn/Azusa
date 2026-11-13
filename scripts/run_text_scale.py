"""Run a pinned full-corpus DuRetrieval BM25 experiment without changing Azusa data.

Uses Azusa's tokenizer extracted from a supplied source snapshot. Builds BM25 once;
query timings therefore exclude index construction, downloads and model inference.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import time
import urllib.request
from pathlib import Path


DATASETS = {
    "C-MTEB/DuRetrieval": "a1a333e290fe30b10f3f56498e3a0d911a693ced",
    "C-MTEB/DuRetrieval-qrels": "497b7bd1bbb25cb3757ff34d95a8be50a3de2279",
}


def digest(path):
    """Hash exact artifact bytes for later reproduction."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tokenizer(path):
    """Extract the regex and tokenizer from the actual Azusa source, without model imports."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "BM25_TOKEN_PATTERN"
            for target in node.targets
        ):
            selected.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "tokenize_for_bm25":
            selected.append(node)
    if len(selected) != 2:
        raise ValueError("Expected Azusa regex and tokenizer definitions")
    namespace = {"re": re}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["tokenize_for_bm25"]


def metrics(ranking, relevant, k):
    """Compute binary retrieval metrics with the requested cutoff and complete qrels."""
    if not relevant or len(set(ranking)) != len(ranking):
        raise ValueError("Missing gold or duplicate ranking")
    hits = [int(pid in relevant) for pid in ranking[:k]]
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return {"hit": float(any(hits)), "recall": sum(hits) / len(relevant),
            "mrr": next((1 / (rank + 1) for rank, hit in enumerate(hits) if hit), 0.0),
            "ndcg": dcg / ideal}


def fetch_data(root):
    """Download pinned Parquet files, retain source metadata, and read all splits."""
    import pyarrow.parquet as pq

    tables, provenance = {}, []
    for repo, revision in DATASETS.items():
        api = f"https://hf-mirror.com/api/datasets/{repo}/revision/{revision}"
        with urllib.request.urlopen(api, timeout=90) as response:
            metadata = json.load(response)
        if metadata["sha"] != revision:
            raise ValueError("Dataset revision mismatch")
        (root / (repo.split("/")[-1] + "-metadata.json")).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        for entry in metadata["siblings"]:
            name = entry["rfilename"]
            if not name.endswith(".parquet"):
                continue
            url = f"https://hf-mirror.com/datasets/{repo}/resolve/{revision}/{name}"
            path = root / (repo.split("/")[-1] + "-" + Path(name).name)
            print("Downloading", url, flush=True)
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                            "--connect-timeout", "20", "--max-time", "300", "--retry", "2",
                            "--output", str(path), url], check=True)
            rows = pq.read_table(path).to_pylist()
            split = Path(name).name.split("-")[0]
            if split in tables:
                raise ValueError("Unexpected sharded split; do not silently overwrite")
            tables[split] = rows
            provenance.append({"repo": repo, "revision": revision, "url": url,
                               "rows": len(rows), "sha256": digest(path)})
    return tables, provenance


def run(output, source):
    """Validate complete identities, warm up, then evaluate all queries over the full corpus."""
    import numpy as np
    from rank_bm25 import BM25Okapi

    tokenize = load_tokenizer(source)
    tables, inputs = fetch_data(output)
    corpus, queries = tables["corpus"], tables["queries"]
    ids = [str(row["id"]) for row in corpus]
    qids = [str(row["id"]) for row in queries]
    if len(set(ids)) != len(ids) or len(set(qids)) != len(qids):
        raise ValueError("Duplicate input identities")
    corpus_ids, query_ids = set(ids), set(qids)
    gold = {}
    for row in tables["dev"]:
        qid, pid = str(row["qid"]), str(row["pid"])
        if qid not in query_ids or pid not in corpus_ids:
            raise ValueError("Qrels identity absent from full dataset")
        if row["score"] > 0:
            gold.setdefault(qid, set()).add(pid)
    if set(gold) != query_ids:
        raise ValueError("Some queries have no positive gold; refuse silent dropping")
    started = time.perf_counter()
    index = BM25Okapi([tokenize(row["text"]) for row in corpus])
    build_ms = (time.perf_counter() - started) * 1000
    print(f"Built corpus={len(corpus)} queries={len(queries)} in {build_ms:.1f} ms", flush=True)

    def search(text):
        # Stable descending sort gives the same score/index tie-break as Azusa.
        scores = index.get_scores(tokenize(text))
        return [ids[int(i)] for i in np.argsort(-scores, kind="stable")[:100]]

    started = time.perf_counter()
    for row in queries[:10]:
        search(row["text"])
    warmup_ms = (time.perf_counter() - started) * 1000
    observations = []
    with (output / "queries.jsonl").open("x", encoding="utf-8") as stream:
        for number, row in enumerate(queries, 1):
            started = time.perf_counter()
            ranking = search(row["text"])
            latency = (time.perf_counter() - started) * 1000
            item = {"query_id": str(row["id"]), "query": row["text"],
                    "ranking": ranking, "relevant_ids": sorted(gold[str(row["id"])]),
                    "latency_ms": latency,
                    "metrics": {str(k): metrics(ranking, gold[str(row["id"])], k)
                                for k in (5, 10, 100)}}
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            observations.append(item)
            if number % 50 == 0 or number == 1:
                print(f"Evaluated {number}/{len(queries)}", flush=True)
    latencies = sorted(item["latency_ms"] for item in observations)
    summary = {str(k): {name: sum(item["metrics"][str(k)][name] for item in observations)
                            / len(observations) for name in ("hit", "recall", "mrr", "ndcg")}
               for k in (5, 10, 100)}
    report = {"status": "success", "corpus_count": len(corpus), "query_count": len(queries),
              "qrels_count": len(tables["dev"]), "inputs": inputs,
              "source_sha256": digest(source), "runner_sha256": digest(Path(__file__)),
              "python": platform.python_version(), "platform": platform.platform(),
              "packages": {name: importlib.metadata.version(name) for name in ("numpy", "rank-bm25", "pyarrow")},
              "build_ms": build_ms, "warmup_queries": 10, "warmup_ms": warmup_ms,
              "latency_ms": {"mean": sum(latencies) / len(latencies),
                             "p50": latencies[math.ceil(.5 * len(latencies)) - 1],
                             "p95": latencies[math.ceil(.95 * len(latencies)) - 1]},
              "metrics": summary,
              "limitations": ["BM25 only; no dense/rerank/answer quality claim",
                              "Index built once; query includes tokenization, scoring, full stable sort",
                              "Single sequential pass on shared server; no SLA or repeated-run claim",
                              "Azusa lowercase Latin words and Chinese single-character tokenizer",
                              "Binary qrels score>0; unjudged passages treated as nonrelevant"]}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    """Create a new run directory, preserving failed-run status and partial observations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        assert metrics(["a"], {"a", "b"}, 3)["recall"] == .5
        assert abs(metrics(["a"], {"a", "b"}, 3)["ndcg"] - 1 / (1 + 1 / math.log2(3))) < 1e-12
        try:
            metrics(["a", "a"], {"a"}, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate ranking accepted")
        assert load_tokenizer(args.source)("Hello 你好") == ["hello", "你", "好"]
        print("Self-test passed")
        return
    if args.output is None or args.source is None:
        parser.error("--output and --source are required")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args.output, args.source)
    except Exception as error:
        (args.output / "failure.json").write_text(json.dumps({"status": "failed", "error": repr(error)}), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
