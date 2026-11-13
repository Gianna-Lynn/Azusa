"""验证矩阵失败保留、输入追溯、三格式报告和拒绝覆盖的交付契约。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import run_benchmark_lab as lab
import pytest


def summary() -> dict:
    """返回固定摘要，用于隔离矩阵组织与检索后端。"""
    return {
        "query_count": 2,
        "avg_hit_at_k": 1.0,
        "avg_recall_at_k": 0.75,
        "avg_mrr": 0.5,
        "avg_ndcg_at_k": 0.625,
        "p50_retrieval_latency_ms": 3.0,
        "p95_retrieval_latency_ms": 7.0,
    }


def test_run_matrix_keeps_successes_and_failures(monkeypatch, tmp_path) -> None:
    """每份输入仅加载一次；一个模式失败不抹掉另一模式的全部 k 结果。"""
    embedding_path = tmp_path / "embedding.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    embedding_path.write_text("fixture embeddings", encoding="utf-8")
    eval_path.write_text("fixture cases", encoding="utf-8")
    load_counts = {"embeddings": 0, "eval": 0}

    def load_embeddings(path: Path) -> list:
        """记录语料加载次数，不启动编码模型。"""
        load_counts["embeddings"] += 1
        return [object()]

    def load_cases(path: Path) -> list:
        """记录金标加载次数，返回单条替身。"""
        load_counts["eval"] += 1
        return [{"query": "q"}]

    def run(mode: str, cases: list, records: list, top_k: int) -> dict:
        """构造一个稳定失败模式，测试完整矩阵分母。"""
        if mode == "broken":
            raise RuntimeError("controlled failure")
        return summary()

    monkeypatch.setattr(lab, "load_embedding_records", load_embeddings)
    monkeypatch.setattr(lab, "load_eval_cases", load_cases)
    monkeypatch.setattr(lab, "run_eval_for_mode", run)

    report = lab.run_matrix(
        [("natural-300-30", embedding_path, eval_path)],
        ["bm25", "broken"],
        [3, 5],
    )

    assert load_counts == {"embeddings": 1, "eval": 1}
    assert len(report["rows"]) == 2
    assert len(report["failures"]) == 2
    assert {row["top_k"] for row in report["rows"]} == {3, 5}
    assert all(item["error"] == "controlled failure" for item in report["failures"])
    assert report["expected_cell_count"] == 4
    assert report["variants"]["natural-300-30"]["embedding"]["sha256"]


def test_write_outputs_delivers_all_formats_and_failures(tmp_path: Path) -> None:
    """三种格式均须保留失败；第二次写入不得覆盖第一次任何一个文件。"""
    row = {
        "variant": "fixed-256-0",
        "mode": "bm25",
        "top_k": 3,
        "query_count": 2,
        "hit_at_k": 1.0,
        "recall_at_k": 0.75,
        "mrr": 0.5,
        "ndcg_at_k": 0.625,
        "p50_ms": 3.0,
        "p95_ms": 7.0,
    }
    report = {
        "schema_version": 1,
        "rows": [row],
        "failures": [{"variant": "fixed-256-0", "mode": "hnsw", "top_k": 3, "error": "offline"}],
    }

    lab.write_outputs(tmp_path, report)

    assert json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8")) == report
    with (tmp_path / "benchmark.csv").open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
        assert rows[0]["variant"] == "fixed-256-0"
        assert [row["status"] for row in rows] == ["success", "failed"]
        assert rows[1]["error"] == "offline"
    markdown = (tmp_path / "benchmark.md").read_text(encoding="utf-8")
    assert "P95 ms" in markdown
    assert "Failed cells" in markdown
    assert "offline" in markdown
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        lab.write_outputs(tmp_path, report)
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}


def test_missing_input_preserves_every_failed_coordinate(tmp_path):
    """variant 加载失败也必须覆盖所有 mode×k，而不是提前中断或缩小分母。"""
    report = lab.run_matrix([("missing", tmp_path / "missing.jsonl", tmp_path / "eval.jsonl")],
                            ["bm25", "hnsw"], [1, 3])
    assert report["rows"] == []
    assert len(report["failures"]) == report["expected_cell_count"] == 4
    assert all(item["stage"] == "load" for item in report["failures"])


def test_matrix_rejects_changed_input_instead_of_publishing_success(monkeypatch, tmp_path):
    """执行期间文件被换掉时，不得把旧数据成绩绑定到新的文件身份。"""
    embedding_path = tmp_path / "embedding.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    embedding_path.write_text("before", encoding="utf-8")
    eval_path.write_text("cases", encoding="utf-8")
    monkeypatch.setattr(lab, "load_embedding_records", lambda path: [])
    monkeypatch.setattr(lab, "load_eval_cases", lambda path: [])

    def mutate(*args):
        """返回正常摘要，但同时替换输入，以检验运行后的身份复核。"""
        embedding_path.write_text("after", encoding="utf-8")
        return summary()

    monkeypatch.setattr(lab, "run_eval_for_mode", mutate)
    report = lab.run_matrix([("changed", embedding_path, eval_path)], ["bm25"], [1])
    assert report["rows"] == []
    assert len(report["failures"]) == 1
    assert "changed" in report["failures"][0]["error"]


@pytest.mark.parametrize("modes,ks", [([], [1]), (["bm25"], []), (["bm25", "bm25"], [1]), (["bm25"], [True])])
def test_matrix_rejects_empty_duplicate_or_invalid_axes(modes, ks, tmp_path):
    """非法坐标在读取数据和发起检索前被拒绝。"""
    with pytest.raises(ValueError):
        lab.run_matrix([("fixture", tmp_path / "a", tmp_path / "b")], modes, ks)
