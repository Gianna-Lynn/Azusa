"""验证真实 BM25 评测、实际重排后端记录及 CLI 的失败和产物边界。"""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from azusa_rag.cli import benchmark_lab as lab
from azusa_rag.cli import eval as evaluation
from azusa_rag.loaders import load_embedding_records
from azusa_rag import search_sparse as sparse
from azusa_rag import search_hybrid as hybrid

FIXTURE = Path(__file__).parent / "fixtures"
EMBEDDINGS = FIXTURE / "retrieval_smoke_embedding.jsonl"
CASES = FIXTURE / "retrieval_smoke_eval.jsonl"


def test_model_and_server_overrides_are_process_local():
    """在独立子进程读取配置，既验证覆盖生效，也避免污染当前进程和系统环境。"""
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); from azusa_rag import config; "
        "assert config.EMBEDDING_MODEL_NAME == 'fixture-embedding'; "
        "assert config.RERANK_MODEL_NAME == 'fixture-reranker'; "
        "assert config.EMBEDDING_DEVICE == config.RERANK_DEVICE == 'cpu'; "
        "assert config.SEARCH_URL == 'http://127.0.0.1:38199/search'"
    )
    subprocess.run([sys.executable, "-B", "-s", "-c", script, str(source_root)],
                   env={**os.environ, "AZUSA_EMBEDDING_MODEL": "fixture-embedding",
                        "AZUSA_RERANK_MODEL": "fixture-reranker", "AZUSA_EMBEDDING_DEVICE": "cpu",
                        "AZUSA_RERANK_DEVICE": "cpu", "AZUSA_SERVER_URL": "http://127.0.0.1:38199/"},
                   check=True, timeout=10)


def test_real_bm25_matrix_retains_raw_rankings_and_provenance():
    """使用固定小语料实际检索，验证 JSON 可重算而非只保留聚合数字。"""
    report = lab.run_matrix([("smoke", EMBEDDINGS, CASES)], ["bm25"], [1, 3])
    assert report["failures"] == []
    assert len(report["rows"]) == len(report["runs"]) == 2
    assert report["provenance"]["source"]["files"]
    assert "latency_policy" in report["provenance"]
    for run in report["runs"]:
        summary = run["summary"]
        assert summary["query_count"] == 3
        assert summary["avg_mrr"] == 1.0
        assert summary["rerank_backends"] == {"not_used": 3}
        assert len(summary["cases"]) == 3
        assert all(case["retrieved_chunk_ids"] for case in summary["cases"])


def test_model_rerank_is_recorded_only_after_valid_scoring(monkeypatch):
    """成功模型排序必须保留真实后端，不依赖请求 mode 推测。"""
    records = load_embedding_records(EMBEDDINGS)[:2]
    monkeypatch.setattr(sparse, "score_query_passage_pairs", lambda *args: [0.1, 0.9])
    trace = {}
    assert sparse.rerank_candidates("q", [0, 1], records, 1, trace=trace) == [1]
    assert trace == {"rerank_backend": "model"}


def test_non_finite_model_scores_trigger_explicit_fallback(monkeypatch):
    """分数个数正确也不代表有效；NaN 必须降级并留下明确原因。"""
    records = load_embedding_records(EMBEDDINGS)[:2]
    monkeypatch.setattr(sparse, "score_query_passage_pairs", lambda *args: [float("nan"), 0.0])
    trace = {}
    with pytest.warns(UserWarning, match="falling back"):
        result = sparse.rerank_candidates("q", [0, 1], records, 1, trace=trace)
    assert len(result) == 1
    assert trace["rerank_backend"] == "rule_fallback"
    assert "finite" in trace["rerank_fallback_reason"]


def test_fallback_is_visible_in_per_query_and_aggregate_trace(monkeypatch):
    """真实门面通过规则回退完成评测时，成绩和每题都不能冒充模型重排。"""
    records = load_embedding_records(EMBEDDINGS)
    cases = evaluation.load_eval_cases(CASES)
    monkeypatch.setattr(hybrid, "embed_query", lambda *args: [1.0, 0.0])
    monkeypatch.setattr(hybrid, "ensure_hnsw_index_loaded", lambda *args: "fixture")
    monkeypatch.setattr(hybrid, "search_core_hnsw", lambda *args, **kwargs: list(range(len(records))))

    def unavailable(*args):
        """模拟确定的模型故障，不加载模型或发起外部请求。"""
        raise RuntimeError("controlled unavailable model")

    monkeypatch.setattr(sparse, "score_query_passage_pairs", unavailable)
    with pytest.warns(UserWarning, match="falling back"):
        summary = evaluation.run_eval_for_mode("hybrid_rerank", cases, records, 3)
    assert summary["rerank_backends"] == {"rule_fallback": 3}
    assert summary["warmup_trace"]["rerank_backend"] == "rule_fallback"
    assert all("controlled unavailable" in case["trace"]["rerank_fallback_reason"] for case in summary["cases"])


@pytest.mark.parametrize("problem", ["duplicate", "missing_gold"])
def test_invalid_corpus_cannot_start_retrieval(monkeypatch, problem):
    """重复身份或金标不在语料中属于输入错误，不允许跑完后生成误导分数。"""
    records = load_embedding_records(EMBEDDINGS)
    cases = evaluation.load_eval_cases(CASES)
    if problem == "duplicate":
        records[1] = replace(records[1], chunk_id=records[0].chunk_id)
    else:
        cases[0] = {**cases[0], "relevant_chunk_ids": ["does-not-exist"]}
    monkeypatch.setattr(evaluation, "search", lambda **kwargs: pytest.fail("must not search"))
    with pytest.raises(ValueError):
        evaluation.run_eval_for_mode("bm25", cases, records, 3)


def test_eval_outputs_use_new_directories_and_preserve_latest(monkeypatch, tmp_path):
    """普通评测的新默认路径也不能覆盖用户保留的 latest 或同名实验。"""
    monkeypatch.setattr(evaluation, "EVAL_RUNS_DIR", tmp_path)
    latest = tmp_path / "latest"
    latest.mkdir()
    sentinel = latest / "eval_result_bm25.json"
    sentinel.write_text("historical", encoding="utf-8")
    first, _ = evaluation.write_eval_outputs("bm25", {"result": 1}, None, False)
    second, _ = evaluation.write_eval_outputs("bm25", {"result": 2}, None, False)
    assert first.parent != second.parent
    assert sentinel.read_text(encoding="utf-8") == "historical"
    with pytest.raises(FileExistsError):
        evaluation.write_eval_outputs("bm25", {"result": 3}, None, False, output_dir=first.parent)
    assert json.loads(first.read_text(encoding="utf-8")) == {"result": 1}


def test_benchmark_cli_saves_failures_and_exits_nonzero(monkeypatch, tmp_path):
    """CLI 输入缺失时照样交付失败清单，且退出码不能伪装成成功。"""
    output = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", ["benchmark-lab", "--variant", f"bad={tmp_path / 'missing'},{CASES}",
                                      "--modes", "bm25", "--top-k", "1,3", "--output-dir", str(output)])
    with pytest.raises(SystemExit) as caught:
        lab.main()
    assert caught.value.code == 1
    report = json.loads((output / "benchmark.json").read_text(encoding="utf-8"))
    assert len(report["failures"]) == 2
    assert report["rows"] == []
    assert (output / "benchmark.csv").exists()
    assert (output / "benchmark.md").exists()


def test_eval_all_modes_keeps_success_and_reports_failed_mode(monkeypatch, tmp_path):
    """模式故障不抹掉已完成模式；总报告列出失败并返回非零退出码。"""
    original = evaluation.run_eval_for_mode

    def run(mode, **kwargs):
        """BM25 真实执行，HNSW 用受控故障避免碰到外部服务。"""
        if mode == "hnsw":
            raise RuntimeError("controlled server unavailable")
        return original(mode, **kwargs)

    monkeypatch.setattr(evaluation, "EVAL_MODES", ["bm25", "hnsw"])
    monkeypatch.setattr(evaluation, "run_eval_for_mode", run)
    output = tmp_path / "eval"
    monkeypatch.setattr(sys, "argv", ["eval", "--mode", "all", "--embedding-path", str(EMBEDDINGS),
                                      "--eval-path", str(CASES), "--output-dir", str(output)])
    with pytest.raises(SystemExit) as caught:
        evaluation.main()
    assert caught.value.code == 1
    report = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert [item["mode"] for item in report["completed"]] == ["bm25"]
    assert report["failures"][0]["mode"] == "hnsw"
    assert (output / "eval_result_bm25.json").is_file()
