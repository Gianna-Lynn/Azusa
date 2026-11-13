"""核验索引实验的分位数和坏单元保留，不用昂贵数据重跑性能测试。"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("index_decision", Path(__file__).resolve().parents[2]/"scripts/run_index_decision.py")
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)


def test_nearest_rank_does_not_interpolate_small_sample():
    """小样本 P95 是最大观测，不能凭插值伪造不存在的尾延迟。"""
    assert lab.percentile([1, 2, 100], 0.95) == 100
    assert lab.percentile([100, 1, 2], 0.5) == 2
    with pytest.raises(ValueError):
        lab.percentile([], 0.95)


def test_invalid_worker_ranking_remains_failed_cell(monkeypatch, tmp_path):
    """重复 ID 不能放大 Recall；原始响应仍保留用于定位失败。"""
    payload = {"queries": [{"ids": [0, 0], "latency_ms": 1}], "peak_rss_bytes": 1024}
    monkeypatch.setattr(lab.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(payload).encode(), stderr=b""))
    config = {"base": [[1], [2]], "queries": [[1]], "k": 2}
    result = lab.execute(Path("unused-worker"), config, tmp_path, "duplicate")
    assert result["status"] == "failure"
    assert "Invalid ranking" in result["error"]
    assert (tmp_path/"duplicate.stdout").exists()
