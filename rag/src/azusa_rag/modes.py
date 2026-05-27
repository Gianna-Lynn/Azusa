"""
集中维护检索 mode 的公共定义。

这样 run_eval.py、run_search.py、run_workflow.py 就不用各自维护一份：
- mode 列表
- 展示顺序
- mode 的说明文字
"""

from __future__ import annotations

SEARCH_MODES = ["brute", "hnsw", "hybrid", "hybrid_rerank", "bm25"]
MODE_DISPLAY_ORDER = ["brute", "hnsw", "bm25", "hybrid", "hybrid_rerank"]
MODE_ORDER_INDEX = {mode: index for index, mode in enumerate(MODE_DISPLAY_ORDER)}

MODE_METADATA = {
    "brute": {
        "family": "dense-only",
        "label": "Brute Force Dense",
        "notes": "直接用向量相似度暴力搜索, 作为最朴素的 dense baseline.",
    },
    "hnsw": {
        "family": "dense-only",
        "label": "HNSW Dense",
        "notes": "通过 Rust HNSW 服务执行近似向量检索, 用来观察 ANN 近似检索是否保持效果.",
    },
    "bm25": {
        "family": "sparse-only",
        "label": "BM25 Sparse",
        "notes": "只依赖词面匹配的 BM25 baseline, 适合观察专有名词、年份和字面命中收益.",
    },
    "hybrid": {
        "family": "dense+sparse",
        "label": "Hybrid",
        "notes": "先召回 dense 和 BM25 sparse, 再按当前 merge 策略合并去重.",
    },
    "hybrid_rerank": {
        "family": "dense+sparse+rerank",
        "label": "Hybrid Rerank",
        "notes": "在 hybrid 候选集基础上再做 rerank, 用来观察排序纠偏收益.",
    },
}


def get_mode_order_key(mode: str) -> tuple[int, str]:
    """
    返回 mode 的稳定展示顺序。
    未登记的 mode 仍然能安全排到最后。
    """
    return (MODE_ORDER_INDEX.get(mode, len(MODE_DISPLAY_ORDER)), mode)
