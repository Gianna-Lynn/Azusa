"""
提供最小 reranker 能力:
1. 延迟加载 BGE reranker
2. 对 (query, passage) 对打分
3. 在加载或推理失败时把异常抛给调用方, 由 search.py 决定是否 fallback
"""
from __future__ import annotations

import warnings

import torch
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _TRANSFORMERS_IMPORT_ERROR = None
except ImportError as error:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    _TRANSFORMERS_IMPORT_ERROR = error

from .config import (
    RERANK_BATCH_SIZE,
    RERANK_DEVICE,
    RERANK_ENABLED,
    RERANK_MAX_LENGTH,
    RERANK_MODEL_NAME,
)

_TOKENIZER = None
_MODEL = None
_MODEL_DEVICE = None


def _resolve_device(requested_device: str) -> str:
    """
    按配置选择 reranker 设备.
    如果请求 cuda 但当前机器不可用, 自动回退到 cpu.
    """
    if requested_device.startswith("cuda"):
        if torch.cuda.is_available():
            return requested_device

        warnings.warn("CUDA is not available for reranker, falling back to CPU.")
        return "cpu"

    return requested_device


def _load_reranker_components() -> tuple:
    """
    延迟加载 reranker 的 tokenizer / model / device.
    这样搜索过程中可以复用同一份模型实例, 不必每次重新初始化.
    """
    global _TOKENIZER, _MODEL, _MODEL_DEVICE

    if not RERANK_ENABLED:
        raise RuntimeError("Reranker is disabled by config.")

    if _TRANSFORMERS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "transformers is not available for reranker."
        ) from _TRANSFORMERS_IMPORT_ERROR

    if _TOKENIZER is not None and _MODEL is not None and _MODEL_DEVICE is not None:
        return _TOKENIZER, _MODEL, _MODEL_DEVICE

    model_device = _resolve_device(RERANK_DEVICE)
    # GPU 上优先用 float16 减少显存占用, CPU 上保持 float32 兼容性更稳.
    model_dtype = torch.float16 if model_device.startswith("cuda") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANK_MODEL_NAME,
        dtype=model_dtype,
    )
    # 这里把模型显式搬到目标设备, 后面每个 batch 的输入也会送到同一设备上.
    model = model.to(model_device)
    model.eval()

    # 缓存在模块级变量里, 后续 search.py 多次调用时就能直接复用.
    _TOKENIZER = tokenizer
    _MODEL = model
    _MODEL_DEVICE = model_device
    return _TOKENIZER, _MODEL, _MODEL_DEVICE


def score_query_passage_pairs(query: str, passage_list: list[str]) -> list[float]:
    """
    对一批 (query, passage) 对计算 reranker 相关性分数.
    返回值与 passage_list 一一对应, 分数越高表示越相关.
    """
    if not passage_list:
        return []

    tokenizer, model, model_device = _load_reranker_components()
    score_list: list[float] = []

    with torch.no_grad():
        for start in range(0, len(passage_list), RERANK_BATCH_SIZE):
            batch_passages = passage_list[start : start + RERANK_BATCH_SIZE]
            # BGE reranker 期望的输入结构是 [query, passage] 成对出现.
            batch_pairs = [[query, passage] for passage in batch_passages]

            batch_dict = tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=RERANK_MAX_LENGTH,
                return_tensors="pt",
            )
            # tokenizer 产出的每个张量都要搬到模型所在设备上, 否则前向会报 device mismatch.
            batch_dict = {
                key: value.to(model_device)
                for key, value in batch_dict.items()
            }

            # SequenceClassification 模型输出 logits;
            # 这里把它拉平成一维分数列表, 保持和输入 passage 顺序一致.
            logits = model(**batch_dict, return_dict=True).logits.view(-1)
            score_list.extend(logits.float().cpu().tolist())

    return score_list
