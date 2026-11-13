"""
提供最小 reranker 能力 (cross-encoder 重排):
1. 延迟加载 BGE reranker (tokenizer + sequence classification 模型),
   解析设备并在模块级缓存复用, 避免重复初始化.
2. 对 (query, passage) 对按 batch 打分, 分数越高表示越相关.
3. 在加载或推理失败时把异常抛给调用方, 由 search.py 决定是否 fallback.
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
    按配置解析 reranker 实际使用的设备.

    Args:
        requested_device: 配置中声明的目标设备, 例如 "cuda" 或 "cpu".

    Returns:
        解析后的设备字符串. 请求 cuda 但当前机器不可用时回退为 "cpu"
        并发出 UserWarning; 其余情况原样返回.

    Note:
        解析结果同时决定模型加载时的 dtype: cuda 上使用 float16 节省显存,
        cpu 上使用 float32 保证兼容性.
    """
    if requested_device.startswith("cuda"):
        if torch.cuda.is_available():
            return requested_device

        warnings.warn("CUDA is not available for reranker, falling back to CPU.")
        return "cpu"

    return requested_device


def _load_reranker_components() -> tuple:
    """
    延迟加载 reranker 的 tokenizer / model / device, 并缓存复用.

    首次调用时按 RERANK_MODEL_NAME 从本地路径加载 tokenizer 和
    sequence classification 模型, 解析设备后把模型搬到目标设备并置为
    eval 模式; 三者缓存在模块级变量中, 后续调用直接复用, 不必每次重新初始化.

    Returns:
        (tokenizer, model, model_device) 三元组.

    Raises:
        RuntimeError: reranker 被配置禁用 (RERANK_ENABLED 为 False),
            或 transformers 未安装.
        OSError 等加载异常: 模型路径不存在或加载失败时原样向上抛出.
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
    # 输入与模型使用同一设备。
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

    把同一 query 与 passage_list 中每个 passage 组成 (query, passage) 对,
    按 batch 送入 BGE reranker, 返回与 passage_list 一一对应的相关性分数;
    分数越高表示 query 与 passage 越相关.

    Args:
        query: 检索查询文本, 与每个 passage 分别配对打分.
        passage_list: 候选段落文本列表, 允许为空.

    Returns:
        与 passage_list 一一对应的 float 分数列表; 空输入返回空列表.

    Note:
        - 按 RERANK_BATCH_SIZE 分批推理, 批内统一 padding 并截断到
          RERANK_MAX_LENGTH, 输入张量自动搬到模型所在设备.
        - 分数是模型的原始 logits; 推理在 torch.no_grad 下进行, 不产生梯度.
        - 模型加载或推理失败时异常向上抛出, 由 search.py 决定是否 fallback.
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
                for key, value in batch_dict.items() # 这种写法是“边遍历边生成新字典”的简写
            }

            # SequenceClassification 模型输出 logits;
            # 分数顺序与输入 passage 一致。
            # 将输入字典展开为模型参数，并以带字段名的结果对象返回输出。
            # logits 是模型的原始分数，通常分数越高表示 query 与 passage 越相关。
            # view(-1) 将 logits 展平成一维，便于按 passage 顺序收集分数。
            logits = model(**batch_dict, return_dict=True).logits.view(-1)

            # 转为 CPU 上的 Python 浮点数后，逐个追加到结果列表，避免生成嵌套列表。
            score_list.extend(logits.float().cpu().tolist())
    return score_list
