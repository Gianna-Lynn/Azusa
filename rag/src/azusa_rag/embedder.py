"""
接收ChunkRecord, 进行编码, 返回EmbeddingRecord
"""
from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    EMBEDDING_DIM,
    EMBEDDING_MAX_LENGTH,
    EMBEDDING_MODEL_NAME,
)
from .schemas import ChunkRecord, EmbeddingRecord

_TOKENIZER = None
_MODEL = None
_MODEL_DEVICE = None


def _resolve_device(requested_device: str) -> str:
    """
    按配置选择设备. 如果请求cuda但当前不可用, 自动回退到cpu.
    """
    if requested_device.startswith("cuda"):
        if torch.cuda.is_available():
            return requested_device

        warnings.warn("CUDA is not available, falling back to CPU.")
        return "cpu"

    return requested_device


def _load_embedding_components() -> tuple:
    """
    延迟加载tokenizer和model, 避免每个chunk重复初始化模型.
    """
    global _TOKENIZER, _MODEL, _MODEL_DEVICE

    if _TOKENIZER is not None and _MODEL is not None and _MODEL_DEVICE is not None:
        return _TOKENIZER, _MODEL, _MODEL_DEVICE

    model_device = _resolve_device(EMBEDDING_DEVICE)
    model_dtype = torch.float16 if model_device.startswith("cuda") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL_NAME,
        padding_side="left",
    )
    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL_NAME,
        dtype=model_dtype,
    )
    model = model.to(model_device)
    model.eval()

    _TOKENIZER = tokenizer
    _MODEL = model
    _MODEL_DEVICE = model_device
    return _TOKENIZER, _MODEL, _MODEL_DEVICE


def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    使用Qwen官方推荐的last-token pooling方式提取embedding.
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])

    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    batch_index = torch.arange(batch_size, device=last_hidden_states.device)
    return last_hidden_states[batch_index, sequence_lengths]


def embed_text_list(text_list: list[str]) -> list[list[float]]:
    """
    把一批文本编码成向量列表.
    """
    if not text_list:
        return []

    # 这里拿到的是“已经准备好的编码器组件”:
    # tokenizer负责把字符串变成token id,
    # model负责把token序列变成hidden states,
    # model_device表示这次推理实际跑在cpu还是cuda上.
    tokenizer, model, model_device = _load_embedding_components()

    embedding_list: list[list[float]] = []

    # 这里只做推理, 不做训练, 所以用no_grad()关闭梯度计算, 节省显存和时间.
    with torch.no_grad():
        # 按批次处理文本, 避免一次性把所有文本都塞进模型导致显存/内存压力过大.
        for start in range(0, len(text_list), EMBEDDING_BATCH_SIZE):
            batch_text_list = text_list[start : start + EMBEDDING_BATCH_SIZE]

            # tokenizer会把一批字符串转成模型可接收的张量字典.
            # 常见键包括:
            # - input_ids: 每个token对应的整数编号
            # - attention_mask: 哪些位置是真实文本, 哪些位置只是padding补位
            batch_dict = tokenizer(
                batch_text_list,
                padding=True,                   # 把这一批文本补齐到相同长度, 方便一起送进模型
                truncation=True,                # 如果文本太长, 就截断到max_length
                max_length=EMBEDDING_MAX_LENGTH,# 限制单条文本最多使用多少token
                return_tensors="pt",            # 直接返回PyTorch张量, 而不是普通Python列表
            )

            # tokenizer刚生成的张量默认在CPU上.
            # 如果模型跑在cuda, 输入也必须搬到同一个device上, 否则会报错.
            batch_dict = {
                key: value.to(model_device)
                for key, value in batch_dict.items()
            }

            # **的意思是“把字典摊开传参”.
            # 这里等价于 model(input_ids=..., attention_mask=..., ...)
            # outputs.last_hidden_state 的shape通常是:
            # (batch_size, sequence_length, hidden_size)
            # 也就是: 一批文本里, 每个token都有一个hidden vector.
            outputs = model(**batch_dict)

            # 但我们最终需要的是“每段文本一个向量”, 不是“每个token一个向量”.
            # 所以这里用Qwen官方推荐的last-token pooling,
            # 把token级表示压成句子/段落级表示.
            embeddings = _last_token_pool(
                outputs.last_hidden_state,
                batch_dict["attention_mask"],
            )   # 此时 embeddings 的shape是 (batch_size, hidden_size)

            hidden_size = embeddings.shape[1]
            if EMBEDDING_DIM > hidden_size:
                raise ValueError(
                    f"EMBEDDING_DIM={EMBEDDING_DIM} exceeds model hidden size {hidden_size}"
                )

            # 当前模型原生输出是1024维.
            # 这里允许按配置只取前EMBEDDING_DIM维, 方便控制存储成本和后续实验.
            embeddings = embeddings[:, :EMBEDDING_DIM]

            # L2归一化后, 每个向量的长度都会变成1.
            # 这样后面做余弦相似度检索时, 比较的重点就是“方向是否接近”.
            # 这里也顺手统一一下项目口径: SIFT1M 这类benchmark主线偏L2, 当前RAG文本检索先按更常见的 cosine 路线走.
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # 最后把这一批embedding搬回CPU, 转成普通Python列表, 方便写入jsonl.
            embedding_list.extend(embeddings.cpu().tolist())

    return embedding_list


def chunk_to_embedding(chunk_record: ChunkRecord) -> EmbeddingRecord:
    """
    把单个ChunkRecord变成单个EmbeddingRecord
    """
    embedding = embed_text_list([chunk_record.text])[0]

    embedding_record = EmbeddingRecord(
        doc_id=chunk_record.doc_id,
        chunk_id=chunk_record.chunk_id,
        chunk_index=chunk_record.chunk_index,
        source=chunk_record.source,
        embedding=embedding,
        page=chunk_record.page,
        text=chunk_record.text,
    )

    return embedding_record


def chunk_to_embedding_batch(chunk_record_list: list[ChunkRecord]) -> list[EmbeddingRecord]:
    """
    把一批ChunkRecord变成一批EmbeddingRecord
    """
    text_list = [chunk_record.text for chunk_record in chunk_record_list]
    embedding_list = embed_text_list(text_list)

    embedding_record_list = []

    for chunk_record, embedding in zip(chunk_record_list, embedding_list):
        embedding_record = EmbeddingRecord(
            doc_id=chunk_record.doc_id,
            chunk_id=chunk_record.chunk_id,
            chunk_index=chunk_record.chunk_index,
            source=chunk_record.source,
            embedding=embedding,
            page=chunk_record.page,
            text=chunk_record.text,
        )
        embedding_record_list.append(embedding_record)

    return embedding_record_list
