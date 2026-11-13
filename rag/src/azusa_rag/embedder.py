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
    """提取每条文本最后一个有效 token 的 hidden state，兼容左右 padding。

    左 padding 直接取末列；右 padding 按 attention_mask 的有效长度定位。"""


    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])

    if left_padding:


        return last_hidden_states[:, -1]


    sequence_lengths = attention_mask.sum(dim=1) - 1


    batch_size = last_hidden_states.shape[0]
    batch_index = torch.arange(batch_size, device=last_hidden_states.device)


    return last_hidden_states[batch_index, sequence_lengths]


def embed_text_list(text_list: list[str]) -> list[list[float]]:
    """按输入顺序分批编码；last-token pooling 后截取配置维度并做 L2 归一化。

    输入按最大 token 长度截断，张量送至模型实际设备；输出为 CPU 浮点列表。空输入返回空列表。"""
    if not text_list:
        return []


    tokenizer, model, model_device = _load_embedding_components()

    embedding_list: list[list[float]] = []


    with torch.no_grad():

        for start in range(0, len(text_list), EMBEDDING_BATCH_SIZE):
            batch_text_list = text_list[start : start + EMBEDDING_BATCH_SIZE]


            batch_dict = tokenizer(
                batch_text_list,
                padding=True,
                truncation=True,
                max_length=EMBEDDING_MAX_LENGTH,
                return_tensors="pt",
            )


            batch_dict = {
                key: value.to(model_device)
                for key, value in batch_dict.items()
            }


            outputs = model(**batch_dict)


            embeddings = _last_token_pool(
                outputs.last_hidden_state,
                batch_dict["attention_mask"],
            )

            hidden_size = embeddings.shape[1]
            if EMBEDDING_DIM > hidden_size:
                raise ValueError(
                    f"EMBEDDING_DIM={EMBEDDING_DIM} exceeds model hidden size {hidden_size}"
                )


            embeddings = embeddings[:, :EMBEDDING_DIM]


            embeddings = F.normalize(embeddings, p=2, dim=1)


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
        document_version=chunk_record.document_version,
        content_hash=chunk_record.content_hash,
        start_char=chunk_record.start_char,
        end_char=chunk_record.end_char,
        section_path=list(chunk_record.section_path),
        metadata=dict(chunk_record.metadata),
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
            document_version=chunk_record.document_version,
            content_hash=chunk_record.content_hash,
            start_char=chunk_record.start_char,
            end_char=chunk_record.end_char,
            section_path=list(chunk_record.section_path),
            metadata=dict(chunk_record.metadata),
        )
        embedding_record_list.append(embedding_record)

    return embedding_record_list
