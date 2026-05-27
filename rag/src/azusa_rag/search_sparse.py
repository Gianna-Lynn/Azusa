"""
sparse 检索和轻量 rerank 的 helper。

这里放两类逻辑：
1. 最小 BM25 sparse 检索
2. hybrid_rerank 里用到的模型 / 规则 rerank
"""

from __future__ import annotations

import re
import warnings

from rank_bm25 import BM25Okapi

from .reranker import score_query_passage_pairs
from .schemas import EmbeddingRecord

BM25_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
# [a-z0-9]+：匹配一段连续的英文小写字母或数字，至少 1 个
# |：表示“或者”
# [\u4e00-\u9fff]：匹配一个中文汉字字符
# 例如 "Hello, Azusa! 123 你好" 会被切成 ["hello", "azusa", "123", "你", "好"]


def collect_query_fragments(query: str) -> list[str]:
    """
    把 query 按行切开，保留非空片段，供 sparse / rerank 共用。
    """
    query_fragments = []
    for line in query.lower().splitlines():
        line = line.strip()
        if line:
            query_fragments.append(line)
    return query_fragments


def tokenize_for_bm25(text: str) -> list[str]:
    """
    为最小 BM25 接入准备的轻量 tokenizer。
    - 英文和数字按连续片段切分
    - 中文按单字切分，这样不用额外分词库也能先跑起来
    """
    text_lower = text.lower()
    token_list = BM25_TOKEN_PATTERN.findall(text_lower)
    return [token for token in token_list if token.strip()]


def build_bm25_index(
    embedding_records: list[EmbeddingRecord],
) -> tuple[BM25Okapi, list[list[str]]]:
    """
    从当前 embedding_records 构建一个最小 BM25 索引。
    当前没有实现缓存，优先保证接通和结果可见。
    """
    tokenized_corpus = []
    for embedding_record in embedding_records:
        # corpus 是语料库的意思。
        # 这里每条 EmbeddingRecord 都会被映射成一个 token 列表。
        tokenized_corpus.append(tokenize_for_bm25(embedding_record.text))
    bm25_index = BM25Okapi(tokenized_corpus)
    return bm25_index, tokenized_corpus


def collect_query_tokens(query: str) -> list[str]:
    """
    给最小规则 rerank 用的轻量 token 列表。
    """
    query_tokens = re.findall(r"\w+", query.lower())
    # 过滤掉太短的 token，比如 a、l、1，减少一点噪声。
    return [token for token in query_tokens if len(token) > 1]


def rule_rerank_score(query: str, text: str) -> int:
    """
    最小规则 rerank 打分：
    - 优先看整句片段是否命中
    - 再补一个很轻量的 token overlap
    """
    text_lower = text.lower()
    query_fragments = collect_query_fragments(query)
    query_tokens = collect_query_tokens(query)

    fragment_hit_score = 0
    fragment_length_score = 0
    for query_fragment in query_fragments:
        if query_fragment in text_lower:
            fragment_hit_score += 1
            fragment_length_score += len(query_fragment)
            # 整句命中的前提下，句子越长，通常说明匹配越具体。

    token_overlap_score = 0
    for query_token in query_tokens:
        if query_token in text_lower:
            token_overlap_score += 1  # token 命中时，每个 token 加 1 分。

    return fragment_hit_score * 1000 + fragment_length_score * 10 + token_overlap_score


def rerank_candidates(
    query: str,
    candidate_ids: list[int],
    embedding_records: list[EmbeddingRecord],
    top_k: int,
) -> list[int]:
    """
    对候选集合做一次 rerank。
    优先尝试模型 reranker；如果模型不可用或推理失败，再回退到规则 rerank。
    同分时尽量保持原始召回顺序稳定。
    """
    try:
        candidate_texts = []
        for candidate_id in candidate_ids:
            # 这里把候选 id 映射回原文文本，再交给 cross-encoder 做逐对打分。
            candidate_texts.append(embedding_records[candidate_id].text)

        model_scores = score_query_passage_pairs(query, candidate_texts)
        if len(model_scores) != len(candidate_ids):
            raise ValueError("Reranker score count does not match candidate count.")

        rerank_items = []
        for original_rank, (candidate_id, rerank_score) in enumerate(zip(candidate_ids, model_scores)):
            rerank_items.append((rerank_score, original_rank, candidate_id))

        # 模型分数优先；如果分数相同，仍尽量保留原始召回顺序，避免结果抖动。
        rerank_items.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [candidate_id for _, _, candidate_id in rerank_items[:top_k]]
    except Exception as error:
        warnings.warn(
            f"BGE reranker failed, falling back to rule-based rerank: {error}"
        )

    rerank_items = []
    for original_rank, candidate_id in enumerate(candidate_ids):
        record = embedding_records[candidate_id]
        rerank_score = rule_rerank_score(query, record.text)
        rerank_items.append((rerank_score, original_rank, candidate_id))

    # fallback 的设计目标不是追求高精度，而是保证 hybrid_rerank 模式始终可用。
    rerank_items.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [candidate_id for _, _, candidate_id in rerank_items[:top_k]]


def search_sparse(
    query: str,
    embedding_records: list[EmbeddingRecord],
    top_k: int = 3,
) -> list[int]:
    """
    最小 BM25 sparse 检索内核。
    在 Python 侧临时构建 BM25 索引，返回 top-k id。
    """
    query_tokens = tokenize_for_bm25(query)
    if not query_tokens:
        return []

    bm25_index, _ = build_bm25_index(embedding_records)
    score_array = bm25_index.get_scores(query_tokens)
    # 这里的 score_array 和 embedding_records 一一对应：
    # 第 i 个分数，对应第 i 条记录。
    score_id_list = [(float(score), record_id) for record_id, score in enumerate(score_array)]
    score_id_list.sort(key=lambda item: (-item[0], item[1]))
    return [record_id for _, record_id in score_id_list[:top_k]]
