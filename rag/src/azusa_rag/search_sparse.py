"""
sparse 检索和轻量 rerank 的 helper。

这里放两类逻辑：
1. 最小 BM25 sparse 检索
2. hybrid_rerank 里用到的模型 / 规则 rerank
"""

from __future__ import annotations

import re
import math
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

    # Arguments
    - `query`: 查询文本。

    # Returns
    转成小写并去除首尾空白后的非空行列表。
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

    # Arguments
    - `text`: 待切分的文本。

    # Returns
    转成小写后切分得到的 token 列表。
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

    返回 (bm25_index, tokenized_corpus) 二元组：
    - bm25_index：基于下方语料构建的 BM25Okapi 索引；
      对它调用 get_scores(query_tokens) 会得到一个和 embedding_records
      长度相同、位置一一对应的分数数组。
    - tokenized_corpus：二维 token 列表，tokenized_corpus[i] 就是
      embedding_records[i].text 经过 tokenize_for_bm25 的结果，
      顺序与 embedding_records 保持一致。
    """
    tokenized_corpus = []   # tokenized_corpus 是一个二维列表, 里面每个元素是一个token列表,
                            # 对应embedding_records里每条记录的文本都被tokenize后得到token列表.
                            # corpus 是语料库的意思。
    for embedding_record in embedding_records:
        tokenized_corpus.append(tokenize_for_bm25(embedding_record.text))
    bm25_index = BM25Okapi(tokenized_corpus)
    return bm25_index, tokenized_corpus


def collect_query_tokens(query: str) -> list[str]:
    """
    给最小规则 rerank 用的轻量 token 列表。

    # Arguments
    - `query`: 查询文本。

    # Returns
    小写查询中长度大于 1 的词字符片段列表。
    """
    query_tokens = re.findall(r"\w+", query.lower())
    # 过滤掉太短的 token，比如 a、l、1，减少一点噪声。
    return [token for token in query_tokens if len(token) > 1]


def rule_rerank_score(query: str, text: str) -> int:
    """
    最小规则 rerank 打分：
    - 优先看整句片段是否命中
    - 再补一个很轻量的 token overlap

    # Arguments
    - `query`: 查询文本；
    - `text`: 待打分的文本。

    # Returns
    整句片段命中、命中片段长度和 token overlap 组合得到的整数分数。
    """
    text_lower = text.lower()
    query_fragments = collect_query_fragments(query) # 按换行拆分查询，并检查整行片段是否出现在候选文本中；
    query_tokens = collect_query_tokens(query)  # 按单词拆分查询，并检查每个单词是否出现在候选文本中。

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
    # 这里的分数设计是：
    # - fragment_hit_score * 1000: 整句片段命中时，分数至少是 1000，确保整句命中优先级最高；
    # - fragment_length_score * 10: 在整句命中的前提下，句子越长，分数越高，说明匹配越具体
    # - token_overlap_score: 在整句命中和长度分数相同的情况下，token 命中数越多，分数越高，说明匹配越丰富

def rerank_candidates(
    query: str,
    candidate_ids: list[int],
    embedding_records: list[EmbeddingRecord],
    top_k: int,
    *,
    trace: dict | None = None,
) -> list[int]:
    """对候选文本重排并返回记录下标。

    模型评分必须数量匹配且有限；失败时显式记录规则回退后端与原因。"""


    try:

        candidate_texts = []
        for candidate_id in candidate_ids:

            candidate_texts.append(embedding_records[candidate_id].text)


        model_scores = score_query_passage_pairs(query, candidate_texts)


        if len(model_scores) != len(candidate_ids):
            raise ValueError("Reranker score count does not match candidate count.")
        if any(not math.isfinite(score) for score in model_scores):
            raise ValueError("Reranker scores must be finite.")


        rerank_items = []
        for original_rank, (candidate_id, rerank_score) in enumerate(zip(candidate_ids, model_scores)):
            rerank_items.append((rerank_score, original_rank, candidate_id))


        rerank_items.sort(key=lambda item: (-item[0], item[1], item[2]))


        if trace is not None:
            trace["rerank_backend"] = "model"
        return [candidate_id for _, _, candidate_id in rerank_items[:top_k]]


    except Exception as error:

        if trace is not None:
            trace["rerank_backend"] = "rule_fallback"
            trace["rerank_fallback_reason"] = f"{type(error).__name__}: {error}"
        warnings.warn(
            f"BGE reranker failed, falling back to rule-based rerank: {error}"
        )


    rerank_items = []
    for original_rank, candidate_id in enumerate(candidate_ids):
        record = embedding_records[candidate_id]
        rerank_score = rule_rerank_score(query, record.text)
        rerank_items.append((rerank_score, original_rank, candidate_id))


    rerank_items.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [candidate_id for _, _, candidate_id in rerank_items[:top_k]]


def search_sparse(
    query: str,
    embedding_records: list[EmbeddingRecord],
    top_k: int = 3,
) -> list[int]:
    """BM25 检索并返回与输入 embedding_records 对应的记录下标。"""
    query_tokens = tokenize_for_bm25(query)
    if not query_tokens or not embedding_records:
        return []

    bm25_index, _ = build_bm25_index(embedding_records)
    score_array = bm25_index.get_scores(query_tokens)


    score_index_list = [(float(score), record_index) for record_index, score in enumerate(score_array)]
    score_index_list.sort(key=lambda item: (-item[0], item[1]))
    return [record_index for _, record_index in score_index_list[:top_k]]
