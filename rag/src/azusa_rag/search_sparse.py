"""
Sparse retrieval and lightweight rerank helpers.
"""

from __future__ import annotations

import re
import warnings

from rank_bm25 import BM25Okapi

from .reranker import score_query_passage_pairs
from .schemas import EmbeddingRecord

BM25_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def collect_query_fragments(query: str) -> list[str]:
    """
    Split a multiline query into non-empty lowercase fragments.
    """
    query_fragments = []
    for line in query.lower().splitlines():
        line = line.strip()
        if line:
            query_fragments.append(line)
    return query_fragments


def tokenize_for_bm25(text: str) -> list[str]:
    """
    Tokenize mixed Chinese / English text for the minimal BM25 baseline.
    """
    token_list = BM25_TOKEN_PATTERN.findall(text.lower())
    return [token for token in token_list if token.strip()]


def build_bm25_index(
    embedding_records: list[EmbeddingRecord],
) -> tuple[BM25Okapi, list[list[str]]]:
    """
    Build a fresh BM25 index from the current embedding corpus.
    """
    tokenized_corpus = []
    for embedding_record in embedding_records:
        tokenized_corpus.append(tokenize_for_bm25(embedding_record.text))
    bm25_index = BM25Okapi(tokenized_corpus)
    return bm25_index, tokenized_corpus


def collect_query_tokens(query: str) -> list[str]:
    """
    Produce lightweight overlap tokens for the rule-based rerank fallback.
    """
    query_tokens = re.findall(r"\w+", query.lower())
    return [token for token in query_tokens if len(token) > 1]


def rule_rerank_score(query: str, text: str) -> int:
    """
    Score one candidate using exact-fragment hits plus rough token overlap.
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

    token_overlap_score = 0
    for query_token in query_tokens:
        if query_token in text_lower:
            token_overlap_score += 1

    return fragment_hit_score * 1000 + fragment_length_score * 10 + token_overlap_score


def rerank_candidates(
    query: str,
    candidate_ids: list[int],
    embedding_records: list[EmbeddingRecord],
    top_k: int,
) -> list[int]:
    """
    Re-rank a candidate pool using the model reranker, then fallback to rules.
    """
    try:
        candidate_texts = []
        for candidate_id in candidate_ids:
            candidate_texts.append(embedding_records[candidate_id].text)

        model_scores = score_query_passage_pairs(query, candidate_texts)
        if len(model_scores) != len(candidate_ids):
            raise ValueError("Reranker score count does not match candidate count.")

        rerank_items = []
        for original_rank, (candidate_id, rerank_score) in enumerate(zip(candidate_ids, model_scores)):
            rerank_items.append((rerank_score, original_rank, candidate_id))

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

    rerank_items.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [candidate_id for _, _, candidate_id in rerank_items[:top_k]]


def search_sparse(
    query: str,
    embedding_records: list[EmbeddingRecord],
    top_k: int = 3,
) -> list[int]:
    """
    Run the minimal BM25 sparse retrieval flow and return record ids.
    """
    query_tokens = tokenize_for_bm25(query)
    if not query_tokens:
        return []

    bm25_index, _ = build_bm25_index(embedding_records)
    score_array = bm25_index.get_scores(query_tokens)
    score_id_list = [(float(score), record_id) for record_id, score in enumerate(score_array)]
    score_id_list.sort(key=lambda item: (-item[0], item[1]))
    return [record_id for _, record_id in score_id_list[:top_k]]
