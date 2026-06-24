"""
Search Agent v0 orchestration.

This module keeps the first Agent milestone deliberately small:
classify the query, rewrite it into one or more retrieval queries, run
retrieval for each hop, then fuse evidence into a stable result object.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .schemas import EmbeddingRecord
from .search import search_records_with_sources


MULTI_HOP_MARKERS = [
    " and ",
    " or ",
    " both ",
    " compare",
    " difference",
    " relation",
    "relationship",
    "为什么",
    "以及",
    "和",
    "与",
    "对比",
    "比较",
    "关系",
    "分别",
]


@dataclass
class QueryPlan:
    query_type: str
    original_query: str
    rewritten_query: str
    retrieval_queries: list[str]
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_query_text(query: str) -> str:
    """
    Collapse whitespace and trim common trailing punctuation.
    """
    normalized = re.sub(r"\s+", " ", query).strip()
    return normalized.rstrip(" ?？。")


def classify_query(query: str) -> str:
    """
    Minimal deterministic query classifier for Agent v0.
    """
    lowered = query.lower()
    if any(marker in lowered for marker in MULTI_HOP_MARKERS):
        return "multi_hop"
    return "single_hop"


def split_multi_hop_query(query: str) -> list[str]:
    """
    Split obvious compound questions into retrieval-sized subqueries.
    """
    parts = re.split(
        r"\s*(?:\band\b|\bor\b|\bboth\b|以及|并且|同时|和|与|、|；|;)\s*",
        query,
        flags=re.IGNORECASE,
    )
    clean_parts = []
    for part in parts:
        normalized = normalize_query_text(part)
        if normalized and normalized not in clean_parts:
            clean_parts.append(normalized)
    return clean_parts


def build_query_plan(query: str) -> QueryPlan:
    """
    Build the classification + rewrite plan used by the Agent.
    """
    rewritten_query = normalize_query_text(query)
    query_type = classify_query(rewritten_query)
    notes = []

    if query_type == "multi_hop":
        retrieval_queries = split_multi_hop_query(rewritten_query)
        if len(retrieval_queries) <= 1:
            retrieval_queries = [rewritten_query]
            notes.append("multi_hop_marker_found_but_no_clean_split")
        else:
            notes.append("compound_query_split_for_multi_hop_retrieval")
    else:
        retrieval_queries = [rewritten_query]
        notes.append("single_hop_query_used_as_rewrite")

    return QueryPlan(
        query_type=query_type,
        original_query=query,
        rewritten_query=rewritten_query,
        retrieval_queries=retrieval_queries,
        notes=notes,
    )


def make_preview(text: str, limit: int = 160) -> str:
    clean_text = text.replace("\n", " ").strip()
    if len(clean_text) <= limit:
        return clean_text
    return clean_text[:limit] + "..."


def fuse_evidence(
    hop_results: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Fuse per-hop retrieval results by repeated hits first, then best rank.
    """
    fused_by_chunk_id: dict[str, dict] = {}

    for hop_index, hop in enumerate(hop_results, start=1):
        for rank, record in enumerate(hop["records"], start=1):
            item = fused_by_chunk_id.setdefault(
                record.chunk_id,
                {
                    "record": record,
                    "best_rank": rank,
                    "hit_count": 0,
                    "hop_indices": [],
                    "retrieval_sources": set(),
                },
            )
            item["best_rank"] = min(item["best_rank"], rank)
            item["hit_count"] += 1
            item["hop_indices"].append(hop_index)

            source_map = hop.get("source_map")
            if source_map is not None:
                item["retrieval_sources"].add(source_map[rank - 1]["source"])

    fused_items = list(fused_by_chunk_id.values())
    fused_items.sort(
        key=lambda item: (
            -item["hit_count"],
            item["best_rank"],
            item["record"].chunk_id,
        )
    )

    evidence = []
    for rank, item in enumerate(fused_items[:top_k], start=1):
        record = item["record"]
        evidence.append(
            {
                "rank": rank,
                "doc_id": record.doc_id,
                "chunk_id": record.chunk_id,
                "source": record.source,
                "chunk_index": record.chunk_index,
                "page": record.page,
                "preview": make_preview(record.text),
                "full_text": record.text,
                "fusion": {
                    "hit_count": item["hit_count"],
                    "best_rank": item["best_rank"],
                    "hop_indices": item["hop_indices"],
                    "retrieval_sources": sorted(item["retrieval_sources"]),
                },
            }
        )
    return evidence


def evidence_to_records(
    evidence: list[dict],
    embedding_records: list[EmbeddingRecord],
) -> list[EmbeddingRecord]:
    """
    Map fused evidence dictionaries back to records for answer generation.
    """
    record_by_chunk_id = {record.chunk_id: record for record in embedding_records}
    return [record_by_chunk_id[item["chunk_id"]] for item in evidence]


def run_search_agent(
    query: str,
    embedding_records: list[EmbeddingRecord],
    mode: str,
    top_k: int,
    with_answer: bool = False,
) -> dict:
    """
    Execute Search Agent v0 and return a structured trace.
    """
    plan = build_query_plan(query)
    hop_results = []

    for hop_index, retrieval_query in enumerate(plan.retrieval_queries, start=1):
        records, source_map = search_records_with_sources(
            query=retrieval_query,
            embedding_records=embedding_records,
            top_k=top_k,
            mode=mode,
        )
        hop_results.append(
            {
                "hop_index": hop_index,
                "query": retrieval_query,
                "records": records,
                "source_map": source_map,
            }
        )

    evidence = fuse_evidence(hop_results, top_k=top_k)
    answer_dict = None
    if with_answer and evidence:
        from .answerer import answer_with_citations

        answer_dict = answer_with_citations(
            query=plan.rewritten_query,
            result_records=evidence_to_records(evidence, embedding_records),
        )

    result = {
        "agent_version": "search_agent_v0",
        "query_plan": plan.to_dict(),
        "mode": mode,
        "top_k": top_k,
        "hops": [
            {
                "hop_index": hop["hop_index"],
                "query": hop["query"],
                "results": [
                    {
                        "rank": rank,
                        "doc_id": record.doc_id,
                        "chunk_id": record.chunk_id,
                        "source": record.source,
                        "preview": make_preview(record.text),
                    }
                    for rank, record in enumerate(hop["records"], start=1)
                ],
            }
            for hop in hop_results
        ],
        "fused_evidence": evidence,
    }

    if answer_dict is not None:
        result["answer"] = answer_dict["answer_text"]
        result["answer_raw"] = answer_dict["answer_raw"]
        result["answer_citations"] = answer_dict["citation_ids"]
        result["answer_context"] = answer_dict["context"]

    return result
