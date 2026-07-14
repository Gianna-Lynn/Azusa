"""Search Agent v0 编排：查询规划、多跳检索、证据融合与引用回答。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .schemas import EmbeddingRecord


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
    query_type: str     # 单跳或者多跳
    original_query: str
    rewritten_query: str
    retrieval_queries: list[str]  # 要拆成几条子查询去搜索
    notes: list[str]              # 备注

    def to_dict(self) -> dict:
        """把查询计划转换成可直接写入 JSON 的普通字典。"""
        return asdict(self)


def normalize_query_text(query: str) -> str:
    """
    压缩连续空白并移除常见尾部标点，得到稳定的检索查询文本。
    """
    normalized = re.sub(r"\s+", " ", query).strip()
    return normalized.rstrip(" ?？。")


def classify_query(query: str) -> str:
    """
    使用确定性关键词规则判断查询属于单跳还是多跳。
    """
    lowered = query.lower()
    if any(marker in lowered for marker in MULTI_HOP_MARKERS):
        return "multi_hop"
    return "single_hop"


def split_multi_hop_query(query: str) -> list[str]:
    """
    按显式中英文连接词把复合问题切成适合检索的子查询。
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
    构造 Agent 使用的查询分类、规范化文本和多跳检索计划。
    """
    rewritten_query = normalize_query_text(query)   # 先清理噪音: 空白和标点
    query_type = classify_query(rewritten_query)    # 判断是不是多跳
    notes = []

    if query_type == "multi_hop":                                       # 如果是多跳
        retrieval_queries = split_multi_hop_query(rewritten_query)      # 按照连词断开, 切成多条子查询.
        if len(retrieval_queries) <= 1:
            retrieval_queries = [rewritten_query]
            notes.append("multi_hop_marker_found_but_no_clean_split")   # note中添加: 找到了多跳标记, 但是没有干净的拆分
        else:
            notes.append("compound_query_split_for_multi_hop_retrieval") # note中添加: 复合查询被拆分为多跳检索
    else:                                                               # 如果是单跳
        retrieval_queries = [rewritten_query]
        notes.append("single_hop_query_used_as_rewrite")                # note中添加: 单跳查询被用作重写

    return QueryPlan(
        query_type=query_type,
        original_query=query,
        rewritten_query=rewritten_query,
        retrieval_queries=retrieval_queries,
        notes=notes,
    )


def make_preview(text: str, limit: int = 160) -> str:
    """把证据正文压缩成指定长度的单行预览文本。"""
    clean_text = text.replace("\n", " ").strip()
    if len(clean_text) <= limit:
        return clean_text
    return clean_text[:limit] + "..."


def fuse_evidence(
    hop_results: list[dict],
    top_k: int,
) -> list[dict]:
    """
    合并各 hop 的检索结果，优先重复命中次数，再比较最佳排名。
    """
    fused_by_chunk_id: dict[str, dict] = {}

    for hop_index, hop in enumerate(hop_results, start=1):
        for rank, record in enumerate(hop["records"], start=1):
            # 使用 dict.setdefault: 如果键 record.chunk_id 已存在, 返回其对应的值;
            # 否则将第二个参数作为默认值插入到字典中并返回该默认值。
            # 这里用于合并同一 chunk_id 的记录, 并在第一次遇到时初始化统计字段。
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
            item["hop_indices"].append(hop_index)   # 经过的"跳"的列表.

            source_map = hop.get("source_map")
            if source_map is not None:
                item["retrieval_sources"].add(source_map[rank - 1]["source"])
            # 因为rank从1开始, 而列表下标从0开始, 所以这里是rank - 1
            # 取出当前排名对应记录的source; record.source来自EmbeddingRecord, 表示这条chunk原始来自哪里.
            # 使用add()放入集合 retrieval_sources, 自动去重.

    fused_items = list(fused_by_chunk_id.values())
    fused_items.sort(                   # 排序逻辑:
        key=lambda item: (
            -item["hit_count"],         # 命中次数多的优先. sort默认是升序的.所以为了让高hit_count的在先, 需要取负号
            item["best_rank"],          # 命中次数相同的, 最优排名靠前的优先. best_rank越小越靠前, rank 1 比 rank 5 更好.
            item["record"].chunk_id,    # 命中次数, 命中排名都相同的, chunk_id小的优先.
        )
    )

    # 将融合后的候选条目转换为最终的 evidence 列表。
    # 每个 evidence 包含排名信息、文档/块标识、来源、页码、预览文本与完整文本，
    # 以及融合元数据（命中次数、最佳排名、经过的跳次索引及检索来源集合）。
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
    把融合后的 evidence 字典映射回记录对象，供 Answerer 构造上下文。
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
    执行 Search Agent v0，并返回查询计划、各跳结果、融合证据和可选回答轨迹。
    """
    from .search import search_records_with_sources

    plan = build_query_plan(query)  # 得到Query Plan, 含有多条子查询.
    hop_results = []

    for hop_index, retrieval_query in enumerate(plan.retrieval_queries, start=1): # 对于每一条子查询
        records, source_map = search_records_with_sources(  # 搜索这一条子查询, 得出这一跳的查询结果.
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

    evidence = fuse_evidence(hop_results, top_k=top_k)  # 将多跳检索结果融合成最终的evidence列表, 并排序.
    answer_dict = None
    if with_answer and evidence:    # 如果需要生成答案, 且有evidence, 则调用answer_with_citations生成答案.
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

    if answer_dict is not None: # 如果生成了答案, 则将答案相关信息加入结果字典.
        result["answer"] = answer_dict["answer_text"]
        result["answer_raw"] = answer_dict["answer_raw"]
        result["answer_citations"] = answer_dict["citation_ids"]
        result["answer_evidence"] = answer_dict["evidence_text"]
        result["answer_uncertainty"] = answer_dict["uncertainty"]
        result["citation_validation"] = answer_dict["citation_validation"]
        result["answer_context"] = answer_dict["context"]

    return result
