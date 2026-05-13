"""
把 Hugging Face 上的小型 QA 数据集样本转成 Azusa RAG 当前可用的:
1. data/raw/*.txt
2. data/eval/*.jsonl

默认走 Hugging Face datasets-server 的 rows API, 这样不依赖 datasets 包.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlencode
import urllib.request

from .chunker import chunk_text_with_spans
from .config import CHUNK_OVERLAP, CHUNK_SIZE, PROJECT_ROOT, QUERY_DIR, RAW_DATA_DIR


DATASET_ROWS_API = "https://datasets-server.huggingface.co/rows"


@dataclass
class ImportedEvalCase:
    """
    表示一条可直接写入 eval jsonl 的评测样本.
    """
    query_id: str
    query: str
    relevant_chunk_ids: list[str]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        """
        把 dataclass 转成普通字典, 便于后续 json 序列化.
        """
        return {
            "query_id": self.query_id,
            "query": self.query,
            "relevant_chunk_ids": self.relevant_chunk_ids,
            "notes": self.notes,
        }


def sanitize_slug(text: str) -> str:
    """
    把任意文本清洗成适合放进文件名 / doc_id 的安全片段.
    """
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    if not slug:
        return "item"
    return slug


def fetch_dataset_rows(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
) -> list[dict[str, Any]]:
    """
    从 Hugging Face datasets-server 拉取一小段数据行.
    """
    query_string = urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    request = urllib.request.Request(
        url=f"{DATASET_ROWS_API}?{query_string}",
        headers={"User-Agent": "Azusa-RAG-Dataset-Importer/0.1"},
        method="GET",
    )
    with urllib.request.urlopen(request) as response:
        response_dict = json.loads(response.read().decode("utf-8"))

    rows = []
    for row_item in response_dict["rows"]:
        # datasets-server 返回的每个元素外层还包了一层 row metadata,
        # 这里统一只取真正的数据本体.
        rows.append(row_item["row"])
    return rows


def build_doc_id(dataset_slug: str, split: str, row_index: int, row_id: str) -> str:
    """
    为导入后的原始文档生成稳定 doc_id.
    """
    row_slug = sanitize_slug(row_id)
    return f"{dataset_slug}_{split}_{row_index:04d}_{row_slug}"


def build_relevant_chunk_ids(
    doc_id: str,
    context: str,
    answer_starts: list[int],
    answer_texts: list[str],
) -> list[str]:
    """
    根据答案字符区间, 推导这条样本命中了哪些 chunk_id.
    """
    chunk_items = chunk_text_with_spans(
        text=context,
        chunk_size=CHUNK_SIZE,
        overlap_size=CHUNK_OVERLAP,
    )
    relevant_chunk_ids = []

    for chunk_index, (_, chunk_start, chunk_end) in enumerate(chunk_items):
        for answer_start, answer_text in zip(answer_starts, answer_texts):
            answer_end = answer_start + len(answer_text)
            # 这里只要 answer span 和 chunk span 有交集,
            # 就把这个 chunk 视为 relevant chunk.
            if answer_start < chunk_end and answer_end > chunk_start:
                chunk_id = f"{doc_id}_chunk_{chunk_index:03d}"
                if chunk_id not in relevant_chunk_ids:
                    relevant_chunk_ids.append(chunk_id)
                break

    if not relevant_chunk_ids:
        raise ValueError(
            f"Failed to map answer spans to chunk ids for doc_id={doc_id}."
        )

    return relevant_chunk_ids


def write_text_file(file_path: Path, text: str) -> None:
    """
    把一篇 context 文本写入 raw 目录中的 txt 文件.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(text, encoding="utf-8")


def write_eval_jsonl(eval_path: Path, eval_cases: list[ImportedEvalCase]) -> None:
    """
    把评测样本列表写成当前项目使用的 jsonl 格式.
    """
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_path.open("w", encoding="utf-8") as output_file:
        for eval_case in eval_cases:
            output_file.write(json.dumps(eval_case.to_dict(), ensure_ascii=False))
            output_file.write("\n")


def write_query_samples(query_dir: Path, eval_cases: list[ImportedEvalCase], prefix: str, limit: int) -> None:
    """
    从评测集里抽几条 query 单独写到 queries 目录, 方便手工检索调试.
    """
    query_dir.mkdir(parents=True, exist_ok=True)
    for index, eval_case in enumerate(eval_cases[:limit], start=1):
        query_path = query_dir / f"{prefix}_query_{index:03d}.txt"
        query_path.write_text(eval_case.query, encoding="utf-8")


def convert_squad_rows_to_eval_cases(
    rows: list[dict[str, Any]],
    dataset_slug: str,
    split: str,
    raw_dir: Path,
) -> list[ImportedEvalCase]:
    """
    把 SQuAD 风格的数据行转成 raw txt + eval case 结构.
    """
    eval_cases = []

    for row_index, row in enumerate(rows):
        question = row["question"].strip()
        context = row["context"]
        answers = row["answers"]
        answer_starts = answers["answer_start"]
        answer_texts = answers["text"]

        if not question or not context.strip():
            continue
        if not answer_starts or not answer_texts:
            continue

        # 保持 context 原样写盘, 这样 answer_start 这种字符偏移才不会错位.
        row_id = str(row.get("id", row_index))
        title = str(row.get("title", "untitled"))
        doc_id = build_doc_id(dataset_slug, split, row_index, row_id)
        raw_path = raw_dir / f"{doc_id}.txt"
        write_text_file(raw_path, context)

        relevant_chunk_ids = build_relevant_chunk_ids(
            doc_id=doc_id,
            context=context,
            answer_starts=answer_starts,
            answer_texts=answer_texts,
        )
        answer_preview = " | ".join(answer_texts[:2])
        eval_cases.append(
            ImportedEvalCase(
                query_id=f"{doc_id}_query",
                query=question,
                relevant_chunk_ids=relevant_chunk_ids,
                # notes 先保留最少但有用的定位信息, 方便之后看 bad case.
                notes=f"title={title}; answers={answer_preview}",
            )
        )

    if not eval_cases:
        raise ValueError("No usable eval cases were produced from dataset rows.")

    return eval_cases


def import_hf_squad_sample(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    raw_dir: Path,
    eval_path: Path,
    query_limit: int,
) -> dict[str, Any]:
    """
    执行一次完整导入:
    拉样本 -> 写 raw txt -> 生成 eval jsonl -> 抽样写 query txt.
    """
    dataset_slug = sanitize_slug(dataset.replace("/", "_"))
    rows = fetch_dataset_rows(
        dataset=dataset,
        config=config,
        split=split,
        offset=offset,
        length=length,
    )
    eval_cases = convert_squad_rows_to_eval_cases(
        rows=rows,
        dataset_slug=dataset_slug,
        split=split,
        raw_dir=raw_dir,
    )
    write_eval_jsonl(eval_path, eval_cases)
    write_query_samples(
        query_dir=QUERY_DIR,
        eval_cases=eval_cases,
        prefix=dataset_slug,
        limit=query_limit,
    )
    return {
        # 返回一个简短摘要, 方便命令行调用后快速确认结果.
        "dataset": dataset,
        "config": config,
        "split": split,
        "requested_rows": length,
        "imported_cases": len(eval_cases),
        "raw_dir": str(raw_dir),
        "eval_path": str(eval_path),
    }


def main() -> None:
    """
    解析命令行参数并启动 Hugging Face 样本导入流程.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Import a Hugging Face QA sample for Azusa RAG eval")
    parser.add_argument("--dataset", default="autoevaluate/squad-sample")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--raw-dir", default=str(RAW_DATA_DIR))
    parser.add_argument(
        "--eval-path",
        default=str(PROJECT_ROOT / "data" / "eval" / "hf_squad_sample_eval.jsonl"),
    )
    parser.add_argument("--query-limit", type=int, default=3)
    args = parser.parse_args()

    result = import_hf_squad_sample(
        dataset=args.dataset,
        config=args.config,
        split=args.split,
        offset=args.offset,
        length=args.length,
        raw_dir=Path(args.raw_dir),
        eval_path=Path(args.eval_path),
        query_limit=args.query_limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
