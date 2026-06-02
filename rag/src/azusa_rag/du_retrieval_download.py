"""
把 Hugging Face 上的 DuRetrieval 小规模样本下载到本地 json 文件。

设计目标:
1. 下载步骤和 importer 解耦
2. 先把线上数据落到本地, 后续 importer 只负责读本地
3. 支持按 query / corpus 规模做小样本试运转
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
import urllib.request

from .config import PROJECT_ROOT


DATASET_ROWS_API = "https://datasets-server.huggingface.co/rows"
DATASET_DU_RETRIEVAL = "C-MTEB/DuRetrieval"
DATASET_DU_RETRIEVAL_QRELS = "C-MTEB/DuRetrieval-qrels"
ROWS_API_PAGE_LIMIT = 100
ROWS_API_MAX_RETRIES = 5
ROWS_API_RETRY_BASE_SECONDS = 1.0
ROWS_API_PAGE_PAUSE_SECONDS = 0.2


def build_parser() -> argparse.ArgumentParser:
    """
    下载 DuRetrieval 小样本快照并落到本地指定目录。
    """
    parser = argparse.ArgumentParser(description="Download a local DuRetrieval sample snapshot")
    parser.add_argument(
        "--output-path",
        default=str(PROJECT_ROOT / "data" / "processed" / "du_retrieval_sample.json"),
        help="Local json snapshot path.",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=20,
        help="How many query rows to keep in the local sample snapshot.",
    )
    parser.add_argument(
        "--corpus-limit",
        type=int,
        default=1000,
        help="How many corpus rows to keep in the local sample snapshot.",
    )
    parser.add_argument(
        "--query-offset",
        type=int,
        default=0,
        help="Offset into the queries split.",
    )
    parser.add_argument(
        "--corpus-offset",
        type=int,
        default=0,
        help="Offset into the corpus split.",
    )
    parser.add_argument(
        "--qrels-offset",
        type=int,
        default=0,
        help="Offset into the qrels split before scanning rows.",
    )
    parser.add_argument(
        "--qrels-scan-length",
        type=int,
        default=10000,
        help="How many qrels rows to scan when matching the selected query/corpus sample.",
    )
    return parser


def fetch_dataset_rows(
    dataset: str,
    split: str,
    offset: int,
    length: int,
    config: str = "default",
) -> list[dict]:
    """
    从 Hugging Face datasets-server 取一段行数据。
    rows API 单次最多稳定返回 100 行，所以这里统一按页循环拉取。
    """
    rows = []
    remaining = length
    current_offset = offset

    while remaining > 0:
        page_length = min(remaining, ROWS_API_PAGE_LIMIT)
        query_string = urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": current_offset,
                "length": page_length,
            }
        )
        request = urllib.request.Request(
            url=f"{DATASET_ROWS_API}?{query_string}",
            headers={"User-Agent": "Azusa-DuRetrieval-Downloader/0.1"},
            method="GET",
        )
        response_dict = None
        for attempt_index in range(ROWS_API_MAX_RETRIES):
            try:
                with urllib.request.urlopen(request) as response:
                    response_dict = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as error:
                if error.code != 429 or attempt_index == ROWS_API_MAX_RETRIES - 1:
                    raise

                # 这里专门对 429 做退避重试，目的是让“中等规模 sample 下载”
                # 更稳定，而不是因为 Hugging Face rows API 临时限流就整轮失败。
                sleep_seconds = ROWS_API_RETRY_BASE_SECONDS * (2 ** attempt_index)
                time.sleep(sleep_seconds)
            except URLError:
                if attempt_index == ROWS_API_MAX_RETRIES - 1:
                    raise

                # 某些网络抖动会表现成 SSL EOF / connection reset，
                # 这里同样按退避重试处理，避免整轮下载被一次瞬时波动打断。
                sleep_seconds = ROWS_API_RETRY_BASE_SECONDS * (2 ** attempt_index)
                time.sleep(sleep_seconds)

        if response_dict is None:
            raise RuntimeError("Failed to fetch dataset rows after retrying")

        page_rows = [row_item["row"] for row_item in response_dict["rows"]]
        rows.extend(page_rows)

        if len(page_rows) < page_length:
            break

        current_offset += page_length
        remaining -= page_length
        if remaining > 0:
            # 在页与页之间轻微停顿，尽量别在连续翻页时把 rows API 顶得太紧。
            time.sleep(ROWS_API_PAGE_PAUSE_SECONDS)

    return rows


def build_local_sample_snapshot(
    corpus_rows: list[dict],
    query_rows: list[dict],
    qrel_rows: list[dict],
) -> dict:
    """
    把下载下来的三部分行数据整理成 importer 已经能直接读取的本地 json 结构。
    """
    query_ids = {row["id"] for row in query_rows}
    corpus_ids = {row["id"] for row in corpus_rows}

    filtered_qrels = []
    for qrel in qrel_rows:
        if qrel["qid"] not in query_ids:
            continue
        if qrel["pid"] not in corpus_ids:
            continue
        filtered_qrels.append(qrel)

    return {
        "corpus": corpus_rows,
        "queries": query_rows,
        "qrels": filtered_qrels,
    }


def write_snapshot(output_path: Path, snapshot: dict) -> None:
    """
    把下载后的小样本快照写成一个本地 json 文件。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()

    # 这里先分别抓取 corpus / queries / qrels 三份线上数据，
    # 再在本地裁成 importer 能直接消费的 json 结构。
    corpus_rows = fetch_dataset_rows(
        dataset=DATASET_DU_RETRIEVAL,
        split="corpus",
        offset=args.corpus_offset,
        length=args.corpus_limit,
    )
    query_rows = fetch_dataset_rows(
        dataset=DATASET_DU_RETRIEVAL,
        split="queries",
        offset=args.query_offset,
        length=args.query_limit,
    )
    qrel_rows = fetch_dataset_rows(
        dataset=DATASET_DU_RETRIEVAL_QRELS,
        split="dev",
        offset=args.qrels_offset,
        length=args.qrels_scan_length,
    )

    snapshot = build_local_sample_snapshot(
        corpus_rows=corpus_rows,
        query_rows=query_rows,
        qrel_rows=qrel_rows,
    )

    output_path = Path(args.output_path)
    write_snapshot(output_path, snapshot)
    print(output_path)
    print(
        json.dumps(
            {
                "corpus_rows": len(snapshot["corpus"]),
                "query_rows": len(snapshot["queries"]),
                "qrel_rows": len(snapshot["qrels"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
