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
    query_id: str                   # query_id 是这条query的唯一标识, 形如 cat_notes_query, 供后续 eval 时对照检查用. 这里沿用 doc_id + "_query" 的命名规则, 这样下游 search / eval 不需要改接口.
    query: str                      # query 是评测样本中的问题文本, 供后续 eval 时直接拿来做检索和问答用.
    relevant_chunk_ids: list[str]   # 哪些 chunk_id 是这个 query 的相关 chunk, 供后续 eval 时对照检查用.
    notes: str                      # 其他备注信息, 供后续查看 eval case 时参考用. 这里先保留一些最基本的定位信息, 方便之后看 bad case 时快速定位到原始数据是什么样子的.

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
    slug是别名的意思. 本函数把原来的文件名修改成成适合放进文件名 / doc_id 的安全片段.
    """
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    if not slug:
        return "item"
    return slug
    # 把各种奇怪字符都换成下划线
    # 去掉开头结尾多余的下划线
    # 全部转成小写
    # 如果最后什么都没剩下，就默认返回 item
    # 比如：
    # Hello, World! 变成 hello_world
    # SQuAD-sample/validation 变成 squad_sample_validation


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
    request = urllib.request.Request( # 构造 HTTP 请求, 直接把参数放在 URL query string 里.
        url=f"{DATASET_ROWS_API}?{query_string}",
        headers={"User-Agent": "Azusa-RAG-Dataset-Importer/0.1"},
        method="GET",
    )
    with urllib.request.urlopen(request) as response: # 发送 HTTP 请求并获取响应, 这里可能会抛出异常, 比如网络错误或者服务器返回错误状态码.
        response_dict = json.loads(response.read().decode("utf-8"))  # response_dict 形如 { ..., "rows": [ {"row": {...}}, {"row": {...}}, ... ] , ...}

    rows = []
    for row_item in response_dict["rows"]: # row_item 形如 {"row": {...}}，真正的数据在 "row" 键里。datasets-server 之所以这么设计，是为了在每条数据外层还包一层 metadata，比如说这个数据是从哪个 dataset/config/split 拉来的，方便后续追踪和调试。但对于我们当前的目的来说，这些 metadata 暂时用不到，所以这里直接取 "row" 键里的内容作为我们真正需要的数据行。
        # datasets-server 返回的每个元素外层还包了一层 row metadata, 这里统一只取真正的数据本体.
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
    chunk_items = chunk_text_with_spans( # 把 context 切成一块块的 chunk, 同时拿到每块 chunk 在 context 里的字符区间. 
        text=context,
        chunk_size=CHUNK_SIZE,
        overlap_size=CHUNK_OVERLAP,
    )
    relevant_chunk_ids = []

    for chunk_index, (_, chunk_start, chunk_end) in enumerate(chunk_items): # 对每个 chunk，看它在原文中的范围 chunk_start ~ chunk_end
        for answer_start, answer_text in zip(answer_starts, answer_texts):  # 对每个答案，看它在原文中的范围 answer_start ~ answer_end
            answer_end = answer_start + len(answer_text)
            # 只要 answer span 和 chunk span 有交集, 就把这个 chunk 标记为 relevant chunk.
            if answer_start < chunk_end and answer_end > chunk_start:
                chunk_id = f"{doc_id}_chunk_{chunk_index:03d}"
                if chunk_id not in relevant_chunk_ids:
                    relevant_chunk_ids.append(chunk_id)
                break

    if not relevant_chunk_ids: # 如果最终没有找到任何相关的 chunk_id, 就抛个错误提醒一下. 这样命令行调用时就不会傻傻地生成一个 eval case, 还以为成功了, 但其实这个 eval case 是个空壳, 后续 eval 时根本测不到任何东西.
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
    把评测样本列表写成当前项目使用的 jsonl 格式. jsonl 文件每行是一个 JSON 对象.
    """
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with eval_path.open("w", encoding="utf-8") as output_file:
        for eval_case in eval_cases:
            output_file.write(json.dumps(eval_case.to_dict(), ensure_ascii=False)) # dumps 方法把一个 Python 对象（这里是 eval_case.to_dict() 返回的字典）转换成一个 JSON 字符串。ensure_ascii=False 参数确保输出的 JSON 字符串中可以包含非 ASCII 字符（比如中文），而不是被转义成 \uXXXX 形式。
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
    eval_cases = [] # 此列表收集最终可写入 jsonl 的评测样本。

    for row_index, row in enumerate(rows):  # 逐行遍历 rows，也就是从 Hugging Face 拉下来的每条数据。
        question = row["question"].strip()  # 问题
        context = row["context"]            # 上下文正文
        answers = row["answers"]            # 答案信息
        answer_starts = answers["answer_start"] # 每个答案在context里的起始字符位置(偏移量)
        answer_texts = answers["text"]          # 每个答案对应的文本

        if not question or not context.strip():     # 基本校验: 问题, 上下文不能为空
            continue
        if not answer_starts or not answer_texts:   # 基本校验: 答案信息中必须有偏移量和文本
            continue

        # 保存txt文件
        row_id = str(row.get("id", row_index))
        title = str(row.get("title", "untitled"))
        doc_id = build_doc_id(dataset_slug, split, row_index, row_id) # 生成doc_id
        raw_path = raw_dir / f"{doc_id}.txt"
        write_text_file(raw_path, context)  # 把 context 原样写入raw目录下的txt文件, 这样 answer_start 这种字符偏移才不会错位.

        # 根据答案字符位置, 推导这条样本命中了哪些 chunk_id.
        relevant_chunk_ids = build_relevant_chunk_ids(
            doc_id=doc_id,
            context=context,
            answer_starts=answer_starts,
            answer_texts=answer_texts,
        )
        answer_preview = " | ".join(answer_texts[:2]) # 取前两个答案文本做预览, 方便后续查看 eval case 时快速了解这个问题的答案是什么. 如果没有答案文本, 就留空.

        # 把这个样本的 query 和相关 chunk_id 列表封装成一个 ImportedEvalCase, 加入 eval_cases 列表中.
        eval_cases.append(
            ImportedEvalCase(
                query_id=f"{doc_id}_query",
                query=question,
                relevant_chunk_ids=relevant_chunk_ids,
                # notes 先保留最少但有用的定位信息, 方便之后看 bad case.
                notes=f"title={title}; answers={answer_preview}",
            )
        )

    # 如果最终没有成功转换出任何 eval case, 就抛个错误提醒一下. 这样命令行调用时就不会傻傻地生成一个空文件, 还以为成功了.
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
    # 这里定义了命令行参数, 包括 dataset/config/split/offset/length/raw_dir/eval_path/query_limit 等, 这样用户在命令行调用这个脚本时就可以指定要导入哪个 dataset/config/split 的数据, 从哪个 offset 开始, 导入多少条, 以及 raw txt 和 eval jsonl 的保存位置.
    parser = argparse.ArgumentParser(description="Import a Hugging Face QA sample for Azusa RAG eval")
    parser.add_argument("--dataset", default="autoevaluate/squad-sample")  # 默认导入 autoevaluate/squad-sample 这个小型 SQuAD 样本数据集, 这样用户直接运行这个脚本就能快速看到效果. 当然用户也可以指定其他 dataset/config/split 来导入不同的数据.
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
    args = parser.parse_args()  # 解析命令行参数, 这样用户在命令行调用这个脚本时就可以指定要导入哪个 dataset/config/split 的数据, 从哪个 offset 开始, 导入多少条, 以及 raw txt 和 eval jsonl 的保存位置.

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
    print(json.dumps(result, ensure_ascii=False, indent=2)) # indent=2 参数让输出的 JSON 字符串更易读, 每个键值对占一行, 并且有缩进.


if __name__ == "__main__":
    main()
