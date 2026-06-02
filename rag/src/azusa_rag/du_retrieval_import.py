"""
把 toy 数据转成两份中间产物：
du_retrieval_passages.jsonl
du_retrieval_eval.jsonl

    passage_records的每一行是一条可检索 passage. 形如:
    {
        "doc_id": "du_retrieval",
        "chunk_id": "p1",
        "chunk_index": 0,
        "source": "toy_du_retrieval",
        "page": None,
        "text": "猫喜欢晒太阳."
    }
    

    eval_records的每一行是一条 query 的金标答案. 形如:
    {
        "query_id": "q1",
        "query": "猫喜欢做什么?",
        "relevant_chunk_ids": ["p1", "p3"]
    }

    eval_records和passage_records的连接点:
    passage_records.chunk_id == eval_records.relevant_chunk_ids 里的那些 pid
"""

import argparse
import json
from pathlib import Path

from .config import CHUNK_OUTPUT_PATH, PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    """
    允许 importer 在不改代码的前提下切换输入输出路径，
    这样 toy 阶段和后面的正式样本阶段都能复用同一套脚本。
    """
    parser = argparse.ArgumentParser(description="Import DuRetrieval toy data into Azusa jsonl artifacts")
    parser.add_argument(
        "--input-path",
        default=str(PROJECT_ROOT / "data" / "toy" / "du_retrieval_toy.json"),
        help="Path to the toy DuRetrieval json file.",
    )
    parser.add_argument(
        "--eval-output-path",
        default=str(PROJECT_ROOT / "data" / "eval" / "du_retrieval_eval.jsonl"),
        help="Path to the eval jsonl output.",
    )
    parser.add_argument(
        "--passages-output-path",
        default=str(CHUNK_OUTPUT_PATH),
        help="Path to the passages jsonl output.",
    )
    parser.add_argument(
        "--notes",
        default="dataset=DuRetrievalToy; split=toy",
        help="Optional notes text written into every eval record.",
    )
    parser.add_argument(
        "--corpus-limit",
        type=int,
        default=None,
        help="Optional max number of corpus passages to keep from the input file.",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=None,
        help="Optional max number of query rows to keep from the input file.",
    )
    return parser


def limit_du_retrieval_rows(corpus, queries, qrels, corpus_limit=None, query_limit=None):
    """
    先对 corpus / queries 做截断，再把 qrels 过滤到同一批 id 范围里。
    这样后面的小样本评测就不会误混入被截掉的数据。
    """
    limited_corpus = corpus[:corpus_limit] if corpus_limit is not None else corpus
    limited_queries = queries[:query_limit] if query_limit is not None else queries

    valid_corpus_ids = {passage["id"] for passage in limited_corpus}
    valid_query_ids = {query["id"] for query in limited_queries}

    limited_qrels = []
    for qrel in qrels:
        if qrel["qid"] not in valid_query_ids:
            continue
        if qrel["pid"] not in valid_corpus_ids:
            continue
        limited_qrels.append(qrel)

    return limited_corpus, limited_queries, limited_qrels


def build_eval_records(corpus, queries, qrels, notes_text=None):
    """
    负责把queries和qrels变成"评测金标".
    对于每个query, 找出它对应哪些正确passage id, 最后生成eval_records.
    eval_records的每一项形如:
    {
        "query_id": "q1",
        "query": "猫喜欢做什么?",
        "relevant_chunk_ids": ["p1", "p3"]
    }
    """
    
    # 从corpus 提取一个 corpus_id 集合
    corpus_ids = {passage["id"] for passage in corpus}
    
    query_id_to_text = { # 声明一个字典, 键是query中的id, 值是query中的text
        query["id"]: query["text"]
        for query in queries
    }

    query_id_to_relevant_ids = {} # 声明一个空字典. 键是query_id, 值是一个列表, 列表内都是relevant_id

    # 创建query_id -> 对应的relevant_id 列表的字典
    # qrel只有在 qid合法 pid合法 score > 0 时, 才进入最终的 eval_records
    for qrel in qrels:

        # 如果不满足score > 0, 认为score值非法.
        if qrel["score"] <= 0:
            continue
        
        query_id = qrel["qid"]
        passage_id = qrel["pid"]

        # 如果一个query_id根本找不到对应的query_text, 那认为这个query_id非法.
        # qrels中假的qid不会进入最终的eval_records.
        if query_id not in query_id_to_text:
            continue

        # 遍历qrels时, 检查pid是不是在corpus_ids这个集合里.
        # 否则, relevant_chunk_ids(金标)当中可能会混杂着并不在corpus_ids(语料库)当中的pid
        # 这就会导致永远无法命中金标, 反应到数据上就是命中率不佳
        if passage_id not in corpus_ids:
            continue

        if query_id not in query_id_to_relevant_ids: 
            query_id_to_relevant_ids[query_id] = []
        
        if passage_id not in query_id_to_relevant_ids[query_id]: # 避免同一个passage_id被重复添加, 得到形如 金标 = ["p1", "p1", "p3"]  的错误结果.
            query_id_to_relevant_ids[query_id].append(passage_id)

    eval_records = []
    for query_id, relevant_ids in query_id_to_relevant_ids.items():
        # 这里把同一条 query 的金标整理成最终要写入 jsonl 的一行。
        eval_record = {
            "query_id": query_id,
            "query": query_id_to_text[query_id],
            "relevant_chunk_ids": relevant_ids,
        }
        if notes_text:
            # 这里补一条统一 notes，方便后面看 bad case 时知道这批 eval 是哪来的。
            eval_record["notes"] = notes_text
        eval_records.append(eval_record)

    return eval_records

def build_passage_records(corpus):
    """
    负责把corpus变成"可检索语料记录"
    把corpus里面的每条passage 整理成后续embedding/search能读取的记录.
    passage_records的每一项形如:
    {
        "doc_id": "du_retrieval",
        "chunk_id": "p1",
        "chunk_index": 0,
        "source": "toy_du_retrieval",
        "page": None,
        "text": "猫喜欢晒太阳."
    }
    
    eval_records和passage_records的连接点:
    passage_records.chunk_id == eval_records.relevant_chunk_ids 里的那些 pid
    """
    passage_records = []
    for index, passage in enumerate(corpus):
        passage_record = {
            "doc_id": "du_retrieval",
            "chunk_id": passage["id"],
            "chunk_index": index,
            "source": "toy_du_retrieval",
            "page": None,
            "text": passage["text"],
        }
        passage_records.append(passage_record)
    return passage_records


def write_jsonl(output_path, records):
    """
    把内存里的eval_records写成一个最小的.jsonl文件
    """
    # 先把父目录补出来，避免第一次运行时因为目录不存在而失败。
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding = "utf-8") as f:
        for record in records:
            line = json.dumps(record, ensure_ascii=False)
            f.write(line)
            f.write("\n")


def main():
    args = build_parser().parse_args()

    input_path = Path(args.input_path)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    corpus = data["corpus"]
    queries = data["queries"]
    qrels = data["qrels"]
    # 这里先按命令行限制缩小数据规模，方便我们用同一个 importer 跑 toy / 小样本 / 更大样本。
    corpus, queries, qrels = limit_du_retrieval_rows(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        corpus_limit=args.corpus_limit,
        query_limit=args.query_limit,
    )
    eval_records = build_eval_records(
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        notes_text=args.notes,
    )
    passage_records = build_passage_records(corpus)

    # 这里把输出路径也参数化，后面切到 data/eval 或 artifacts 时就不用再改脚本正文。
    output_path = Path(args.eval_output_path)
    write_jsonl(output_path, eval_records)
    passages_output_path = Path(args.passages_output_path)
    write_jsonl(passages_output_path, passage_records)
    print(output_path)
    print(passages_output_path)

if __name__ == "__main__":
    main()
