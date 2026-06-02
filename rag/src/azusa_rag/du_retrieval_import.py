"""
我们当前的最低目标:

只使用内存里的 toy corpus / queries / qrels,
先验证它们能不能被整理成 Azusa eval 需要的 eval_records。

暂时不读真实数据集, 不写文件, 也不接入完整 RAG pipeline。

eval_records 形如:
[
    {
        "query_id": "q1",
        "query": "猫喜欢做什么?",
        "relevant_chunk_ids": ["p1", "p3"],
    }
]
"""

# 导入三个列表: 语料库corpus, 查询列表queries, 查询–文档相关性标注列表qrels
import json
from pathlib import Path

# 主流程
def build_eval_records(corpus, queries, qrels):
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

    for query_id, relevant_ids in query_id_to_relevant_ids.items(): # 这里要使用迭代器, 用了.items()    eval_records.append(
        eval_records.append(
            {
                "query_id": query_id,
                "query": query_id_to_text[query_id],
                "relevant_chunk_ids": relevant_ids,
            }
        )

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
    output_path.parent.mkdir(parents=True, exist_ok = True) # parents=True: 若中间目录不存在, 要一并递归创建. exist_ok=True: 若目录已经存在, 不要报错, 继续执行.

    with output_path.open("w", encoding = "utf-8") as f:
        for record in records:
            line = json.dumps(record, ensure_ascii=False)
            f.write(line)
            f.write("\n")



def main():
    toy_path = Path("rag/data/toy/du_retrieval_toy.json")
    with toy_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    corpus = data["corpus"]
    queries = data["queries"]
    qrels = data["qrels"]
    eval_records = build_eval_records(corpus= corpus, queries=queries, qrels=qrels)
    # print(eval_records)

    output_path = Path("rag/output/toy/du_retrieval_eval.jsonl")
    write_jsonl(output_path, eval_records)
    print(output_path)

if __name__ == "__main__":
    main()