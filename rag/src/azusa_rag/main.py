"""
整个项目的主入口, 负责把前面写好的几块拼起来, 形成一条可运行的最小流水线.
主要作用是: 
1. 调用 loaders.py 读文档; 
2. 调用 chunker.py 切 chunk; 
3. 把切好的 chunk 写成 jsonl 文件, 输出到 output/chunks.jsonl 里.
"""

from pathlib import Path
from .schemas import ChunkRecord  # 这里的.表示 "从当前的包"里面找schemas. 但是Python就会问"你现在当前文件属于哪个包?" python -m azusa_rag.main 明确指示当前包是azusa_rag, $env:PYTHONPATH='src'告诉Python: 包azusa_rag在src下面, 去那里查找.
from .chunker import chunk_document
from .config import RAW_DATA_DIR, CHUNK_OUTPUT_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from .loaders import load_documents
import json

def write_jsonl(output_path: Path, records: list[ChunkRecord]) -> None: # jsonl 的意思是：JSON Lines, 也就是：一行一个 JSON 对象, 不是一个大列表
    """
    把records里每个ChunkRecord对象都写成一行JSON, 输出到output_path指定的文件里.
    """
    # 打开输出文件
    # 循环 records
    # 每个 record 先变成字典
    # 再 json.dumps(...)
    # 每条后面补一个换行
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file: # 认为output_file是文件指针, 可以调用.write()方法. 
        # with...as的构型作用是:在这个代码块里,把文件交给output_file使用, 离开代码块时自动关闭.
        for record in records: # 每个record是一个ChunkRecord类型
            record_dict = record.to_dict()  # record_dict是刚准备好的数据
            json_line = json.dumps(record_dict, ensure_ascii=False)    # dumps的返回值更像"编码后的文本". 这个"编码后的文本"是要写入具体文件当中的.
            output_file.write(json_line)
            output_file.write("\n")



def main() -> None:
    """
    主函数
    """

    # 拿配置
    # 调 load_documents(...)
    # 准备一个总列表
    # 循环每份 document
    # 调 chunk_document(...)
    # 把得到的 chunk 追加到总列表
    # 最后调用“写 jsonl”的函数
    raw_data_dir = RAW_DATA_DIR
    chunk_output_path = CHUNK_OUTPUT_PATH
    chunk_size = CHUNK_SIZE
    chunk_overlap = CHUNK_OVERLAP

    documents = load_documents(raw_data_dir) # 返回 list[dict]

    chunk_list = []

    for document in documents:
        chunkrec_list = chunk_document(
            doc_dict=document, 
            chunk_size=chunk_size, 
            overlap_size=chunk_overlap
        )
        # chunkrec_list类型是list[ChunkRecord], 不能用append直接放进列表, 否则会得到"列表套列表".
        # 用extend, 可以代替 for 循环中的逐个 append.
        chunk_list.extend(chunkrec_list)

    write_jsonl(chunk_output_path, chunk_list)

if __name__ == "__main__": # __name__用来判断"这个文件现在是不是作为主程序在跑", 表达式为true, 表示 "这个文件现在是主入口"
    main()
