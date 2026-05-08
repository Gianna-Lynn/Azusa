"""
把 raw/ 中的原始文档读进程序当中.
"""
from pathlib import Path
from .schemas import ChunkRecord
import json

# 读单个文件
def load_text_file(file_path: Path) -> str:

    with file_path.open("r", encoding="utf-8") as text_file:
        return text_file.read()

    

# 读整个目录
def load_documents(files_path: Path) -> list[dict]:
    documents = []
    txt_paths = list(files_path.glob("*.txt")) # glob方法返回的是Generator, 加上list才会变成列表
    if not txt_paths:
        raise ValueError("No .txt files found in directory: " + str(files_path))

    for txt_path in txt_paths:
        text = load_text_file(txt_path)

        if not text.strip():
            continue

        document = {
            "doc_id" : txt_path.stem,   # 形如cat_notes     (去除最后的扩展名)
            "source" : txt_path.name,   # 形如cat_notes.txt
            "text"   : text,
            "page"   : None,
        }
    
        documents.append(document)
    
    if not documents:
        raise ValueError("In directory: " + str(files_path) + ", all the .txt files are blank")

    return documents

def load_chunk_records(jsonl_path: Path) -> list[ChunkRecord]:
    chunk_record_list = []

    with jsonl_path.open("r", encoding="utf-8") as jsonl_file:
        
        for line in jsonl_file:
            record_dict = json.loads(line)  # 这里的record_dict是一个字典, 这个字典的结构刚好和ChunkRecord一致
            chunk_record = ChunkRecord(**record_dict) # chunks.jsonl里每一行的键, 都和ChunkRecord的字段名一致. **方法可以"把字典摊开", 直接赋值进去.
            chunk_record_list.append(chunk_record)

    return chunk_record_list