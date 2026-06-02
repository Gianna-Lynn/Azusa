"""
负责从文件里加载数据, 形成内存里的数据结构, 供后续的切chunk和写jsonl使用.
"""
from pathlib import Path
from .schemas import ChunkRecord, EmbeddingRecord
import json


def resolve_jsonl_input_path(jsonl_path: Path) -> Path:
    """
    优先读取新的分层路径；如果文件还在旧的 output 根目录下，则自动回退。
    """
    if jsonl_path.exists():
        return jsonl_path

    legacy_path = jsonl_path.parent.parent / jsonl_path.name
    if legacy_path.exists():
        return legacy_path

    return jsonl_path

def load_text_file(file_path: Path) -> str:
    """
    打开file_path所指定的文本文件, 读它的内容, 返回一个字符串.
    此为load_documents的helper, 不宜单独import
    """
    with file_path.open("r", encoding="utf-8") as text_file:
        return text_file.read()


def load_documents(files_path: Path) -> list[dict]:
    """
    打开files_path所指定的目录, 找出里面所有的txt文件, 读它们的内容, 把每个文件的内容和一些元信息(比如文件名)放在一个字典里, 最后把所有的字典放在一个列表里返回.
    """
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
    """
    从jsonl文件里读chunk记录, 返回一个ChunkRecord对象的列表.
    """
    jsonl_path = resolve_jsonl_input_path(jsonl_path)
    chunk_record_list = []

    with jsonl_path.open("r", encoding="utf-8") as jsonl_file:
        
        for line in jsonl_file:
            record_dict = json.loads(line)  # 这里的record_dict是一个字典, 这个字典的结构刚好和ChunkRecord一致
            chunk_record = ChunkRecord(**record_dict) # chunks.jsonl里每一行的键, 都和ChunkRecord的字段名一致. **方法可以"把字典摊开", 直接赋值进去.
            chunk_record_list.append(chunk_record)

    return chunk_record_list


def load_embedding_records(jsonl_path: Path) -> list[EmbeddingRecord]:
    """
    从jsonl文件里读embedding记录, 返回一个EmbeddingRecord对象的列表.
    """
    jsonl_path = resolve_jsonl_input_path(jsonl_path)
    embedding_record_list = []

    with jsonl_path.open("r", encoding="utf-8") as jsonl_file:
        
        for line in jsonl_file:
            record_dict = json.loads(line)  # 这里的record_dict是一个字典, 这个字典的结构刚好和ChunkRecord一致
            embedding_record = EmbeddingRecord(**record_dict) # chunks.jsonl里每一行的键, 都和ChunkRecord的字段名一致. **方法可以"把字典摊开", 直接赋值进去.
            embedding_record_list.append(embedding_record)

    return embedding_record_list
