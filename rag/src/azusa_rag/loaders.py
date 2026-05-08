# 把 raw/ 中的原始文档读进程序当中.
from pathlib import Path

# 读单个文件
def load_text_file(file_path: Path) -> str:

    return file_path.open("r", encoding="utf-8").read()
    # TODO: 文件对象没有显式关闭

    

# 读整个目录

def load_documents(files_path: Path) -> list[dict]:
    documents = []

    for txt_path in files_path.glob("*.txt"):
        text = load_text_file(txt_path)
        document = {
            "doc_id" : txt_path.stem,   # 形如cat_notes     (去除最后的扩展名)
            "source" : txt_path.name,   # 形如cat_notes.txt
            "text"   : text,
            "page"   : None,
        }
    
        documents.append(document)
    
    return documents