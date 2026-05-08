"""
负责把 "整份文档文本" 切成 "多个chunk"
"""
from .schemas import ChunkRecord # 相对导入, 因为chunker.py 和 schemas.py在同一个包azusa_regs里面. 我可以从同一个包的 schemas模块中直接拿取ChunkRecord这个类.

# 第一层只管“怎么切文本”
def chunk_text(text: str, chunk_size: int, overlap_size: int) -> list[str]:
    if chunk_size <= 0:     # 不允许负chunk长度
        raise ValueError("chunk_size must be positive")
    if overlap_size >= chunk_size:  # 不允许重叠长度大于chunk长度
        raise ValueError("overlap_size must be less than chunk_size")
    if overlap_size < 0:    # 不允许负重叠
        raise ValueError("overlap_size must be non-negative")

    
    str_list = []

    start = 0
    step = chunk_size - overlap_size # step 仅用来更新start


    while start < len(text):

        end = start + chunk_size  # end表示本次切割的终点
        chunk = text[start: end].strip()  # "切出子串" 利用Python的切片 # strip可以移除打头和尾巴上的空格


        if chunk:
            str_list.append(chunk)

        start += step

    return str_list

# 第二层只管“怎么把切好的文本挂回文档信息”
def chunk_document(doc_dict: dict, chunk_size: int, overlap_size: int) -> list[ChunkRecord]:
    doc_list = []

    doc_id = doc_dict["doc_id"]
    text = doc_dict["text"]
    source = doc_dict["source"]
    page = doc_dict["page"]

    
    str_list = chunk_text(text=text, chunk_size = chunk_size, overlap_size = overlap_size)
    
    for index, chunk in enumerate(str_list):
        chunk_rec = ChunkRecord(
            doc_id = doc_id,    # doc_id 形如cat_notes
            chunk_id = doc_id + "_chunk_" + f"{index:03d}",  # chunk_id是字符串, 形如cat_notes_chunk_000 # 此处补全为三位数字编码
            source = source,    # source 形如cat_notes.txt
            text = chunk,
            page = page,
        )
        doc_list.append(chunk_rec)
        
    return doc_list