"""
负责把 "整份文档文本" 切成 "多个chunk"
"""
from .schemas import ChunkRecord # 相对导入, 因为chunker.py 和 schemas.py在同一个包azusa_regs里面. 我可以从同一个包的 schemas模块中直接拿取ChunkRecord这个类.

natural_boundaries = ('\n',  '.',  '?',  '!',  '。',  '？',   '！') # 定义边界字符

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
        remaining_length = len(text) - start

        if remaining_length <= chunk_size:  # 如果剩下的长度不够chunk_size
            chunk = text[start: len(text)]  # 直接把剩下的内容都归为一个chunk  
            if chunk:                       # 如果收集到的不是空文本
                str_list.append(chunk)      # 放进str_list
            break

        end = start + chunk_size  # end表示本次切割的终点
        # 这是之前的end的赋值逻辑, 在这里需要依据last_boundary进行修改. 
        
        candidate = text[start: end] # "切出子串" 利用Python的切片 # strip可以移除打头和尾巴上的空格 # 由于引入了边界字符机制, 不再使用.strip

        last_boundary = -1    # 注意last_boundary是相对于candidate的索引, 不是相对于大text的索引. 所以调整后的长度是 last_boundary + 1

        for boundary in natural_boundaries:
            pos = candidate.rfind(boundary)
            if pos > last_boundary:
                last_boundary = pos
                # 可能需要考虑一种情况, 即便是最靠后的pos也会导致整个chunk的长度过小
        if last_boundary != -1:
            adjusted_length = last_boundary + 1
            min_chunk_length = chunk_size * 0.5

            if adjusted_length >= min_chunk_length:
                end = start + last_boundary + 1 # Python切片中传入的是左闭右开, last_boundary是在candidate里的索引, +1可以保证边界字符也被放进candidate

        chunk = text[start: end].strip()

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
            chunk_index = index,
            source = source,    # source 形如cat_notes.txt
            text = chunk,
            page = page,
        )
        doc_list.append(chunk_rec)
        
    return doc_list