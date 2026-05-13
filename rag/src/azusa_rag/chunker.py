"""
负责把 "整份文档文本" 切成 "多个chunk"
"""
from .schemas import ChunkRecord # 相对导入, 因为chunker.py 和 schemas.py在同一个包azusa_regs里面. 我可以从同一个包的 schemas模块中直接拿取ChunkRecord这个类.

natural_boundaries = ('\n',  '.',  '?',  '!',  '。',  '？',   '！') # 定义边界字符

def _validate_chunk_params(chunk_size: int, overlap_size: int) -> None:
    """
    检查 chunk 参数是否合法, 避免后续切分时出现死循环或无效配置.
    """
    if chunk_size <= 0:     # 不允许负chunk长度
        raise ValueError("chunk_size must be positive")
    if overlap_size >= chunk_size:  # 不允许重叠长度大于chunk长度
        raise ValueError("overlap_size must be less than chunk_size")
    if overlap_size < 0:    # 不允许负重叠
        raise ValueError("overlap_size must be non-negative")


def _strip_span(text: str, start: int, end: int) -> tuple[int, int]:
    """
    在不改变原文主体内容的前提下, 去掉一个候选 span 两端的空白字符.
    """
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def chunk_text_with_spans(text: str, chunk_size: int, overlap_size: int) -> list[tuple[str, int, int]]:
    """
    把文本切成多个 chunk, 同时保留每个 chunk 在原文里的字符区间.
    """
    _validate_chunk_params(chunk_size, overlap_size)

    chunk_items = []

    start = 0
    while start < len(text):
        remaining_length = len(text) - start

        if remaining_length <= chunk_size:  # 如果剩下的长度不够chunk_size
            stripped_start, stripped_end = _strip_span(text, start, len(text))
            chunk = text[stripped_start: stripped_end]
            if chunk:
                chunk_items.append((chunk, stripped_start, stripped_end))
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

        stripped_start, stripped_end = _strip_span(text, start, end)
        chunk = text[stripped_start: stripped_end]

        if chunk:
            chunk_items.append((chunk, stripped_start, stripped_end))

        # 下一块从“本块真实结束位置 - overlap”继续,
        # 这样在边界收缩后也不会把中间一段文本跳过去.
        next_start = end - overlap_size
        if next_start <= start:
            # 理论上很少走到这里, 这里只是兜底避免极端情况下卡死.
            next_start = start + 1
        start = next_start

    return chunk_items


# 第一层只管“怎么切文本”
def chunk_text(text: str, chunk_size: int, overlap_size: int) -> list[str]:
    """
    对外暴露的最简切块接口: 只返回 chunk 文本, 不返回 span 信息.
    """
    str_list = []
    chunk_items = chunk_text_with_spans(text, chunk_size, overlap_size)
    for chunk_text_item, _, _ in chunk_items:
        str_list.append(chunk_text_item)
    return str_list

# 第二层只管“怎么把切好的文本挂回文档信息”
def chunk_document(doc_dict: dict, chunk_size: int, overlap_size: int) -> list[ChunkRecord]:
    """
    把一份原始文档切成多个 ChunkRecord, 供后续写 jsonl 和 embedding 使用.
    """
    doc_list = []

    doc_id = doc_dict["doc_id"]
    text = doc_dict["text"]
    source = doc_dict["source"]
    page = doc_dict["page"]

    
    chunk_items = chunk_text_with_spans(text=text, chunk_size=chunk_size, overlap_size=overlap_size)
    
    for index, (chunk, _, _) in enumerate(chunk_items):
        # chunk_id 仍然沿用原来的编号规则, 这样下游 search / eval 不需要改接口.
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
