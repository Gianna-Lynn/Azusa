"""
接收ChunkRecord, 返回EmbeddingRecord
"""
from .schemas import ChunkRecord, EmbeddingRecord


# 把单个ChunkRecord变成单个EmbeddingRecord
def chunk_to_embedding(chunk_record: ChunkRecord) -> EmbeddingRecord:
    
    embedding = [1.0, 2.0, 3.0]  # list[float]类型, 纯属占位, 这个向量是用来表示"已经编码完毕后的产物"
    
    embedding_record = EmbeddingRecord(
        doc_id= chunk_record.doc_id,
        chunk_id= chunk_record.chunk_id,
        chunk_index= chunk_record.chunk_index,
        source = chunk_record.source,
        text= chunk_record.text,
        embedding= embedding,
        page= chunk_record.page,
    )

    return embedding_record



# 把一批ChunkRecord变成一批EmbeddingRecord
def chunk_to_embedding_batch(chunk_record_list: list[ChunkRecord]) -> list[EmbeddingRecord]: 
    embedding_record_list = []

    for chunk_record in chunk_record_list:
        embedding_record = chunk_to_embedding(chunk_record)
        embedding_record_list.append(embedding_record) # 这里可能要判空?

    return embedding_record_list # 这里可能要判空?