"""
这个文件负责实现检索的功能, 也就是RAG里那个"R"的部分. 主要是:
1. 启动时读 embedding.jsonl
2. 按行号给每条 embedding 一个整数 id
3. 用 embedding 建 Azusa 索引
4. 同时把整条记录保留在 Python 内存列表里
5. query 也做 embedding
6. 调 Azusa 的搜索接口拿回 top-k id
7. 再用 records[id] 找回原始 chunk 信息
"""

from .loaders import load_embedding_records, load_documents
from .config import EMBEDDING_OUTPUT_PATH, QUERY_DIR, HEALTH_URL, SEARCH_URL, INSERT_URL
from .schemas import EmbeddingRecord
from .embedder import embed_text_list
import json
import urllib.request

def get_query()->str:
    """
    最简化的得到query str的函数.仅取目录中所有txt文件读取得到的第一个str
    """
    list_dict = load_documents(QUERY_DIR)
    return list_dict[0]["text"]     # str

def embed_query(query: str) -> list[float]:
    """
    用embed_text_list方法得到query vector
    """
    list_float = embed_text_list([query])[0]    # embed_text_list 返回 list[list[float]], 但最外层list只有一个元素, 而这个元素是一个list[float]
    return list_float

def dot_product(vec1: list[float], vec2: list[float]) -> float:
    """
    两个向量的点积.最好提前将两个向量归一化.
    """
    if len(vec1) != len(vec2):
        raise ValueError("vec1 and vec2 have different length, cannot caculate dot product")
    ans = 0.0
    for elem1, elem2 in zip(vec1, vec2):
        ans += elem1 * elem2
    return ans

def embedvecs_to_searchvecs(embedding_record_list: list[EmbeddingRecord])-> list[tuple[int, list[float]]]:
    """
    把EmbeddingRecord列表变成(id, vector)列表
    """
    # 输出的 list[tuple[int, list[float]]] 中tuple包含的是 id, vector (这两位是检索的核心)
    id_vector_list = []
    for index, embedding_record in enumerate(embedding_record_list):
        embedding = embedding_record.embedding
        id_vector_list.append((index, embedding))
    return id_vector_list

def search_core_brute(query_vec: list[float], search_vecs: list[tuple[int, list[float]]], top_k = 3) -> list[int]:
    """
    暴力搜索内核.接受query向量,待检索的所有向量,返回一个列表,里面是最优的top-k个id
    """

    # search_vecs的结构是: list[(id, vector)]
    score_id_list = []

    for (id, vector) in search_vecs:
        score = dot_product(query_vec, vector)
        score_id_list.append((score, id))

        # sort包含tier逻辑, 规定dot_product相同的时候让id小的在前面, 而不是只用reverse=True
    score_id_list.sort(key= lambda item:(-item[0], item[1]))

    top_k_list = []
    for (_, id) in score_id_list:
        top_k_list.append(id)
    
    return top_k_list[:top_k]

def search_core_hnsw(query_vec: list[float], search_vecs: list[tuple[int, list[float]]], top_k = 3) -> list[int]:
    """
    hnsw搜索内核.接口和search_core_brute一致.
    """
    # 1. 组一个 dict
    # 这个结构对应到azusa_server.rs中的struct SearchRequest
    request_dict = {
        "vector": query_vec,
        "k": top_k,
        "ef_search" : 16,
    }

    # 2. json.dumps(...)转换为一个json行
    request_json = json.dumps(request_dict, ensure_ascii=False) # dumps返回str
    # HTTP发送前通常要求把字符串变成bytes
    request_body = request_json.encode("utf-8") # bytes

    # 3. POST 到 http://127.0.0.1:3000/search

    # 构造一个request对象, 告诉Python下面的信息:
    request = urllib.request.Request(
        url = SEARCH_URL,   # 目标URL
        data = request_body,                    # 请求体
        headers = {"Content-Type": "application/json"}, # 请求头里带有这些内容
        method = "POST",                        # 请求方法
    )
    
    # 4. 读回响应 JSON
    with urllib.request.urlopen(request) as response: # urlopen返回_UrlopenRet, 是一个HTTP响应对象 
          # 按照azusa_server的search_handler, 返回的是 Json<SearchResponse>
          # SearchResponse 的 JSON 形状是: {"results": [{"id": 3, "score": 0.982}, ...]}
          response_body: bytes = response.read() # bytes, 形如：b'{"results":[{"id":3,"score":0.982},{"id":7,"score":0.961},{"id":1,"score":0.933}]}'
          response_json = response_body.decode("utf-8") # 把 bytes 解码成 Python 字符串(str), 形如 '{"results":[{"id":3,"score":0.982},{"id":7,"score":0.961},{"id":1,"score":0.933}]}'(备注:这是纯文本, 是纯文本, str)
          response_data = json.loads(response_json) # 把 JSON 字符串 str 解析成 Python 内置对象dict, 形如 {"results":[{"id":3,"score":0.982},{"id":7,"score":0.961},{"id":1,"score":0.933}]}

    # 5. 提取 results 里的 id    
    score_id_list = []
    for hit in response_data["results"]:
        score_id_list.append(hit["id"])

    # 6. 返回 list[int]
    return score_id_list

def ensure_hnsw_index_loaded(embedding_records: list[EmbeddingRecord]) -> None:
    """
    确保 Rust Hnsw 中已经有数据了.
    """
    # 先请求 /health
    with urllib.request.urlopen(HEALTH_URL) as response:
        # 按照azusa_server的health_handler, 返回的是 Json<HealthResponse>
        # Json<HealthResponse> 形如:  {"status":"ok","record_count":0}
        response_body: bytes = response.read()
        response_json = response_body.decode("utf-8")
        response_data = json.loads(response_json)
    record_count = response_data["record_count"]
    
    # 最小安全策略:
    # - record_count == 0: server 还是空索引, 继续走后面的逐条插入;
    # - record_count == len(embedding_records): 暂时认为 server 里大概率已经是当前这批数据;
    # - 其他情况: 不静默复用旧索引, 直接报错提醒先重启 azusa_server.
    #
    # 这样至少能把“旧索引混进新数据集”的问题从静默错位, 改成显式报错.
    if record_count == len(embedding_records):
        return
    if record_count != 0:
        raise ValueError(
            "Azusa server already contains a different in-memory index. "
            "Please restart azusa_server before loading a new embedding dataset."
        )
    # record_count == 0，就逐条 /insert

    for index, embedding_record in enumerate(embedding_records):
        
        # 和上面的search逻辑基本一致:
        # 1. 组一个 dict
        # index 是 id
        # embedding_record是数据本体
        insert_dict = embedding_record_to_azusa_insert_record(embedding_record=embedding_record, record_id=index)
        
        # 2. json.dumps(...)转换为一个json行
        insert_json = json.dumps(insert_dict, ensure_ascii=False) # dumps返回str
        # HTTP发送前通常要求把字符串变成bytes
        insert_body = insert_json.encode("utf-8") # bytes

        # 3. POST 到 http://127.0.0.1:3000/insert

        # 构造一个request对象, 告诉Python下面的信息:
        request = urllib.request.Request(
            url = INSERT_URL,   # 目标URL
            data = insert_body,                    # 请求体
            headers = {"Content-Type": "application/json"}, # 请求头里带有这些内容
            method = "POST",                        # 请求方法
        )

        # 4. 读回响应 JSON
        with urllib.request.urlopen(request) as response: # urlopen返回_UrlopenRet, 是一个HTTP响应对象 
            # # 按照azusa_server的search_handler, 返回的是 Json<SearchResponse>
            # # SearchResponse 的 JSON 形状是: {"results": [{"id": 3, "score": 0.982}, ...]}
            # response_body: bytes = response.read() # bytes, 形如：b'{"results":[{"id":3,"score":0.982},{"id":7,"score":0.961},{"id":1,"score":0.933}]}'
            # response_json = response_body.decode("utf-8") # 把 bytes 解码成 Python 字符串(str), 形如 '{"results":[{"id":3,"score":0.982},{"id":7,"score":0.961},{"id":1,"score":0.933}]}'(备注:这是纯文本, 是纯文本, str)
            # response_data = json.loads(response_json) # 把 JSON 字符串 str 解析成 Python 内置对象dict, 形如 {"results":[{"id":3,"score":0.982},{"id":7,"score":0.961},{"id":1,"score":0.933}]}
            response.status

def embedding_record_to_azusa_insert_record(
    embedding_record: EmbeddingRecord,
    record_id: int,
) -> dict:
    """
    把EmbeddingRecord变成VectorRecord, 送给azusa_server的insert方法
    """
    # VectorRecord形如:
    # {
    #   "id": 0,
    #   "vector": [...],
    #   "metadata": {
    #     "doc_id": "...",
    #     "chunk_id": "...",
    #     "source": "...",
    #     "page": null
    #   }
    # }

    # EmbeddingRecord是一个dataclass, 不是dict, 用"."访问字段, 它形如:
    # {
    #     doc_id    
    #     chunk_id    
    #     chunk_index
    #     source
    #     embedding
    #     page
    #     text
    # }
    return{
        "id": record_id,
        "vector": embedding_record.embedding,
        "metadata": {
            "doc_id": embedding_record.doc_id,
            "chunk_id": embedding_record.chunk_id,
            "source": embedding_record.source,
            "page": embedding_record.page,
        },
    }

    

def search(query: str, embedding_records: list[EmbeddingRecord],top_k: int, mode="brute"):
    """
    完整的搜索流程.
    """
    # query -> query_vector
    query_vec = embed_query(query=query) # list[float]
    # embedding_records -> search_vectors
    search_vecs = embedvecs_to_searchvecs(embedding_record_list=embedding_records)
    # 调检索核，拿回 top-k id
    if mode == "brute":
        top_k_list = search_core_brute(query_vec=query_vec, search_vecs=search_vecs, top_k=top_k)
    elif mode == "hnsw":
        # 补充逻辑: 确保 azusa server里已经存放了embedding_records. 如果没有, 那就往里放数据.
        ensure_hnsw_index_loaded(embedding_records)
        top_k_list = search_core_hnsw(query_vec=query_vec, search_vecs=search_vecs, top_k=top_k)
    else:
        raise ValueError("Search mode must be brute or hnsw")
    # 用 id 回查原始 EmbeddingRecord
    result_embedrcd_list = []
    for top_k_id in top_k_list:
        result_embedrcd_list.append(embedding_records[top_k_id])
    # 返回结果
    return result_embedrcd_list

def main():
    query_str = get_query()
    embedding_records_list = load_embedding_records(EMBEDDING_OUTPUT_PATH)
    top_k = 3

    # 补充逻辑:EmbeddingRecord 映射成 适用于Azusa的 VectorRecord

    result_list = search(query=query_str, embedding_records=embedding_records_list, top_k=top_k, mode="brute")
    # 补充逻辑: 交给Azusa搜索, 得到id和其他内容, 然后映射回原始的 EmbeddingRecord

    for result in result_list:
        print(result.text) # dataclass不是字典,需要用.访问, 而不是[]访问
    
if __name__ == "__main__":
    main()


