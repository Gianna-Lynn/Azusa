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


def read_json_response(response) -> dict:
    """
    把 HTTP 响应体读出来, 并解析成 Python 字典.
    """
    response_body: bytes = response.read()
    response_json = response_body.decode("utf-8")
    response_data = json.loads(response_json)
    return response_data


def get_json(url: str) -> dict:
    """
    对给定 URL 发 GET 请求, 返回解析后的 JSON 字典.
    """
    with urllib.request.urlopen(url) as response:
        return read_json_response(response)


def post_json(url: str, request_dict: dict) -> dict:
    """
    对给定 URL 发 JSON POST 请求, 返回解析后的 JSON 字典.
    """
    request_json = json.dumps(request_dict, ensure_ascii=False)
    request_body = request_json.encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return read_json_response(response)


def post_json_no_response(url: str, request_dict: dict) -> None:
    """
    对给定 URL 发 JSON POST 请求, 但不读取业务返回值.
    用于 /insert 这类只关心状态码的接口.
    """
    request_json = json.dumps(request_dict, ensure_ascii=False)
    request_body = request_json.encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        response.status

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

def build_search_vecs(embedding_record_list: list[EmbeddingRecord])-> list[tuple[int, list[float]]]:
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

    # 2. 调用 HTTP helper，把 query vector 送到 azusa_server 的 /search
    # 按照azusa_server的search_handler, 返回的是 Json<SearchResponse>
    # SearchResponse 的 JSON 形状是: {"results": [{"id": 3, "score": 0.982}, ...]}
    response_data = post_json(SEARCH_URL, request_dict)

    # 3. 提取 results 里的 id
    score_id_list = []
    for hit in response_data["results"]:
        score_id_list.append(hit["id"])

    # 4. 返回 list[int]
    return score_id_list

def ts030_sparse_score(query: str, text: str) -> int:
    """
    给一个 query 和一个 text，返回一个整数分数，表示它们有多少关键词重合。
    """
    query_lower = query.lower() # str(全小写)
    text_lower = text.lower() # str(全小写)
    # query_lower = query_lower.split() # 用空格切词, 得到 list[str]
    # text_lower = text_lower.split() # list[str]
    # query_set = set(query_lower) # 转成集合
    # text_set = set(text_lower)
    # cross_set = query_set & text_set  # 求交集, 算交集大小
    # return len(cross_set)
    query_lines = query_lower.splitlines()  # list[str]
    query_fragments = []
    for line in query_lines:
        line = line.strip()
        if not line:
            continue
        query_fragments.append(line)
    
    score = 0
    for query_fragment in query_fragments:   # 检查query里面的关键句子是不是在chunk文本中出现过
        if query_fragment in text_lower:  # 只要 query_fragment 是 text_lower 的一个子串, 就加分
            score += 1
    # TODO:
    # 目前需要query_fragment是text_lower中的子串才能加分, 需要原样出现, 过于严格. 可以把query_fragment进一步往下切分.
    
    return score

def ts030_search_sparse(query: str, embedding_records_list : list[EmbeddingRecord], top_k = 3) -> list[int]:
    """
    最小sparse检索内核:
    遍历所有记录, 用token overlap打分, 返回 top-k 的 id
    """
    # 遍历每个 embedding_record
    # 用 ts030_sparse_score(query, record.text) 算分 (直接用文本比对, 而不是取向量)
    # 记成 (score, id)
    # 排序
    # 返回 top-k 的 id
    score_id_list = []
    for index, embedding_record in enumerate(embedding_records_list):
        score = ts030_sparse_score(query, embedding_record.text)
        record_id = index
        score_id_list.append((score, record_id))
    score_id_list.sort(key= lambda item:(-item[0], item[1]))
    top_k_list = []
    for (_, record_id) in score_id_list:
        top_k_list.append(record_id)
    return top_k_list[:top_k]

def ts030_hybrid_merge(dense_ids: list[int], sparse_ids: list[int], top_k: int) -> list[int]:
    """
    最小hybrid merge:
    混合dense(语义相似,如HNSW)和sparse(字面匹配,如BM25)的搜索序列结果.
    当前只实现合二为一, 去重截断. 没有加权打分, 也没有二次排序.
    """
    # 输入一份 dense_ids
    # 输入一份 sparse_ids
    # 先保留 dense 的顺序
    # 再补进 sparse 里还没出现过的
    # 最后截断到 top_k
    merged_ids = []
    seen_ids = set()

    for dense_id in dense_ids:
        if dense_id in seen_ids:
            continue
        merged_ids.append(dense_id)
        seen_ids.add(dense_id)

    for sparse_id in sparse_ids:
        if sparse_id in seen_ids:
            continue
        merged_ids.append(sparse_id)
        seen_ids.add(sparse_id)

    return merged_ids[:top_k]

def ts030_build_source_map(
    merged_ids: list[int],
    dense_ids: list[int],
    sparse_ids: list[int],
) -> list[dict]:
    dense_ids_set = set(dense_ids)
    sparse_ids_set = set(sparse_ids)
    source_items = []
    for merged_id in merged_ids:
        if merged_id in dense_ids_set and merged_id in sparse_ids_set:
            source = "dense + sparse"
        elif merged_id in dense_ids_set:
            source = "dense"
        else: 
            source = "sparse"
        source_items.append(
            {
                "id": merged_id,
                "source": source,
            }
        )
    return source_items

def ts030_search_hybrid_with_sources(
    query: str,
    embedding_records: list[EmbeddingRecord],
    top_k: int,
) -> tuple[list[int], list[dict]]:
    """
    hybrid专用的helper:
    返回dense和sparse搜索结果merge后的id列表, 返回每条结果的来源标记.
    """
    query_vec = embed_query(query)
    search_vecs = build_search_vecs(embedding_records)
    ensure_hnsw_index_loaded(embedding_records)
    dense_ids =  search_core_hnsw(query_vec, search_vecs, top_k)
    sparse_ids = ts030_search_sparse(query, embedding_records, top_k)
    merged_ids = ts030_hybrid_merge(dense_ids, sparse_ids, top_k)
    source_map = ts030_build_source_map(merged_ids, dense_ids, sparse_ids)
    return merged_ids, source_map

def ensure_hnsw_index_loaded(embedding_records: list[EmbeddingRecord]) -> None:
    """
    确保 Rust Hnsw 中已经有数据了.
    """
    # 先请求 /health
    # 按照azusa_server的health_handler, 返回的是 Json<HealthResponse>
    # Json<HealthResponse> 形如:  {"status":"ok","record_count":0}
    response_data = get_json(HEALTH_URL)
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
        insert_dict = build_insert_record(embedding_record=embedding_record, record_id=index)

        # 2. 调用 HTTP helper，把这条记录送进 azusa_server
        post_json_no_response(INSERT_URL, insert_dict)

def build_insert_record(
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
    search_vecs = build_search_vecs(embedding_record_list=embedding_records)
    # 调检索核，拿回 top-k id
    if mode == "brute":
        top_k_list = search_core_brute(query_vec=query_vec, search_vecs=search_vecs, top_k=top_k)
    elif mode == "hnsw":
        # 补充逻辑: 确保 azusa server里已经存放了embedding_records. 如果没有, 那就往里放数据.
        ensure_hnsw_index_loaded(embedding_records)
        top_k_list = search_core_hnsw(query_vec=query_vec, search_vecs=search_vecs, top_k=top_k)
    elif mode == "hybrid":
        top_k_list, _ = ts030_search_hybrid_with_sources(query, embedding_records, top_k)
    else:
        raise ValueError("Search mode must be brute, hnsw, or hybrid")
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


