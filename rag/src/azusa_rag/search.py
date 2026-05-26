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
from rank_bm25 import BM25Okapi  # BM25 是一种经典的稀疏检索算法, 这里用它来实现最小 sparse 检索内核. 以后可以替换成更复杂的 sparse 检索器, 比如基于倒排索引的 BM25 或者 TF-IDF.
import json
import re
import urllib.request
import warnings

from .reranker import score_query_passage_pairs

BM25_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
# [a-z0-9]+：匹配一段连续的英文小写字母或数字，至少 1 个.  | 表示"或者"
# [\u4e00-\u9fff]：匹配一个中文汉字字符
# tokenizer里面已经做了text.lower()了, 所以这里直接在正则里写小写字母就行了.
# 例如 "Hello, Azusa! 123 你好" 会被切成 ["hello", "azusa", "123", "你", "好"].

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
    query_lower = query.lower()
    text_lower = text.lower()
    query_fragments = ts030_collect_query_fragments(query_lower)
    
    score = 0
    for query_fragment in query_fragments:   # 检查query里面的关键句子是不是在chunk文本中出现过
        if query_fragment in text_lower:  # 只要 query_fragment 是 text_lower 的一个子串, 就加分
            score += 1
    # TODO:
    # 目前需要query_fragment是text_lower中的子串才能加分, 需要原样出现, 过于严格. 可以把query_fragment进一步往下切分.
    return score

def ts030_collect_query_fragments(query: str) -> list[str]:
    """
    把 query 按行切开, 保留非空片段, 供 sparse / rerank 共用.
    """
    query_lines = query.lower().splitlines()
    query_fragments = []
    for line in query_lines:
        line = line.strip()
        if not line:
            continue
        query_fragments.append(line)
    return query_fragments

def ts030_tokenize_for_bm25(text: str) -> list[str]:
    """
    为最小BM25接入准备的轻量tokenizer.
    - 英文和数字按照连续片段切分
    - 中文先按照单字切分, 这样不用额外的分词库, 也能有最小可用分词效果
    """
    text_lower = text.lower()
    token_list = BM25_TOKEN_PATTERN.findall(text_lower)
    filtered_tokens = []
    for token in token_list:
        if not token.strip():
            continue
        filtered_tokens.append(token)
    return filtered_tokens

def ts030_build_bm25_index(
    embedding_records: list[EmbeddingRecord],
) -> tuple[BM25Okapi, list[list[str]]]:
    """
    从当前 embedding_records 构建一个最小 BM25 索引.
    当前没有实现缓存, 优先保证接通和结果可见.
    """
    tokenized_corpus = []  # corpus 是语料库的意思. 该列表的每条记录, 都对应一个token列表(该列表式一个嵌套列表)
    for embedding_record in embedding_records:
        tokenized_corpus.append(ts030_tokenize_for_bm25(embedding_record.text))  # embedding_record.text 取出当前记录的正文, 然后送进分词函数, 得到一个token列表, 把这个token列表加入embedding_record
    bm25_index = BM25Okapi(tokenized_corpus)  # 生成一个BM25Okapi索引对象.可以利用这个对象对query打分和排序.
    return bm25_index, tokenized_corpus

def ts030_collect_query_tokens(query: str) -> list[str]:
    """
    给最小 rerank 用的轻量 token 列表.
    这里只做很粗糙的切分, 先够用即可.
    """
    query_tokens = re.findall(r"\w+", query.lower()) # 用正则提取所有词字符片段, 得到token列表. 输入：Hello, Azusa 123 输出：["hello", "azusa", "123"]
    filtered_tokens = []
    for token in query_tokens:
        if len(token) <= 1: # 去掉太短的token, 比如a, l, 1
            continue
        filtered_tokens.append(token)
    return filtered_tokens

def ts030_rerank_score(query: str, text: str) -> int:
    """
    最小 rerank 打分:
    - 优先看整句片段是否命中
    - 再补一个很轻量的 token overlap
    """
    text_lower = text.lower()
    query_fragments = ts030_collect_query_fragments(query) # 取得整句片段
    query_tokens = ts030_collect_query_tokens(query)  # 统计片段长度

    fragment_hit_score = 0
    fragment_length_score = 0
    for query_fragment in query_fragments:
        if query_fragment in text_lower:
            fragment_hit_score += 1    # 如果整句命中, 加1000分
            fragment_length_score += len(query_fragment)    # 整句命中的前提下, 整句越长, 说明匹配越具体, 分数越高. 长度值每多1, 加10分. 

    token_overlap_score = 0
    for query_token in query_tokens:  
        if query_token in text_lower:
            token_overlap_score += 1   # 如果token命中, 每个token加1分

    return fragment_hit_score * 1000 + fragment_length_score * 10 + token_overlap_score

def ts030_rerank_candidates(
    query: str,
    candidate_ids: list[int],
    embedding_records: list[EmbeddingRecord],
    top_k: int,
) -> list[int]:
    """
    对候选集合做一次 rerank.
    优先尝试模型 reranker; 如果模型不可用或推理失败, 再回退到规则 rerank.
    同分时保持原始召回顺序稳定.
    """
    try:
        candidate_texts = []
        for candidate_id in candidate_ids:
            candidate_texts.append(embedding_records[candidate_id].text)

        # 这里把候选 id 映射回原文文本, 再交给 cross-encoder 做逐对打分.
        model_scores = score_query_passage_pairs(query, candidate_texts)
        if len(model_scores) != len(candidate_ids):
            raise ValueError("Reranker score count does not match candidate count.")

        rerank_items = []
        for original_rank, (candidate_id, rerank_score) in enumerate(zip(candidate_ids, model_scores)):
            rerank_items.append((rerank_score, original_rank, candidate_id))

        # 模型分数优先; 如果分数相同, 仍然尽量保留原始召回顺序, 避免结果抖动.
        rerank_items.sort(key=lambda item: (-item[0], item[1], item[2])) # 先看rerank分数, 分数高的在前面; 如果分数一样, 看原始召回顺序, 越早被召回的越靠前; 如果前两项都一样, 看id, id越小的越靠前

        reranked_ids = []
        for (_, _, candidate_id) in rerank_items:
            reranked_ids.append(candidate_id)

        return reranked_ids[:top_k]
    except Exception as error:
        warnings.warn(
            f"BGE reranker failed, falling back to rule-based rerank: {error}"
        )

    # fallback 的设计目标不是追求高精度, 而是保证 hybrid_rerank 模式始终可用.
    rerank_items = []
    for original_rank, candidate_id in enumerate(candidate_ids):
        record = embedding_records[candidate_id]
        rerank_score = ts030_rerank_score(query, record.text)
        rerank_items.append((rerank_score, original_rank, candidate_id))

    rerank_items.sort(key=lambda item: (-item[0], item[1], item[2])) # 先看rerank分数, 分数高的在前面; 如果分数一样, 看原始召回顺序, 越早被召回的越靠前; 如果前两项都一样, 看id, id越小的越靠前

    reranked_ids = []
    for (_, _, candidate_id) in rerank_items:
        reranked_ids.append(candidate_id)

    return reranked_ids[:top_k]

def ts030_search_sparse(query: str, embedding_records_list : list[EmbeddingRecord], top_k = 3) -> list[int]:
    """
    最小 BM25 sparse 检索内核.
    在 Python 侧临时构建 BM25索引, 返回top-k的id
    """
    query_tokens = ts030_tokenize_for_bm25(query)
    if not query_tokens:
        return []

    bm25_index, _ = ts030_build_bm25_index(embedding_records_list)
    score_array = bm25_index.get_scores(query_tokens)
    
    score_id_list = []
    # 旧逻辑: 用 ts030_sparse_score(query, record.text) 算分 (直接用文本比对, 而不是取向量)
    # for index, embedding_record in enumerate(embedding_records_list):
    #     score = ts030_sparse_score(query, embedding_record.text)
    #     record_id = index
    #     score_id_list.append((score, record_id))

    for record_id, score in enumerate(score_array):
        score_id_list.append((float(score), record_id))
    
    score_id_list.sort(key= lambda item:(-item[0], item[1]))
    top_k_list = []
    for (_, record_id) in score_id_list:
        top_k_list.append(record_id)
    return top_k_list[:top_k]

def ts030_hybrid_merge(dense_ids: list[int], sparse_ids: list[int], top_k: int) -> list[int]:
    """
    最小 hybrid merge.
    混合 dense(语义相似, 如 HNSW) 和 sparse(字面匹配) 的搜索结果.
    当前只做“按顺序合并 + 去重 + 截断”, 不做加权打分.
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

    # dense 结果先保留原顺序, sparse 只负责补充 dense 没召回到的候选.
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
    """
    为 hybrid 结果补充来源标记, 方便观察每条结果来自 dense、sparse 还是两者都有.
    """
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
    hybrid 专用 helper.
    先分别跑稠密检索(dense, 如HNSW, 是语义向量检索)和稀疏检索(sparse, 如BM25, 是传统关键词检索), 
    再合并结果, 最后返回 id 列表和来源标记.
    """
    query_vec = embed_query(query) # 把 query 做成向量
    search_vecs = build_search_vecs(embedding_records) # 再把所有 embedding_records 转成检索用的向量列表
    ensure_hnsw_index_loaded(embedding_records) # hybrid 的 dense 部分使用了 HNSW, 因此需要先保证 server 端索引已装载.
    dense_ids =  search_core_hnsw(query_vec, search_vecs, top_k)    # 用 dense 检索拿一批结果
    sparse_ids = ts030_search_sparse(query, embedding_records, top_k)   # 用 sparse 检索再拿一批结果
    merged_ids = ts030_hybrid_merge(dense_ids, sparse_ids, top_k)   # 把 dense 和 sparse 结果合并去重
    source_map = ts030_build_source_map(merged_ids, dense_ids, sparse_ids) # 给每个结果补一个来源标签
    return merged_ids, source_map

def ts030_search_hybrid_rerank_with_sources(
    query: str,
    embedding_records: list[EmbeddingRecord],
    top_k: int,
) -> tuple[list[int], list[dict]]:
    """
    基于ts030_search_hybrid_with_sources做新增和修改.
    实现最小 hybrid rerank.
    先扩大 dense + sparse 的候选召回, 再用 reranker 对候选集做二次重排.
    """
    # 新增:把候选集从top_k扩大到candidate_k
    # 这里先拿到一个“更宽”的候选集:
    # 当前 hybrid merge 仍然是 dense 优先、sparse 补位的合并策略,
    # 因此前排候选天然会带一点 dense 的主题偏向, 不一定已经是最终最优排序.
    # 如果 candidate_k 取得太小, 某些“本来相关, 但被 merge 倾向压后”的 chunk
    # 可能连 rerank 的入场资格都拿不到; 先把候选池放宽, reranker 才有空间纠偏,
    # 把原本排位靠后的正确结果重新提回 top-k.
    candidate_k = min(len(embedding_records), max(20, top_k * 5))
    
    query_vec = embed_query(query)
    search_vecs = build_search_vecs(embedding_records)
    ensure_hnsw_index_loaded(embedding_records)
    dense_ids = search_core_hnsw(query_vec, search_vecs, candidate_k)
    sparse_ids = ts030_search_sparse(query, embedding_records, candidate_k)
    merged_ids = ts030_hybrid_merge(dense_ids, sparse_ids, candidate_k)
    
    # 新增:用rerank重排序一次
    reranked_ids = ts030_rerank_candidates(
        query=query,
        candidate_ids=merged_ids,
        embedding_records=embedding_records,
        top_k=top_k,
    )

    source_map = ts030_build_source_map(reranked_ids, dense_ids, sparse_ids)
    return reranked_ids, source_map

def ensure_hnsw_index_loaded(embedding_records: list[EmbeddingRecord]) -> None:
    """
    确保 Rust HNSW server 里已经装入了当前这批 embedding 数据.
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
    把 EmbeddingRecord 变成 azusa_server /insert 所需的 VectorRecord 字典.
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


def search(
        query: str, 
        embedding_records: list[EmbeddingRecord],
        top_k: int, 
        mode="brute"
    ) -> list[EmbeddingRecord]:
    """
    完整的搜索流程.
    按 mode 选择 brute / hnsw / hybrid / hybrid_rerank, 最终返回 EmbeddingRecord 列表.
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
    elif mode == "hybrid_rerank":
        # hybrid_rerank 在 hybrid 召回的基础上再走一次 rerank 重排.
        top_k_list, _ = ts030_search_hybrid_rerank_with_sources(query, embedding_records, top_k)
    elif mode == "bm25":
        top_k_list = ts030_search_sparse(query, embedding_records, top_k)
    else:
        raise ValueError("Search mode must be bm25, brute, hnsw, hybrid,or hybrid_rerank")
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
