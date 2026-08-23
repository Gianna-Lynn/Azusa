"""提供 Kaiji 语义网关的教学脚手架，明确参数、证据字段、路由与错误契约。"""

from __future__ import annotations

import json
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import unquote, urlparse

from .config import EMBEDDING_OUTPUT_PATH
from .loaders import load_embedding_records
from .modes import SEARCH_MODES
from .schemas import EmbeddingRecord

MAX_REQUEST_BYTES = 64 * 1024
MAX_K = 20

SearchFunction = Callable[
    [str, list[EmbeddingRecord], int, str],
    tuple[list[EmbeddingRecord], list[dict] | None],
]


def _default_search_function(
    query: str,
    embedding_records: list[EmbeddingRecord],
    top_k: int,
    mode: str,
) -> tuple[list[EmbeddingRecord], list[dict] | None]:
    # 只有真实默认后端收到请求时才导入模型相关模块，让契约测试保持轻量。
    from .search import search_records_with_sources

    return search_records_with_sources(query, embedding_records, top_k, mode)


class GatewayError(Exception):
    """表示可安全返回给客户端、并携带 HTTP 与重试策略的失败。"""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: HTTPStatus,
        retryable: bool,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.retryable = retryable

    def to_dict(self) -> dict[str, Any]:
        """把错误转换为稳定的 JSON 响应外壳。"""

        return {
            "error": {
                "code": self.code,
                "message": str(self),
                "retryable": self.retryable,
            }
        }


def validate_search_request(query: Any, k: Any, mode: Any) -> tuple[str, int, str]:
    """校验并规范化语义检索参数；所有调用方错误均不可重试。"""

    # AZUSA-GW-01：完成请求参数契约。
    # - query 必须是非空、非纯空白字符串，成功时返回 strip 后的文本；
    # - k 必须是 int 但不能是 bool，并满足 1 <= k <= MAX_K；
    # - mode 必须属于 SEARCH_MODES；
    # - 每种参数错误都抛 GatewayError，HTTP 400 且 retryable=False。
    if not isinstance(query, str):
        raise GatewayError(
            code="invalid_query",
            message=f"Invalid query. Expected query to be str, however query is {type(query)}.",
            status=HTTPStatus.BAD_REQUEST,
            retryable=False,
        )
    if not query.strip():
        raise GatewayError(
            code="invalid_query",
            message=f"Invalid query. Query should be non-blank, non-empty str.",
            status=HTTPStatus.BAD_REQUEST,
            retryable=False,
        )

    if not (isinstance(k, int) and not isinstance(k, bool)):
        raise GatewayError(
            "invalid_k",
            f"Invalid k. Expected k to be int, however k is {type(k)}.",
            status=HTTPStatus.BAD_REQUEST,
            retryable=False,
        )
    if not (1 <= k and k <= MAX_K):
        raise GatewayError(
            "invalid_k",
            f"Invalid k. Expected 1 <= k <= MAX_K, however k is {k}.",
            status=HTTPStatus.BAD_REQUEST,
            retryable=False,
        )
    if mode not in SEARCH_MODES:
        raise GatewayError(
            "invalid_mode",
            f"Invalid mode. Expected mode must be one of {','.join(SEARCH_MODES)}, however mode is {type(mode)}.",
            status=HTTPStatus.BAD_REQUEST,
            retryable=False,
        )    
    return query.strip(), k, mode

class KaijiGatewayService:
    """把 Azusa 检索函数与记录表包装为 Kaiji 所需的两个语义操作。"""

    def __init__(
        self,
        embedding_records: list[EmbeddingRecord],
        *,
        search_function: SearchFunction = _default_search_function,
    ) -> None:
        self._records = embedding_records
        self._records_by_chunk_id = {
            record.chunk_id: record for record in embedding_records # 这里的key是字符串, 形如"stromules-1962", 而不是整数.
        }
        self._search_function = search_function

    def search_azusa(self, query: Any, k: Any, mode: Any) -> dict[str, Any]:
        """执行检索并映射完整证据字段；后端异常统一为可重试 503。"""

        # AZUSA-GW-02：
        # 1. 先调用 validate_search_request；
        # 2. 只调用一次 self._search_function(query, records, k, mode)；
        # 3. 每条 EmbeddingRecord 映射 chunk_id/text/score/source/page/metadata；
        # 4. source_map 存在时，按结果顺序保存 retrieval_source；
        # 5. 未预期的后端异常转成 retryable=True 的 HTTP 503 GatewayError。
        query, k, mode = validate_search_request(query, k, mode)
        try:
            EmbeddingRecord_list, source_map_list= self._search_function(query, self._records, k, mode)
        except GatewayError:
            raise
        except Exception as exc:
            raise GatewayError(
                code="search_backend_failed",
                message=f"Azusa search backend failed: {exc}",
                status=HTTPStatus.SERVICE_UNAVAILABLE,
                retryable=True,
            )
        # source_map 如果不是None, 根据rag\src\azusa_rag\search_hybrid.py中的build_source_map定义, source_map应当形如:
        # [
        #     {"id": 12, "source": "dense + sparse"},
        #     {"id": 35, "source": "dense"}, 
        #     {"id": 43, "source": "sparse"},
        # ]
        hits = list()
        # 为避免source_map = None的情况, 使用索引读取source_map
        for index, embedding_record in enumerate(EmbeddingRecord_list):
            retrieval_source = None
            if source_map_list is not None and index < len(source_map_list):
                retrieval_source = source_map_list[index].get("source")
                hits.append({
                    "chunk_id":embedding_record.chunk_id,   
                    "text":embedding_record.text,
                    "score":None, # Azusa定义的EmbeddingRecord没有score字段.
                    "source":embedding_record.source,
                    "page":embedding_record.page,
                    "metadata":{
                        "doc_id":embedding_record.doc_id,
                        "chunk_index":embedding_record.chunk_index,
                        "retrieval_source":retrieval_source,    # 从这个条目当中取回: {"id": 35, "source": "bm25 sparse"},
                    },
                })

        return {"mode":mode, "hits":hits}

    def get_chunk_by_id(self, chunk_id: Any) -> dict[str, Any]:
        """按稳定 chunk_id 返回完整证据；非法或未知标识不可重试。"""

        # AZUSA-GW-03：
        # - 非字符串、空或纯空白 ID 返回不可重试 HTTP 400；
        # - 使用 _records_by_chunk_id 做精确查找，不做模糊或字符串强转；
        # - 未知 ID 返回不可重试 HTTP 404；
        # - 成功响应包含 chunk_id/text/source/page 与 doc_id/chunk_index metadata。
        
        if not isinstance (chunk_id, str):
            raise GatewayError(
                code="invalid_chunk_id",
                message=f"chunk_id should be str, however chunk_id is {type(chunk_id)}",
                status=HTTPStatus.BAD_REQUEST,
                retryable=False,
            )
        if not chunk_id.strip():
            raise GatewayError(
                code="invalid_chunk_id",
                message=f"chunk_id should be non-blank str",
                status=HTTPStatus.BAD_REQUEST,
                retryable=False,
            )
        
        record = self._records_by_chunk_id.get(chunk_id) # record.chunk_id: record for record in embedding_records
        # self._records_by_chunk_id 形如:
        # {
        #     "chunk-A": record_A,
        #     "chunk-B": record_B,
        #     "chunk-C": record_C,
        #     "chunk-D": record_D,
        # }
        # 调用.get(chunk_id)之后, 会得到一个唯一确定的chunk记录record.
        if record is None:
            raise GatewayError(
                code="chunk_not_found",
                message=f"get record by chunk_id found unknown chunk_id",
                status=HTTPStatus.NOT_FOUND,
                retryable=False,
            )
        return {
            # 成功响应包含 chunk_id/text/source/page 与 doc_id/chunk_index metadata。
            "chunk_id" : chunk_id,
            "text": record.text,
            "source": record.source,
            "page": record.page,
            "metadata":{
                "doc_id": record.doc_id,
                "chunk_index": record.chunk_index
            }
        }

@lru_cache(maxsize=1)
def get_default_service() -> KaijiGatewayService:
    """惰性加载正常流水线产物，并在进程内复用默认服务。"""

    return KaijiGatewayService(load_embedding_records(EMBEDDING_OUTPUT_PATH))


def search_azusa(query: str, k: int, mode: str) -> dict[str, Any]:
    """提供任务书要求的进程内稳定检索入口。"""

    return get_default_service().search_azusa(query, k, mode)


def get_chunk_by_id(chunk_id: str) -> dict[str, Any]:
    """提供任务书要求的进程内稳定证据读取入口。"""

    return get_default_service().get_chunk_by_id(chunk_id)


class KaijiGatewayHTTPServer(ThreadingHTTPServer):
    """把语义服务附着到支持并发请求的教学 HTTP Server。"""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        service: KaijiGatewayService,
    ) -> None:
        super().__init__(server_address, KaijiGatewayRequestHandler)
        self.service = service


class KaijiGatewayRequestHandler(BaseHTTPRequestHandler):
    """练习健康检查、语义搜索、证据读取与结构化错误响应。"""

    server: KaijiGatewayHTTPServer
    def do_POST(self) -> None:
        # AZUSA-GW-04a：只接受Kaiji发来的 POST /search_azusa；使用 _read_json 解析对象，调用
        # Service；GatewayError 保留状态码，其他异常返回不泄露堆栈的可重试 500。
        
        # 1. 确认路径是 /search_azusa
        if self.path != "/search_azusa":
            self._write_error(
                GatewayError(
                    "route_not_found",
                    f"path should be '/search_azusa', however path is {self.path}",
                    status=HTTPStatus.NOT_FOUND,
                    retryable=False,
                )
            )
            return
        try:
            # 2. 读取JSON body, 从body中取得 query/k/mode. 
            json_dict = self._read_json()  # json_dict 形如 {'query': 'Kaiji', 'k': 1, 'mode': 'bm25'}. 失败直接抛出GatewayError.
            # F:\Repositories\Kaiji\src\kaiji\adapters\azusa.py 当中定义了Kaiji的post内容
            # response = self._client.post(
            #     url="/search_azusa", 
            #     json={
            #         "query":query,
            #         "k":k,
            #         "mode":mode
            #     }
            # )

            query = json_dict.get("query")
            k = json_dict.get("k")
            mode = json_dict.get("mode")
        
            #.search_azusa(...)可以校验这三个参数.
            result = self.server.service.search_azusa(query, k, mode) # 注意这个地方不能直接调用search_azusa, 它会加载完整的默认 embedding 数据和搜索依赖
            # result 不需要判空, 因为 result可能是空字典.
            self._write_json(
                status=HTTPStatus.OK,
                payload=result,
            )
        except GatewayError as g:
            self._write_error(g)
        except Exception:
            self._write_error(
                GatewayError(
                    "internal_server_error",
                    "internal_server_error",
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    retryable=True,
                )
            )

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._write_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "service": "azusa-kaiji-gateway",
                    "record_count": len(self.server.service._records),
                },
            )
            return
        # AZUSA-GW-04b：只接受 GET /chunks/{chunk_id}，对单个路径段做 URL 解码；
        # 未知路由返回结构化 404，Service 的 GatewayError 保留自身状态与策略。

        try:
            prefix = "/chunks/"
            if path[:len(prefix)] != prefix:        # 非法的path, 形如"/unknown"
                raise GatewayError(
                    "route_not_found",
                    f"{path} not found",
                    status=HTTPStatus.NOT_FOUND,
                    retryable=False,
                )
            encoded_chunk_id = path[len(prefix):]   # 形如 "folder%2Fchunk%201"
            chunk_id = unquote(encoded_chunk_id)    # 形如 "folder/chunk 1"
            result_dict = self.server.service.get_chunk_by_id(chunk_id)
            self._write_json(
                HTTPStatus.OK,
                result_dict,
            )
        except GatewayError as g:
            self._write_error(g)
        except Exception:
            self._write_error(
                GatewayError(
                    "chunk_not_found",
                    f"{chunk_id} not found",
                    status=HTTPStatus.NOT_FOUND,
                    retryable=False,
                )
            )

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        try:
            length = int(raw_length or "0")
        except ValueError as exc:
            raise GatewayError(
                "invalid_content_length",
                "Content-Length must be an integer",
                status=HTTPStatus.BAD_REQUEST,
                retryable=False,
            ) from exc
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise GatewayError(
                "invalid_body_size",
                f"request body must be between 1 and {MAX_REQUEST_BYTES} bytes",
                status=HTTPStatus.BAD_REQUEST,
                retryable=False,
            )
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise GatewayError(
                "invalid_json",
                "request body must be valid UTF-8 JSON",
                status=HTTPStatus.BAD_REQUEST,
                retryable=False,
            ) from exc
        if not isinstance(payload, dict):
            raise GatewayError(
                "invalid_json_object",
                "request JSON must be an object",
                status=HTTPStatus.BAD_REQUEST,
                retryable=False,
            )
        return payload

    def _write_error(self, error: GatewayError) -> None:
        self._write_json(error.status, error.to_dict())

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def build_http_server(
    host: str,
    port: int,
    *,
    service: KaijiGatewayService | None = None,
) -> KaijiGatewayHTTPServer:
    """构造监听指定地址的 HTTP Server，并允许测试注入轻量服务。"""

    return KaijiGatewayHTTPServer(
        (host, port),
        service or get_default_service(),
    )
