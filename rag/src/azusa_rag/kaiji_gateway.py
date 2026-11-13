"""Kaiji 语义网关：请求校验、检索与证据映射、HTTP 路由及结构化错误。"""

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
    if not isinstance(mode, str) or mode not in SEARCH_MODES:
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
            record.chunk_id: record for record in embedding_records
        }
        self._search_function = search_function

    def search_azusa(self, query: Any, k: Any, mode: Any) -> dict[str, Any]:
        """执行检索并映射完整证据字段；后端异常统一为可重试 503。"""


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


        hits = list()

        for index, embedding_record in enumerate(EmbeddingRecord_list):
            retrieval_source = None
            if source_map_list is not None and index < len(source_map_list):
                retrieval_source = source_map_list[index].get("source")
            hits.append({
                "chunk_id": embedding_record.chunk_id,
                "text": embedding_record.text,
                "score": None,
                "source": embedding_record.source,
                "page": embedding_record.page,
                "metadata": {
                    "doc_id": embedding_record.doc_id,
                    "chunk_index": embedding_record.chunk_index,
                    "retrieval_source": retrieval_source,
                },
            })

        return {"mode":mode, "hits":hits}

    def get_chunk_by_id(self, chunk_id: Any) -> dict[str, Any]:
        """按稳定 chunk_id 读取证据；非法或未知 ID 返回结构化错误。"""


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

        record = self._records_by_chunk_id.get(chunk_id)


        if record is None:
            raise GatewayError(
                code="chunk_not_found",
                message=f"get record by chunk_id found unknown chunk_id",
                status=HTTPStatus.NOT_FOUND,
                retryable=False,
            )
        return {

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
    """按稳定 chunk_id 读取证据；非法或未知 ID 返回结构化错误。"""

    return get_default_service().get_chunk_by_id(chunk_id)


class KaijiGatewayHTTPServer(ThreadingHTTPServer):
    """将语义服务绑定到支持并发请求的 HTTP Server。"""

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

            json_dict = self._read_json()


            query = json_dict.get("query")
            k = json_dict.get("k")
            mode = json_dict.get("mode")


            result = self.server.service.search_azusa(query, k, mode)

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


        try:
            prefix = "/chunks/"
            if path[:len(prefix)] != prefix:
                raise GatewayError(
                    "route_not_found",
                    f"{path} not found",
                    status=HTTPStatus.NOT_FOUND,
                    retryable=False,
                )
            encoded_chunk_id = path[len(prefix):]
            chunk_id = unquote(encoded_chunk_id)
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
