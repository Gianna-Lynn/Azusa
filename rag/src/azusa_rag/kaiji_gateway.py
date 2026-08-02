"""为 Kaiji 提供稳定语义 HTTP 契约，并映射 Azusa 的检索记录与结构化错误。"""

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

    if not isinstance(query, str) or not query.strip():
        raise GatewayError(
            "invalid_query",
            "query must be a non-empty string",
            status=HTTPStatus.BAD_REQUEST,
            retryable=False,
        )
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K:
        raise GatewayError(
            "invalid_k",
            f"k must be an integer between 1 and {MAX_K}",
            status=HTTPStatus.BAD_REQUEST,
            retryable=False,
        )
    if not isinstance(mode, str) or mode not in SEARCH_MODES:
        raise GatewayError(
            "invalid_mode",
            f"mode must be one of: {', '.join(SEARCH_MODES)}",
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
            records, source_map = self._search_function(
                query,
                self._records,
                k,
                mode,
            )
        except GatewayError:
            raise
        except Exception as exc:
            raise GatewayError(
                "search_backend_failed",
                f"Azusa search backend failed: {exc}",
                status=HTTPStatus.SERVICE_UNAVAILABLE,
                retryable=True,
            ) from exc

        hits = []
        for index, record in enumerate(records):
            retrieval_source = None
            if source_map is not None and index < len(source_map):
                retrieval_source = source_map[index].get("source")
            hits.append(
                {
                    "chunk_id": record.chunk_id,
                    "text": record.text,
                    "score": None,
                    "source": record.source,
                    "page": record.page,
                    "metadata": {
                        "doc_id": record.doc_id,
                        "chunk_index": record.chunk_index,
                        "retrieval_source": retrieval_source,
                    },
                }
            )
        return {"hits": hits, "mode": mode}

    def get_chunk_by_id(self, chunk_id: Any) -> dict[str, Any]:
        """按稳定 chunk_id 返回完整证据；非法或未知标识不可重试。"""

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise GatewayError(
                "invalid_chunk_id",
                "chunk_id must be a non-empty string",
                status=HTTPStatus.BAD_REQUEST,
                retryable=False,
            )
        record = self._records_by_chunk_id.get(chunk_id)
        if record is None:
            raise GatewayError(
                "chunk_not_found",
                f"unknown chunk_id: {chunk_id}",
                status=HTTPStatus.NOT_FOUND,
                retryable=False,
            )
        return {
            "chunk_id": record.chunk_id,
            "text": record.text,
            "source": record.source,
            "page": record.page,
            "metadata": {
                "doc_id": record.doc_id,
                "chunk_index": record.chunk_index,
            },
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
    """实现健康检查、语义搜索、证据读取与结构化错误响应。"""

    server: KaijiGatewayHTTPServer

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/search_azusa":
            self._write_error(
                GatewayError(
                    "route_not_found",
                    "route not found",
                    status=HTTPStatus.NOT_FOUND,
                    retryable=False,
                )
            )
            return
        try:
            payload = self._read_json()
            result = self.server.service.search_azusa(
                payload.get("query"),
                payload.get("k"),
                payload.get("mode"),
            )
            self._write_json(HTTPStatus.OK, result)
        except GatewayError as exc:
            self._write_error(exc)
        except Exception:
            self._write_error(
                GatewayError(
                    "internal_error",
                    "unexpected gateway error",
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
        prefix = "/chunks/"
        if not path.startswith(prefix):
            self._write_error(
                GatewayError(
                    "route_not_found",
                    "route not found",
                    status=HTTPStatus.NOT_FOUND,
                    retryable=False,
                )
            )
            return
        try:
            chunk_id = unquote(path[len(prefix) :])
            self._write_json(
                HTTPStatus.OK,
                self.server.service.get_chunk_by_id(chunk_id),
            )
        except GatewayError as exc:
            self._write_error(exc)

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
