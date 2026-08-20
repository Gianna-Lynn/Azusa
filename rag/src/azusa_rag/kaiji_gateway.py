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

    # TODO 1：完成请求参数契约。
    # - query 必须是非空、非纯空白字符串，成功时返回 strip 后的文本；
    # - k 必须是 int 但不能是 bool，并满足 1 <= k <= MAX_K；
    # - mode 必须属于 SEARCH_MODES；
    # - 每种参数错误都抛 GatewayError，HTTP 400 且 retryable=False。
    raise NotImplementedError("TODO 1：实现搜索请求校验")


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

        # TODO 2：
        # 1. 先调用 validate_search_request；
        # 2. 只调用一次 self._search_function(query, records, k, mode)；
        # 3. 每条 EmbeddingRecord 映射 chunk_id/text/score/source/page/metadata；
        # 4. source_map 存在时，按结果顺序保存 retrieval_source；
        # 5. 未预期的后端异常转成 retryable=True 的 HTTP 503 GatewayError。
        raise NotImplementedError("TODO 2：实现检索门面与证据字段映射")

    def get_chunk_by_id(self, chunk_id: Any) -> dict[str, Any]:
        """按稳定 chunk_id 返回完整证据；非法或未知标识不可重试。"""

        # TODO 3：
        # - 非字符串、空或纯空白 ID 返回不可重试 HTTP 400；
        # - 使用 _records_by_chunk_id 做精确查找，不做模糊或字符串强转；
        # - 未知 ID 返回不可重试 HTTP 404；
        # - 成功响应包含 chunk_id/text/source/page 与 doc_id/chunk_index metadata。
        raise NotImplementedError("TODO 3：实现稳定证据读取")


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
        # TODO 4a：只接受 POST /search_azusa；使用 _read_json 解析对象，调用
        # Service；GatewayError 保留状态码，其他异常返回不泄露堆栈的可重试 500。
        self._write_error(
            GatewayError(
                "todo_post_route",
                "TODO 4a：实现 POST /search_azusa",
                status=HTTPStatus.NOT_IMPLEMENTED,
                retryable=False,
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
        # TODO 4b：只接受 GET /chunks/{chunk_id}，对单个路径段做 URL 解码；
        # 未知路由返回结构化 404，Service 的 GatewayError 保留自身状态与策略。
        self._write_error(
            GatewayError(
                "todo_get_route",
                "TODO 4b：实现 GET /chunks/{chunk_id}",
                status=HTTPStatus.NOT_IMPLEMENTED,
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
