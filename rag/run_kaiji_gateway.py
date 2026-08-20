"""从指定的 Embedding JSONL 启动供 Kaiji 调用的语义 HTTP 网关。"""

from __future__ import annotations

import argparse
from pathlib import Path

from bootstrap import ensure_src_path

ensure_src_path()

from azusa_rag.config import EMBEDDING_OUTPUT_PATH
from azusa_rag.kaiji_gateway import KaijiGatewayService, build_http_server
from azusa_rag.loaders import load_embedding_records


def parse_args() -> argparse.Namespace:
    """解析监听地址与可复现的 Embedding 输入路径。"""

    parser = argparse.ArgumentParser(description="启动 Kaiji 语义检索网关。")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3010)
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=EMBEDDING_OUTPUT_PATH,
        help="Embedding JSONL 路径；默认读取正常流水线产物。",
    )
    return parser.parse_args()


def main() -> None:
    """加载指定证据集并持续提供 HTTP 服务，直到用户中断。"""

    args = parse_args()
    service = KaijiGatewayService(load_embedding_records(args.embedding_path))
    server = build_http_server(args.host, args.port, service=service)
    print(
        f"Azusa Kaiji 网关正在监听 http://{args.host}:{args.port}，"
        f"已加载 {len(server.service._records)} 条记录：{args.embedding_path}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
