from __future__ import annotations

import argparse

from bootstrap import ensure_src_path

ensure_src_path()

from azusa_rag.kaiji_gateway import build_http_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Kaiji semantic gateway.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3010)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = build_http_server(args.host, args.port)
    print(
        f"Azusa Kaiji gateway listening on http://{args.host}:{args.port} "
        f"with {len(server.service._records)} records"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
