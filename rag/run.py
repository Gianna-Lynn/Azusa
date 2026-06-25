"""
统一的 RAG 命令入口。

目标：
1. 把越来越多的 ``run_*.py`` 收拢到一个入口下
2. 保留旧脚本兼容性，不强制立刻迁移
3. 让 README 只需要教一套命令形态：``python run.py <command> ...``
"""

from __future__ import annotations

import argparse
import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from bootstrap import ensure_src_path

ensure_src_path()


@dataclass(frozen=True)
class CommandSpec:
    name: str
    module_name: str
    help_text: str


COMMAND_SPECS = [
    CommandSpec("chunk", "run_main", "Build chunks from data/raw into output/artifacts/chunks.jsonl."),
    CommandSpec("embed", "run_embed", "Build embeddings from chunks into output/artifacts/embedding.jsonl."),
    CommandSpec("search", "run_search", "Run one retrieval query and write output/search_runs/latest/."),
    CommandSpec("agent", "run_agent", "Run Search Agent v0 and write output/search_runs/latest/."),
    CommandSpec("eval", "run_eval", "Run retrieval eval and write output/eval_runs/latest/."),
    CommandSpec("workflow", "run_workflow", "Run the consolidated workflow and write output/eval_runs/latest/."),
    CommandSpec("import-hf-eval", "run_import_hf_eval", "Import a Hugging Face QA sample snapshot."),
    CommandSpec("import-du-retrieval", "run_import_du_retrieval", "Convert a DuRetrieval snapshot into passages/eval jsonl."),
    CommandSpec("download-du-retrieval", "run_download_du_retrieval", "Download a local DuRetrieval sample snapshot."),
    CommandSpec("du-retrieval-workflow", "run_du_retrieval_workflow", "Run the DuRetrieval end-to-end workflow."),
]

COMMAND_BY_NAME = {spec.name: spec for spec in COMMAND_SPECS}


@contextmanager
def patched_argv(command_name: str, argv: list[str]) -> Iterator[None]:
    """
    让既有入口脚本在不改签名的情况下，也能被统一入口安全转发。
    """
    original_argv = sys.argv[:]
    sys.argv = [command_name, *argv]
    try:
        yield
    finally:
        sys.argv = original_argv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified Azusa RAG command runner.",
    )
    subparsers = parser.add_subparsers(dest="command")

    for spec in COMMAND_SPECS:
        subparsers.add_parser(spec.name, help=spec.help_text)

    return parser


def dispatch_command(command_name: str, forwarded_argv: list[str]) -> None:
    spec = COMMAND_BY_NAME[command_name]
    module = importlib.import_module(spec.module_name)
    command_main = getattr(module, "main")

    with patched_argv(command_name, forwarded_argv):
        command_main()


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args, forwarded_argv = parser.parse_known_args(argv)

    if args.command is None:
        parser.print_help()
        return

    dispatch_command(args.command, forwarded_argv)


if __name__ == "__main__":
    main()
