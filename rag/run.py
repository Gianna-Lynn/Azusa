"""
统一的 RAG 命令入口。

目标：
1. 把越来越多的 ``run_*.py`` 收拢到一个入口下
2. 从包内 cli 模块加载命令，不再依赖根目录 run_*.py
3. 让 README 只需要教一套命令形态：``python run.py <command> ...``
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from dataclasses import dataclass
# 仅独立脚本需要定位 src；包内命令不再修改导入路径。
src_path = str(Path(__file__).resolve().parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


@dataclass(frozen=True)
class CommandSpec:
    name: str
    module_name: str
    help_text: str


COMMAND_SPECS = [
    CommandSpec("gateway", "azusa_rag.cli.gateway", "Serve the Kaiji semantic retrieval HTTP gateway."),
    CommandSpec("chunk", "azusa_rag.cli.chunk", "Build chunks from data/raw into output/artifacts/chunks.jsonl."),
    CommandSpec("embed", "azusa_rag.cli.embed", "Build embeddings from chunks into output/artifacts/embedding.jsonl."),
    CommandSpec("kb-sync", "azusa_rag.cli.kb_sync", "Incrementally synchronize multi-format source documents and embeddings."),
    CommandSpec("search", "azusa_rag.cli.search", "Run one retrieval query and write output/search_runs/latest/."),
    CommandSpec("agent", "azusa_rag.cli.agent", "Run Search Agent v0 and write output/search_runs/latest/."),
    CommandSpec("agent-eval", "azusa_rag.cli.agent_eval", "Run the Stage 3 Agent sample set and save structured traces."),
    CommandSpec("eval", "azusa_rag.cli.eval", "Run retrieval eval into a new output/eval_runs/runs/ directory."),
    CommandSpec("workflow", "azusa_rag.cli.workflow", "Run the consolidated workflow and write output/eval_runs/latest/."),
    CommandSpec("import-hf-eval", "azusa_rag.cli.import_hf_eval", "Import a Hugging Face QA sample snapshot."),
    CommandSpec("import-du-retrieval", "azusa_rag.cli.import_du_retrieval", "Convert a DuRetrieval snapshot into passages/eval jsonl."),
    CommandSpec("download-du-retrieval", "azusa_rag.cli.download_du_retrieval", "Download a local DuRetrieval sample snapshot."),
    CommandSpec("du-retrieval-workflow", "azusa_rag.cli.du_retrieval_workflow", "Run the DuRetrieval end-to-end workflow."),
    CommandSpec("benchmark-lab", "azusa_rag.cli.benchmark_lab", "Run retrieval mode/top-k/chunk-artifact experiment matrices."),
]

COMMAND_BY_NAME = {spec.name: spec for spec in COMMAND_SPECS}


def build_parser() -> argparse.ArgumentParser:
    """构造统一入口及其全部子命令的参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Unified Azusa RAG command runner.",
    )
    subparsers = parser.add_subparsers(dest="command")

    for spec in COMMAND_SPECS:
        # 子命令自己的模块负责完整参数与 --help；这里仅识别命令名并透传其余参数。
        subparsers.add_parser(spec.name, help=spec.help_text, add_help=False)

    return parser


def dispatch_command(command_name: str, forwarded_argv: list[str]) -> None:
    """加载目标子命令模块，并把剩余参数原样转交给它。"""
    spec = COMMAND_BY_NAME[command_name]
    module = importlib.import_module(spec.module_name)
    command_main = getattr(module, "main")

    command_main(forwarded_argv)


def main(argv: list[str] | None = None) -> None:
    """解析统一入口参数，并执行用户选择的子命令。"""
    parser = build_parser()
    args, forwarded_argv = parser.parse_known_args(argv)

    if args.command is None:
        parser.print_help()
        return

    dispatch_command(args.command, forwarded_argv)


if __name__ == "__main__":
    main()
