"""验证迁移后的入口分发、参数传递与源码身份，不加载模型或改写业务产物。"""

import importlib
import importlib.util
from pathlib import Path
import sys

import pytest

from azusa_rag.cli import embed
from azusa_rag.evaluation_provenance import source_identity


def load_runner():
    """按文件位置加载唯一脚本入口，不依赖当前工作目录。"""
    path = Path(__file__).resolve().parents[1] / "run.py"
    spec = importlib.util.spec_from_file_location("azusa_cli_test_runner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("command", [
    "chunk", "embed", "kb-sync", "search", "gateway", "agent", "agent-eval",
    "eval", "workflow", "benchmark-lab", "import-hf-eval", "import-du-retrieval",
    "download-du-retrieval", "du-retrieval-workflow",
])
def test_every_command_help_exits_before_execution(command, capsys):
    """所有命令都能通过统一入口查看帮助，包含原先遗漏的 gateway。"""
    with pytest.raises(SystemExit) as caught:
        load_runner().main([command, "--help"])
    assert caught.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_embed_custom_paths_are_forwarded_without_mutating_argv(monkeypatch, tmp_path):
    """覆盖旧入口忽略自定义路径的问题，并确认分发不再改写全局 argv。"""
    calls = []
    monkeypatch.setattr(embed, "build_embedding_artifacts", lambda **kwargs: calls.append(kwargs))
    original_argv = sys.argv[:]
    chunks = tmp_path / "chunks.jsonl"
    embeddings = tmp_path / "embedding.jsonl"
    load_runner().main(["embed", "--chunks-path", str(chunks), "--embedding-output-path", str(embeddings)])
    assert calls == [{"chunk_input_path": chunks, "embedding_output_path": embeddings}]
    assert sys.argv == original_argv
    assert list(tmp_path.iterdir()) == []


def test_embed_rejects_unknown_arguments_before_execution(monkeypatch):
    """未知参数不能被忽略后意外编码默认产物。"""
    def unexpected(**kwargs):
        """参数非法时不应进入产物生成。"""
        pytest.fail("unexpected embedding execution")
    monkeypatch.setattr(embed, "build_embedding_artifacts", unexpected)
    with pytest.raises(SystemExit) as caught:
        load_runner().main(["embed", "--unknown-option"])
    assert caught.value.code == 2


def test_workflow_imports_without_legacy_root_scripts():
    """包内 workflow 可直接导入，不再依赖已移除的根目录脚本。"""
    workflow = importlib.import_module("azusa_rag.du_retrieval_workflow")
    assert workflow.run_search_result.__module__ == "azusa_rag.cli.search"
    assert workflow.run_eval_for_mode.__module__ == "azusa_rag.cli.eval"


def test_source_identity_includes_moved_cli_files():
    """源码指纹应覆盖迁入子包的执行代码，且不读取已删除的 bootstrap。"""
    paths = {Path(item["path"]) for item in source_identity()["files"]}
    assert Path(embed.__file__).resolve() in paths
    assert Path(importlib.import_module("azusa_rag.cli.search").__file__).resolve() in paths
    assert not any(path.name == "bootstrap.py" for path in paths)
