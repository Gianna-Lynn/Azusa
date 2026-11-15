"""为检索评测记录源码、数据与运行配置身份，并创建不覆盖历史结果的输出目录。"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import subprocess
import sys
from uuid import uuid4

from . import config


def file_identity(path: Path) -> dict:
    """返回解析后的路径、字节长度和 SHA-256，供加载前后比较输入身份。

    只读文件，不创建目录。文件缺失、不可读等异常直接传播，不能把另一份路径
    上的旧产物猜测为本次输入，也不能用空哈希冒充已核验。
    """
    path = path.resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def runtime_config() -> dict:
    """返回配置白名单和已加载模型的实际设备，不触发模型加载。

    设备为 None 表示该组件尚未加载，不能解释为 CPU 或 CUDA 实测结果；
    配置路径也不等价于模型权重内容哈希。不会遍历或保存任意环境变量。
    """
    names = (
        "EMBEDDING_MODEL_NAME", "EMBEDDING_DIM", "EMBEDDING_DEVICE",
        "EMBEDDING_BATCH_SIZE", "EMBEDDING_MAX_LENGTH", "RERANK_ENABLED",
        "RERANK_MODEL_NAME", "RERANK_DEVICE", "RERANK_BATCH_SIZE", "RERANK_MAX_LENGTH",
        "CHUNK_SIZE", "CHUNK_OVERLAP", "URL",
    )
    result = {name: getattr(config, name) for name in names}
    for component in ("embedder", "reranker"):
        module = sys.modules.get(f"azusa_rag.{component}")
        result[f"{component}_actual_device"] = getattr(module, "_MODEL_DEVICE", None)
    return result


def source_identity() -> dict:
    """记录提交、dirty 状态及参与评测的 Python 源码精确字节哈希。

    Git 查询只读且有超时；Git 不可用时保留源码身份，将提交与 dirty 置为未知。
    只保存文件哈希，不将工作区 diff 正文或 Git 配置写入报告。
    """
    rag_root = config.PROJECT_ROOT
    # 递归包含 cli 等子包，迁移入口后仍记录全部 Python 执行源码。
    files = sorted((rag_root / "src" / "azusa_rag").rglob("*.py"))
    files += [rag_root / "run.py"]
    source_files = [file_identity(path) for path in files]
    result = {"git_commit": None, "git_dirty": None, "files": source_files}
    try:
        def git(*arguments: str) -> str:
            """执行有超时的只读 Git 查询，不刷新索引或访问网络。"""
            return subprocess.check_output(
                ["git", "--no-optional-locks", "-C", str(rag_root.parent), *arguments],
                stderr=subprocess.DEVNULL, text=True, timeout=5,
            ).strip()
        result["git_commit"] = git("rev-parse", "HEAD")
        result["git_dirty"] = bool(git("status", "--porcelain", "--untracked-files=normal"))
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def experiment_identity() -> dict:
    """生成本次实验身份；包版本和平台描述用于约束跨环境比较，不代表性能保证。"""
    packages = {}
    for name in ("rank-bm25", "torch", "transformers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "run_id": uuid4().hex,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source_identity(),
        "runtime_config": runtime_config(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": packages},
        "latency_policy": "每格一次显式预热不计入逐查询延迟；计入编码、客户端身份检查、HTTP、检索和重排。BM25 包含每次建索引；跨机器不可直接比较。",
        "metric_policy": "binary nDCG@k 的 IDCG 使用请求的 k；短结果不能缩小分母；重复命中拒绝。",
        "model_identity_policy": "记录配置路径和已加载设备；未校验权重字节，不保证不同本地模型版本完全复现。",
    }


def new_run_directory(parent: Path, label: str = "eval") -> Path:
    """返回带 UTC 时间、规范化标签和随机后缀的候选目录路径。

    本函数不执行 mkdir，也不保证跨进程排他；调用方仍须独占创建结果文件，
    才能确保并发冲突或重复执行不会覆盖历史产物。
    """
    safe_label = "".join(char if char.isalnum() or char in "-_" else "_" for char in label)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return parent / f"{safe_label or 'eval'}-{stamp}-{uuid4().hex[:8]}"
