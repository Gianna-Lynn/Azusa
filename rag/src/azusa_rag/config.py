"""
定义全局的配置项, 方便在项目的其他地方引用
"""
from pathlib import Path

# 自动定位rag这个小项目的根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # F:\Repositories\Azusa\rag
# 从当前这个config.py文件的位置, 一路往上找, 找到 rag/这个项目根目录
# __file__表示 "当前这个Python文件自己的路径"
# 也就是F:\Repositories\Azusa\rag\src\azusa_rag\config.py
# Path(__file__)把这个路径变成pathlib.Path对象, 后面可以方便的做"往上一级""拼路径"这样的操作
# .resolve()可以把路径变成完整的绝对路径
# .parents[2]表示"往上找第3层父目录"
# 这里拿到的就是F:\Repositories\Azusa\rag



RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = PROJECT_ROOT / "output"

ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"
SEARCH_RUNS_DIR = OUTPUT_DIR / "search_runs"
EVAL_RUNS_DIR = OUTPUT_DIR / "eval_runs"
LATEST_SEARCH_RUN_DIR = SEARCH_RUNS_DIR / "latest"
SEARCH_RUN_ARCHIVE_DIR = SEARCH_RUNS_DIR / "archive"
LATEST_EVAL_RUN_DIR = EVAL_RUNS_DIR / "latest"
EVAL_RUN_ARCHIVE_DIR = EVAL_RUNS_DIR / "archive"

DU_RETRIEVAL_WORKFLOW_DIR = PROCESSED_DATA_DIR / "du_retrieval_workflow"
DU_RETRIEVAL_LATEST_DIR = DU_RETRIEVAL_WORKFLOW_DIR / "latest"
DU_RETRIEVAL_RUNS_DIR = DU_RETRIEVAL_WORKFLOW_DIR / "runs"

HF_IMPORT_ROOT_DIR = PROJECT_ROOT / "data" / "import_snapshots" / "hf_eval"
HF_IMPORT_LATEST_DIR = HF_IMPORT_ROOT_DIR / "latest"
HF_IMPORT_RUNS_DIR = HF_IMPORT_ROOT_DIR / "runs"

CHUNK_OUTPUT_PATH = ARTIFACTS_DIR / "chunks.jsonl"

EMBEDDING_OUTPUT_PATH = ARTIFACTS_DIR / "embedding.jsonl"

EMBEDDING_MODEL_NAME = r"D:\Chuo_Data\Models\Qwen3-Embedding-0.6B" # 这个是专门用来编码的模型, 接下来的配置要和embedder.py里调用的地方保持一致.
EMBEDDING_DIM = 512       # 模型的向量维度.
EMBEDDING_DEVICE = "cuda" # 模型所在的设备.
EMBEDDING_BATCH_SIZE = 1  # 模型的批处理大小.
EMBEDDING_MAX_LENGTH = 512 # 编码时允许使用的最大token长度.

RERANK_ENABLED = True
RERANK_MODEL_NAME = r"D:\Chuo_Data\Models\bge-reranker-v2-m3"
RERANK_DEVICE = "cuda"
RERANK_BATCH_SIZE = 4
RERANK_MAX_LENGTH = 512


CHUNK_SIZE = 300
CHUNK_OVERLAP = 30

QUERY_DIR = PROJECT_ROOT / "data" / "queries"

URL = "http://127.0.0.1:3000"
HEALTH_URL = f"{URL}/health"
SEARCH_URL = f"{URL}/search"
INSERT_URL = f"{URL}/insert"
