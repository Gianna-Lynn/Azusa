from pathlib import Path

# 自动定位rag这个小项目的根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
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

CHUNK_OUTPUT_PATH = OUTPUT_DIR / "chunks.jsonl"

EMBEDDING_OUTPUT_PATH = OUTPUT_DIR / "embedding.jsonl"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 30