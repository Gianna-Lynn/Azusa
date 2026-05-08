"""
本文件不是包azusa_rag内部业务模块, 而是一个"启动器", 把src加到Python搜索路径当中,
然后调用真正的main.
"""
import sys
from pathlib import Path

# 找到 rag/ 根目录
PROJECT_ROOT = Path(__file__).resolve().parent # 这样取得的是当前这个文件所在的目录的完整路径 .../rag

# 把 rag/src 加入 sys.path, 而sys.path是 Python import的时候会去搜索的目录列表.
# 如果已经加入,Python就会在src/下面看到 azusa_rag/和内含的文件. 这样azusa_rag就被识别成了一个包, 内含的每个文件就是包里的一个模块.
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# 导入azusa_rag.main里面的main()并调用

from azusa_rag.main import main
if __name__ == "__main__":
    main()



