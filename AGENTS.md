# 协作说明

Azusa 包含 Rust 向量索引与 HTTP 服务，以及 Python RAG、知识库和评测流程。

## 修改边界

- 修改前检查 `git status`、相关模块与测试，保留用户已有改动。
- 保持改动聚焦；不要将模型、数据、缓存或实验输出提交到仓库。
- `main` 保留已完成能力；`develop` 保存开发任务和学习资料；`solution` 保存参考实现。
- 不把答案整体合入 main，不把已完成模块重新改成练习占位。
- main 的注释说明接口、算法与边界；完整学习批注保留在 develop。
- 不改用户或系统环境变量、网络与全局配置；不使用 sudo。
- `src/legacy/` 和 `tests/archive/` 只有任务明确涉及才修改。

## 模块

- `src/index/`：HNSW 建图、搜索、访问器和工作区。
- `src/bin/azusa_server.rs`：不可变快照装载、检索和记录查询。
- `rag/src/azusa_rag/`：知识库、检索、重排、回答、Agent 与网关。
- `rag/run.py`：Python 统一命令入口。
- `scripts/`：HTTP smoke、实验和归档校验。

## 验证

从根目录运行：

```powershell
cargo test --locked
cargo run --locked --example retrieval_demo
```

从 rag 目录、使用现有项目 Python 环境运行：

```powershell
python -m pytest tests -q -p no:cacheprovider -o pythonpath=src
python run.py --help
```

先运行与修改相关的测试。真实 HTTP 测试需要已构建的服务可执行文件；不能把缺少程序
导致的 skipped 当作验收。纯契约测试不应下载模型或改动默认模型配置。

## 数据契约

- 修改稳定 ID、哈希、manifest 或 embedding 复用前，检查兼容性和删除语义。
- Rust 服务重启后索引丢失；不同语料快照不能静默复用或追加。
- RRF 按排名融合，不直接比较 dense/sparse 原始分数。
- HTTP 状态码、网关错误码和 retryable 是外部契约，改动须同步文档与测试。
- 运行实验使用独立目录；未经要求不覆盖 latest 或已有报告。
- 完成后检查 `git diff`，确认不混入业务逻辑之外的无关改动或生成物。

项目入口见 README；使用、架构、接口与实验见 `docs/README.md`。
