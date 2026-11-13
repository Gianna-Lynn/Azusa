# DuRetrieval 全量向量检索：2026-11-11 服务器部署

最终状态：**成功**，2026-11-11 14:55:52（北京时间）完成，全部 8,000 条查询观测已取回
本地并复算校验。完整流程约 34.5 分钟，建图约 20.51 分钟。详见
[结果与归档](text-dense-results-20261111.md)。下文保留部署时的配置、观察和估时记录。

## 实验目标

对与 BM25 基线相同的 100,001 条中文段落、2,000 条查询，比较精确向量检索和
Azusa 自己的 Rust HNSW。分别报告与人工相关性标注的匹配质量，以及相对精确
向量排名的近邻召回；两者不能混为一个 Recall。

本轮执行先前方案的步骤 1–3，不包含 RRF、reranker 或回答模型。

## 固定配置

- 模型：`BAAI/bge-small-zh-v1.5`，commit `7999e1d3359715c523056ef9478215996d62a620`。
- 这是实验固定模型，区别于项目默认的 Qwen3-Embedding；未修改默认配置。
- 按[官方模型卡](https://huggingface.co/BAAI/bge-small-zh-v1.5)使用 CLS pooling，
  查询前缀“为这个句子生成表示以用于检索相关文章：”，段落不加前缀。
- GPU fp16 推理，L2 归一化后按 float32 保存；最大长度 512 token，batch 32。
  原文超长样本数量单独记录，不把截断隐藏起来。
- 使用完整 DuRetrieval 数据，逐字节校验与前一轮报告中的输入哈希一致。
- 精确搜索与图检索调用相同的 Azusa Cosine 度量代码。
- HNSW：m=16、m_max=32、ef_construction=200，层级概率 0.5，源码 RNG seed=42，
  按数据行顺序插入一次；同一张图查询 ef_search=100/200/400。
- 每个配置返回 top-100，计算 @5/@10/@100 的质量与近邻指标。
- 每配置前 10 条查询分别先做一次未计时的相同请求预热，再记录该题；其余顺序执行。
  单次图、单次运行，不宣称统计显著性或生产 SLA。

## 服务器目录和进程

独立目录：`/mnt/sevenT/zeliny/Experiments/azusa-dense-20261111-a/`。
没有覆盖旧服务器仓库，也未修改 KandaEnv 或全局环境变量，无 sudo。

- `source/`：当前本地 Rust 源码与 Cargo.lock 的独立快照，包含已有学习批注。
- `source-snapshot.json`：每个源文件 SHA-256、本地提交基底及已有修改状态。
- `model-prefetch/`：固定模型文件、原始元信息、文件哈希。
- `run_text_dense_scale.py`：实际执行的编排脚本。
- `run-001.log`：nohup 主日志；主进程初始 PID 为 17689。
- `run-001/`：f32 向量、ID 映射、编码计时、各配置逐题结果及汇总。
- `worker-smoke/verification.json`：实际 release HTTP 服务验证记录。

编排进程使用上一轮实验的隔离 venv（基于 KandaEnv）。进程级设置 GPU 0、
BLAS/OMP 两线程、nice 10，GPU 内存分配限制为该卡的 25%。编码后释放模型，
再启动原生检索进程。后台主进程 PPID=1、无 TTY，stdin=/dev/null，日志重定向，
SIGHUP 已忽略。服务只绑定 loopback 的随机端口，不暴露外网。

Rust HTTP 服务是本次实验适配器，直接调用当前快照的 HnswIndex；不是改写或
替代生产 azusa_server。报告保留实际二进制、编排脚本与源码清单哈希。

## 产物与计时口径

- `base.f32`、`queries.f32`：同一批冻结的归一化向量。
- `base-encoding.json`、`queries-encoding.json`：编码总时间、截断数与向量哈希。
- `configuration.json`：模型、图、输入、依赖、设备、源码与二进制身份。
- `progress.json`：当前阶段、已完成数量与吞吐；只表示进度，不代表成功。
- `worker.log`：每千条建图进度、最终端口与构建耗时。
- `exact-ef100.jsonl`：精确排名、每题质量与计时。
- `hnsw-ef100/200/400.jsonl`：图排名、证据质量、相对精确 top-k 的近邻召回。
- 每配置 `*-summary.json`：配置完成即保存，便于保留部分成功结果。
- `report.json`：只有全部配置、全部查询成功后才生成最终报告。
- `failure.json`：捕获异常时保存错误，不删除已保存的结果。

Rust 原生搜索计时不包括 HTTP；客户端计时包括 HTTP、序列化与响应处理。
二者均使用预编码查询向量，不含查询编码、KB 指纹检查、回答生成或生产网关开销。
精确 baseline 同样经过该 HTTP 适配器，避免仅把它的原生计时与另一模式的 HTTP
计时混比。编码、建图、检索分别报告，不把批量编码均值冒充单请求 P95。

## 验证和观察

本地 Python 语法及短排名分母、nearest-rank、重复排名检查通过；本地 Rust
测试通过。服务器 release Rust 排名测试通过；HTTP smoke 覆盖健康信息、精确
排名、HNSW 排名、原生计时与非法维度拒绝。

14:24:16（北京时间）建立五分钟观察基线：编码完成 26,624 条，约 179 条/秒，
无错误，GPU 0 总使用约 3.85 GiB。按要求不早于 14:29:16 再检查。
五分钟复查在 14:29:46（北京时间）执行，间隔 5 分 30 秒：进度增至 91,648 条，
约 191 条/秒，预计剩余段落编码约 44 秒；无 failure.json，主进程仍为 PPID=1、
无 TTY、忽略 SIGHUP，GPU 0 总显存约 3.85 GiB，观察通过。

随后确认全部段落和查询编码完成，自动进入全量建图；查询编码约 1.64 秒，
建图前 3,000 条约 16.05 秒，原生子进程 PID 为 12519。该小段速度不能线性
外推整张图，因为图增长后插入成本会上升。

截止时间为当日 21:50（北京时间）。基于已完成编码和早期建图进度，暂按剩余
30–90 分钟预算，预计 15:00–16:00 左右完成，保守预留至 17:00；有望在截止前
取回结果，但共享负载、进程错误等仍可能影响完成时间，不能作保证。

查看状态：

```powershell
ssh E5_server 'cat /mnt/sevenT/zeliny/Experiments/azusa-dense-20261111-a/run-001/progress.json; tail -5 /mnt/sevenT/zeliny/Experiments/azusa-dense-20261111-a/run-001.log'
```

即使 SSH 断开任务也会继续；服务器重启、OOM 或应用错误仍可能使任务失败。
未设置自动重启，避免失败后无意重复昂贵实验或覆盖原始产物。
