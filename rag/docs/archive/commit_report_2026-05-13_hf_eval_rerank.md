# Commit Report: HF Eval Import + Hybrid Rerank + Chunk Coverage Fix

## 背景

这次提交围绕 `rag/` 子项目做了三组互相关联的增强:

1. 接入一个可快速扩展测试范围的 Hugging Face 小型评测数据入口。
2. 为 `hybrid_rerank` 模式接入模型 reranker, 并保留规则 fallback。
3. 修复 chunker 在自然边界截断后可能跳过文本区间的问题。

这三部分合在一起, 让当前最小 RAG 链路不再只依赖手写的少量样本, 也让 hybrid rerank 模式在真实问答样本上更容易观察效果。

## 主要改动

### 1. 新增 Hugging Face 样本导入器

新增文件:

- `rag/src/azusa_rag/hf_dataset_import.py`
- `rag/run_import_hf_eval.py`

实现内容:

- 通过 Hugging Face `datasets-server` 的 rows API 拉取 SQuAD 风格小样本。
- 默认数据集为 `autoevaluate/squad-sample`。
- 自动生成:
  - `data/raw/*.txt`
  - `data/eval/hf_squad_sample_eval.jsonl`
  - 少量 `data/queries/*.txt`
- 导入时会基于当前 chunk 规则, 把答案字符区间映射成 `relevant_chunk_ids`, 从而直接兼容现有 `run_eval.py`。

设计原因:

- 不依赖本地 `datasets` 包, 减少环境要求。
- 保持现有 `raw txt -> chunk -> embedding -> search/eval` 主链路不变。
- 先从小样本做起, 便于快速 smoke test 和 bad case 观察。

### 2. 接入最小 reranker 能力

新增文件:

- `rag/src/azusa_rag/reranker.py`

改动文件:

- `rag/src/azusa_rag/search.py`
- `rag/src/azusa_rag/config.py`

实现内容:

- 新增延迟加载的 reranker 模块:
  - 按配置加载 `AutoTokenizer`
  - 按配置加载 `AutoModelForSequenceClassification`
  - 支持 `cuda -> cpu` 回退
  - 按 batch 对 `(query, passage)` 对打分
- 在 `search.py` 的 `ts030_rerank_candidates(...)` 中:
  - 优先使用模型 reranker 分数重排
  - 若模型不可用 / 推理失败 / 返回数量不匹配, 自动 fallback 到规则打分
- `hybrid_rerank` 的候选集合从原先较小范围扩大到:
  - `candidate_k = min(len(embedding_records), max(20, top_k * 5))`

设计原因:

- 让 `hybrid_rerank` 不再只是纯规则重排。
- 同时保留 fallback, 保证环境不完整时功能仍可运行。
- 扩大候选集合后, reranker 才有机会把原本 dense/sparse 排位靠后的正确项提上来。

### 3. 修复 chunk 覆盖 bug, 并暴露 span 信息

改动文件:

- `rag/src/azusa_rag/chunker.py`

实现内容:

- 拆出 `_validate_chunk_params(...)`
- 拆出 `_strip_span(...)`
- 新增 `chunk_text_with_spans(...)`
- 修正 chunk 推进逻辑:
  - 旧逻辑: 命中自然边界后仍按固定 `step` 推进
  - 新逻辑: 从 `end - overlap_size` 继续推进
- `chunk_document(...)` 改为复用带 span 的切块结果

关键问题:

- 旧逻辑在自然边界收缩 chunk 终点后, 可能导致下一块从更靠后的位置开始, 从而跳过中间一小段文本。
- 这会直接影响:
  - 检索覆盖率
  - eval gold 映射
  - reranker 候选质量

修复效果:

- chunk 之间不再因为边界收缩产生“漏切区间”。
- HF 样本中的答案字符区间可以稳定映射到对应 chunk。

### 4. 文档与忽略规则更新

改动文件:

- `rag/README.md`
- `rag/.gitignore`

实现内容:

- README 补充 HF 导入脚本的使用方式与输出说明。
- `rag/.gitignore` 增加 `*.egg-info/`, 避免把打包生成物里的 `*.txt` 错误纳入版本控制。

## 注释补强

这次对新加代码补了统一风格的说明:

- 为新函数补充了 `"""功能注释"""`。
- 为关键逻辑补了少量行内注释, 重点解释:
  - 为什么保留 span
  - 为什么扩候选再 rerank
  - 为什么要做 reranker fallback
  - 为什么 hybrid 也要先确保 HNSW 索引已加载

涉及文件:

- `rag/src/azusa_rag/chunker.py`
- `rag/src/azusa_rag/hf_dataset_import.py`
- `rag/src/azusa_rag/reranker.py`
- `rag/src/azusa_rag/search.py`
- `rag/run_import_hf_eval.py`

## 验证记录

本地完成过的验证:

1. CLI / 环境检查
   - 确认当前环境可用 `hf`
   - 确认 `ChuoEnv` 中可执行 `hf --version`

2. 语法检查
   - `python -m py_compile` 校验通过:
     - `chunker.py`
     - `hf_dataset_import.py`
     - `run_import_hf_eval.py`
     - `reranker.py`
     - `search.py`

3. Hugging Face 导入验证
   - 实际运行 `python run_import_hf_eval.py --length 2`
   - 成功导入 2 条 `autoevaluate/squad-sample` 样本
   - 成功生成 `hf_squad_sample_eval.jsonl`

4. bug 复现与修复验证
   - 对失败样本检查了 answer span 和 chunk span
   - 确认修复前存在覆盖缺口
   - 修复后确认答案 `1962` 可以命中正确 chunk

## 当前注意事项

1. `rag/src/azusa_rag/config.py` 当前使用的是本地绝对模型路径:
   - `D:\Chuo_Data\Models\Qwen3-Embedding-0.6B`
   - `D:\Chuo_Data\Models\bge-reranker-v2-m3`

这适合当前机器直接跑, 但换机器时需要手动调整。

2. 由于 chunk 逻辑已变化, 旧的:
   - `output/artifacts/chunks.jsonl`
   - `output/artifacts/embedding.jsonl`

  最好重新生成, 否则可能与新的 eval gold 或 rerank 预期不一致。

3. HF 导入脚本当前面向 SQuAD 风格数据结构:
   - `question`
   - `context`
   - `answers.answer_start`
   - `answers.text`

如果未来换别的数据集, 可能需要补一层格式适配。

## 建议后续工作

1. 增加一个“一键导入 -> 重建 chunk -> 重建 embedding -> 评测”的整链脚本。
2. 把本地绝对模型路径逐步收敛到环境变量或配置文件覆盖机制。
3. 在 `run_eval.py` 中增加不同 mode 的结果对比输出, 方便直接比较 rerank 是否带来收益。
