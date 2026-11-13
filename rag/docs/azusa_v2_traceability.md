# Azusa v2 课程要求追踪

| 要求 | 实现边界 | 直接测试 |
| --- | --- | --- |
| 哈希跨换行风格稳定 | `content_digest` | `test_content_digest_normalizes_newlines_and_whitespace` |
| 同名不同路径不碰撞 | `stable_document_id` | `test_stable_document_id_is_readable_and_path_sensitive` |
| 未知 filter 字段拒绝 | `MetadataFilter.from_dict` | `test_metadata_filter_parser_rejects_unknown_and_bad_tags` |
| ACL bool 严格类型 | `AccessContext.from_dict` | `test_access_context_parser_rejects_truthy_non_bool` |
| 无关编辑不漂移 chunk ID | `build_stable_chunks` | `test_chunk_identity_survives_unrelated_prefix_change` |
| reused 同时校验 ID/hash | `plan_changes` | `test_plan_changes_requires_matching_id_and_hash` |
| 只编码变化 chunk | `sync_knowledge_base` | `test_sync_reuses_unchanged_embeddings_and_removes_old_chunks` |
| 删除后 artifact 无旧证据 | `sync_knowledge_base` | `test_sync_empty_collection_deletes_all_prior_chunks` |
| filter 与 ACL 独立 | `metadata_matches` / `acl_allows` | `test_metadata_matches_requires_all_scalars_and_one_tag` / `test_acl_allows_is_tenant_isolated_and_fail_closed` |
| 排名靠前收益更高 | `normalized_discounted_cumulative_gain` | `test_ndcg_rewards_earlier_relevant_results` |
| 短排名不能缩小请求 k 的分母 | `evaluate_case` / nDCG 显式 k | `test_short_ranking_keeps_requested_k_denominator` |
| 重复证据不得放大指标 | nDCG 输入校验 | `test_duplicate_evidence_is_rejected_instead_of_inflating_metrics` |
| 单格失败不吞成功格 | `run_matrix` | `test_run_matrix_keeps_successes_and_failures` |
| 三种报告不隐藏失败 | `write_outputs` | `test_write_outputs_delivers_all_formats_and_failures` |
| 输入变动不能绑定旧成绩 | `run_matrix` 输入哈希复核 | `test_matrix_rejects_changed_input_instead_of_publishing_success` |
| 实际回退不得冒充模型重排 | `rerank_candidates` / 每题 trace | `test_fallback_is_visible_in_per_query_and_aggregate_trace` |

P0-B 已在当前主线实现 L1e、L3a、L3b，正式评测按请求 k 计算 IDCG；参考分支中的
旧实现仍是被审计对象，不能用其“按返回长度计算分母”的行为替代当前契约。
P1 已进一步实现当前主线 L1a、L1b、L2a、L2b、L2c；新增交付测试覆盖编码身份、
权限元数据刷新、损坏产物拒绝和空库删除。manifest v2 与单写者写入边界见
`docs/knowledge_base_v1.md`。Filter/ACL 四项练习留在 develop，完整实现在 solution；main 不导出这些接口。表内 Filter/ACL 的实现与测试位置仅适用于 develop / solution。

答案分支必须全部通过；课程分支在未实现时必须逐项精确落到对应 TODO，而不是被导入错误、
缺失 fixture 或后续任务提前遮挡。
