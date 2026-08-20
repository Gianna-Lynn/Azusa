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
| 单格失败不吞成功格 | `run_matrix` | `test_run_matrix_keeps_successes_and_failures` |
| 三种报告不隐藏失败 | `write_outputs` | `test_write_outputs_delivers_all_formats_and_failures` |

答案分支必须全部通过；课程分支在未实现时必须逐项精确落到对应 TODO，而不是被导入错误、
缺失 fixture 或后续任务提前遮挡。
