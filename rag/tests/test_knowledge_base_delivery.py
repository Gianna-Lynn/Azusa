"""验证真实切块到产物的增量闭环及拒绝损坏输入的写入边界。"""

import json
from dataclasses import replace

import pytest

from azusa_rag.knowledge_base import content_digest, load_document_records, load_manifest, sync_knowledge_base
from azusa_rag.schemas import DocumentRecord


def document(text, **kwargs):
    """构造自有合成文档；不使用模型、外部数据或旧实验文件。"""
    digest = content_digest(text)
    return DocumentRecord("manual", digest[:16], "manual.md", "手册", text, digest, "text/markdown", **kwargs)


def synchronize(tmp_path, docs, embed=None, identity="test-v1"):
    """把一代产物写入 pytest 独立目录，使用确定性二维向量替身。"""
    return sync_knowledge_base(
        docs, chunk_size=61, overlap_size=0,
        manifest_path=tmp_path/"manifest.json", chunks_path=tmp_path/"chunks.jsonl",
        embeddings_path=tmp_path/"embeddings.jsonl",
        embed_texts=embed or (lambda texts: [[float(len(text)), 1.0] for text in texts]),
        embedding_identity=identity,
    )


def snapshot(tmp_path):
    """读取三份当前产物原始字节，以断言失败没有覆盖上一代。"""
    return {name: (tmp_path/name).read_bytes() for name in ("manifest.json", "chunks.jsonl", "embeddings.jsonl")}


def test_real_incremental_metadata_delete_and_empty(tmp_path):
    """真实切块下复用未变向量，同时刷新版本、span 和权限；删除清空所有旧行。"""
    first_doc = document("A"*60 + "。" + "B"*60, metadata={"visibility": "public"})
    first = synchronize(tmp_path, [first_doc])
    assert first["embedded_chunk_count"] == 2
    changed = document("X"*60 + "。" + "B"*60, metadata={"visibility": "private"})
    second = synchronize(tmp_path, [changed])
    assert second["embedded_chunk_count"] == 1
    assert second["reused_embedding_count"] == 1
    stored = [json.loads(line) for line in (tmp_path/"embeddings.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(item["metadata"]["visibility"] == "private" for item in stored)
    assert all(item["document_version"] == changed.version for item in stored)
    assert stored[1]["start_char"] == 61
    def unexpected_embed(texts):
        """全部未变时不应触发编码，即使空列表也不发送。"""
        raise AssertionError(texts)
    third = synchronize(tmp_path, [changed], unexpected_embed)
    assert third["reused_embedding_count"] == 2
    empty = synchronize(tmp_path, [], unexpected_embed)
    assert empty["changes"]["removed_document_ids"] == ["manual"]
    assert (tmp_path/"chunks.jsonl").read_bytes() == b""
    assert (tmp_path/"embeddings.jsonl").read_bytes() == b""
    assert load_manifest(tmp_path/"manifest.json")["schema_version"] == 2


def test_identity_change_reembeds_and_new_document_is_added(tmp_path):
    """模型身份改变使旧向量全部失效；新文档不冒充旧内容复用。"""
    doc = document("same body")
    synchronize(tmp_path, [doc])
    added = replace(doc, document_id="new", source="new.md")
    calls = []
    def embed(texts):
        """记录真实待编码文本数并模拟新的三维模型。"""
        calls.append(texts)
        return [[1.0, 2.0, 3.0] for _ in texts]
    result = synchronize(tmp_path, [doc, added], embed, "test-v2")
    assert len(calls[0]) == 2
    assert result["reused_embedding_count"] == 0
    assert result["changes"]["added_document_ids"] == ["new"]


@pytest.mark.parametrize("vectors,message", [
    ([], "number of vectors"), ([[float("nan")], [1]], "nonempty and finite"),
    ([[True], [1]], "nonempty and finite"), ([[], [1]], "nonempty and finite"),
    ([[1], [2, 3]], "dimensions"),
])
def test_invalid_embedding_leaves_old_generation_untouched(tmp_path, vectors, message):
    """计数、有限值、数值类型、空向量和维度不一致均在替换文件前拒绝。"""
    synchronize(tmp_path, [document("old")])
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match=message):
        synchronize(tmp_path, [document("N"*60 + "。" + "M"*60)], lambda texts: vectors)
    assert snapshot(tmp_path) == before


def test_artifact_tampering_is_not_silently_reused(tmp_path):
    """外部修改或混代产物须恢复整代，不能作为有效旧向量复用。"""
    doc = document("body")
    synchronize(tmp_path, [doc])
    (tmp_path/"embeddings.jsonl").write_text("{}\n", encoding="utf-8")
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="generation mismatch"):
        synchronize(tmp_path, [doc])
    assert snapshot(tmp_path) == before


def test_duplicate_documents_fail_before_encoding(tmp_path):
    """路径规范化可能碰撞；重复 ID 必须显式失败而不是覆盖其中一份文档。"""
    doc = document("body")
    with pytest.raises(ValueError, match="Duplicate"):
        synchronize(tmp_path, [doc, doc])
    assert not (tmp_path/"manifest.json").exists()


def test_empty_source_requires_explicit_choice(tmp_path):
    """误拼目录始终失败，真实空目录只有显式允许才可用于删除。"""
    with pytest.raises(ValueError):
        load_document_records(tmp_path/"missing", allow_empty=True)
    with pytest.raises(ValueError):
        load_document_records(tmp_path)
    assert load_document_records(tmp_path, allow_empty=True) == []


def test_reused_and_new_vector_dimensions_must_match(tmp_path):
    """未改模型身份却产生新维度时，不能把它与未变块的旧向量混写。"""
    synchronize(tmp_path, [document("A"*60 + "。" + "B"*60)])
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match="dimensions"):
        synchronize(tmp_path, [document("X"*60 + "。" + "B"*60)], lambda texts: [[1, 2, 3] for _ in texts])
    assert snapshot(tmp_path) == before


def test_interrupted_file_commit_is_detected_next_time(tmp_path, monkeypatch):
    """第二文件替换失败留下混代时，下次必须拒绝，而非声称三文件事务已回滚。"""
    from pathlib import Path

    synchronize(tmp_path, [document("old")])
    original = Path.replace
    def interrupted(path, target):
        """仅在本次向量产物替换时注入 IO 失败。"""
        if target.name == "embeddings.jsonl":
            raise OSError("simulated interruption")
        return original(path, target)
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "replace", interrupted)
        with pytest.raises(OSError, match="simulated"):
            synchronize(tmp_path, [document("new")])
    with pytest.raises(ValueError, match="generation mismatch"):
        synchronize(tmp_path, [document("new")])


def test_v1_migration_does_not_trust_unbound_embeddings(tmp_path):
    """旧清单可以加载，但未记录编码身份的向量必须重新生成。"""
    doc = document("body")
    synchronize(tmp_path, [doc])
    manifest = json.loads((tmp_path/"manifest.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    manifest.pop("embedding_identity")
    manifest.pop("artifacts_sha256")
    (tmp_path/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = synchronize(tmp_path, [doc])
    assert result["embedded_chunk_count"] == 1
    assert result["reused_embedding_count"] == 0
