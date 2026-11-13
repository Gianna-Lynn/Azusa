"""Typed metadata filtering and a minimal deny-by-default ACL policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schemas import EmbeddingRecord


@dataclass(frozen=True)
class MetadataFilter:
    tenant_id: str | None = None
    department: str | None = None
    document_type: str | None = None
    language: str | None = None
    visibility: str | None = None
    tags_any: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MetadataFilter | None":
        # TODO AZUSA-V2-L1c：实现 fail-closed filter 输入协议。
        # None 保持 None；拒绝未知字段；五个标量只能是 str 或 None；tags_any 必须
        # 是字符串 list，内部冻结为 frozenset。不得用 truthy/falsy 掩盖错误类型。
        raise NotImplementedError("TODO AZUSA-V2-L1c：实现 MetadataFilter 输入校验")


@dataclass(frozen=True)
class AccessContext:
    tenant_id: str | None
    departments: frozenset[str] = field(default_factory=frozenset)
    allow_private: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "AccessContext | None":
        # TODO AZUSA-V2-L1d：实现严格 ACL 上下文协议。
        # 只接受 tenant_id/departments/allow_private；tenant_id 为 str|None，
        # departments 为字符串 list，allow_private 必须是真正 bool。
        raise NotImplementedError("TODO AZUSA-V2-L1d：实现 AccessContext 输入校验")


def metadata_matches(record: EmbeddingRecord, metadata_filter: MetadataFilter | None) -> bool:
    # TODO AZUSA-V2-L2d：实现 metadata predicate。
    # 所有非 None 标量必须相等；tags_any 采用“至少一个交集”；无 filter 全部匹配。
    raise NotImplementedError("TODO AZUSA-V2-L2d：实现 metadata 匹配")


def acl_allows(record: EmbeddingRecord, access: AccessContext | None) -> bool:
    """Apply tenant isolation and public/department/private visibility rules."""

    # TODO AZUSA-V2-L2e：实现最小 deny-by-default ACL。
    # access=None 表示调用方未启用 ACL；记录存在 tenant 时必须严格相等；visibility
    # 缺失按 public；department 需属于授权集合；private 只看 allow_private；未知值拒绝。
    raise NotImplementedError("TODO AZUSA-V2-L2e：实现租户与可见性 ACL")


def record_is_visible(
    record: EmbeddingRecord,
    metadata_filter: MetadataFilter | None,
    access: AccessContext | None,
) -> bool:
    return metadata_matches(record, metadata_filter) and acl_allows(record, access)
