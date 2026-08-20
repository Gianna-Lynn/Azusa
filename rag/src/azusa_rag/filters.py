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
        if payload is None:
            return None
        allowed = {"tenant_id", "department", "document_type", "language", "visibility", "tags_any"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown metadata filter fields: {unknown}")
        tags = payload.get("tags_any") or []
        if not isinstance(tags, list) or not all(isinstance(item, str) for item in tags):
            raise ValueError("tags_any must be a list of strings")
        scalar = {key: payload.get(key) for key in allowed - {"tags_any"}}
        if any(value is not None and not isinstance(value, str) for value in scalar.values()):
            raise ValueError("Metadata filter scalar values must be strings")
        return cls(**scalar, tags_any=frozenset(tags))


@dataclass(frozen=True)
class AccessContext:
    tenant_id: str | None
    departments: frozenset[str] = field(default_factory=frozenset)
    allow_private: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "AccessContext | None":
        if payload is None:
            return None
        unknown = sorted(set(payload) - {"tenant_id", "departments", "allow_private"})
        if unknown:
            raise ValueError(f"Unknown access context fields: {unknown}")
        departments = payload.get("departments") or []
        allow_private = payload.get("allow_private", False)
        if not isinstance(departments, list) or not all(isinstance(item, str) for item in departments):
            raise ValueError("departments must be a list of strings")
        if not isinstance(allow_private, bool):
            raise ValueError("allow_private must be a boolean")
        tenant_id = payload.get("tenant_id")
        if tenant_id is not None and not isinstance(tenant_id, str):
            raise ValueError("tenant_id must be a string or null")
        return cls(tenant_id, frozenset(departments), allow_private)


def metadata_matches(record: EmbeddingRecord, metadata_filter: MetadataFilter | None) -> bool:
    if metadata_filter is None:
        return True
    metadata = record.metadata
    for key in ("tenant_id", "department", "document_type", "language", "visibility"):
        expected = getattr(metadata_filter, key)
        if expected is not None and metadata.get(key) != expected:
            return False
    if metadata_filter.tags_any:
        record_tags = metadata.get("tags") or []
        if not metadata_filter.tags_any.intersection(record_tags):
            return False
    return True


def acl_allows(record: EmbeddingRecord, access: AccessContext | None) -> bool:
    """Apply tenant isolation and public/department/private visibility rules."""

    if access is None:
        return True
    metadata = record.metadata
    record_tenant = metadata.get("tenant_id")
    if record_tenant is not None and record_tenant != access.tenant_id:
        return False
    visibility = metadata.get("visibility", "public")
    if visibility == "public":
        return True
    if visibility == "department":
        return metadata.get("department") in access.departments
    if visibility == "private":
        return access.allow_private
    return False


def record_is_visible(
    record: EmbeddingRecord,
    metadata_filter: MetadataFilter | None,
    access: AccessContext | None,
) -> bool:
    return metadata_matches(record, metadata_filter) and acl_allows(record, access)
