from __future__ import annotations

import pytest

from azusa_rag.filters import AccessContext, MetadataFilter, acl_allows, metadata_matches
from azusa_rag.schemas import EmbeddingRecord
from azusa_rag.search import search_records_with_sources


def record(chunk_id: str, text: str, **metadata) -> EmbeddingRecord:
    return EmbeddingRecord(
        doc_id="doc",
        chunk_id=chunk_id,
        chunk_index=0,
        source="source.md",
        embedding=[1.0, 0.0],
        page=None,
        text=text,
        metadata=metadata,
    )


def test_metadata_matches_requires_all_scalars_and_one_tag() -> None:
    finance = record(
        "finance",
        "budget policy",
        tenant_id="acme",
        department="finance",
        visibility="department",
        tags=["policy", "annual"],
    )
    requested = MetadataFilter(department="finance", tags_any=frozenset({"policy"}))

    assert metadata_matches(finance, requested)
    assert not metadata_matches(finance, MetadataFilter(department="engineering"))
    assert not metadata_matches(finance, MetadataFilter(tags_any=frozenset({"security"})))


def test_acl_allows_is_tenant_isolated_and_fail_closed() -> None:
    finance = record("finance", "budget", tenant_id="acme", department="finance", visibility="department")
    assert not acl_allows(finance, AccessContext("acme", frozenset({"engineering"})))
    assert acl_allows(finance, AccessContext("acme", frozenset({"finance"})))
    assert not acl_allows(finance, AccessContext("other", frozenset({"finance"})))
    secret = record("secret", "secret", visibility="mystery")
    assert not acl_allows(secret, AccessContext(None))


def test_filtered_bm25_overfetches_then_preserves_source_alignment() -> None:
    records = [
        record("private", "azusa azusa", tenant_id="acme", visibility="private"),
        record("public", "azusa", tenant_id="acme", visibility="public"),
    ]
    results, sources = search_records_with_sources(
        "azusa",
        records,
        1,
        "bm25",
        access_context=AccessContext("acme"),
        filter_overfetch=2,
    )

    assert [item.chunk_id for item in results] == ["public"]
    assert sources == [{"id": 1, "source": "bm25 sparse"}]


def test_metadata_filter_parser_rejects_unknown_and_bad_tags() -> None:
    try:
        MetadataFilter.from_dict({"owner": "alice"})
    except ValueError as error:
        assert "Unknown" in str(error)
    else:
        raise AssertionError("unknown filter fields must fail closed")

    for bad_tags in ("policy", ["policy", 3]):
        try:
            MetadataFilter.from_dict({"tags_any": bad_tags})
        except ValueError:
            pass
        else:
            raise AssertionError("tags_any must be a string list")


def test_access_context_parser_rejects_truthy_non_bool() -> None:
    try:
        AccessContext.from_dict({"tenant_id": "acme", "departments": [], "allow_private": 1})
    except ValueError as error:
        assert "boolean" in str(error)
    else:
        raise AssertionError("truthy integers must not grant private access")


@pytest.mark.parametrize("bad", [None, "", False, 0, {}, ["policy", 3]])
@pytest.mark.parametrize("parser,field", [
    (MetadataFilter.from_dict, "tags_any"),
    (AccessContext.from_dict, "departments"),
])
def test_explicit_list_fields_reject_invalid_falsy_values(parser, field, bad) -> None:
    """显式非法假值不得经 or [] 静默变成合法的空列表。"""
    with pytest.raises(ValueError):
        parser({field: bad})


@pytest.mark.parametrize("bad", [[], "", False, 0])
@pytest.mark.parametrize("parser", [MetadataFilter.from_dict, AccessContext.from_dict])
def test_parser_rejects_non_object_payloads(parser, bad) -> None:
    with pytest.raises(ValueError):
        parser(bad)


def test_parsers_preserve_none_and_valid_empty_lists() -> None:
    assert MetadataFilter.from_dict(None) is None
    assert AccessContext.from_dict(None) is None
    assert MetadataFilter.from_dict({"tags_any": []}) == MetadataFilter()
    assert AccessContext.from_dict({"departments": []}) == AccessContext(None)
    assert MetadataFilter.from_dict({"tags_any": ["policy", "policy"]}).tags_any == frozenset({"policy"})


def test_acl_defaults_do_not_grant_private_or_cross_tenant_access() -> None:
    access = AccessContext.from_dict({})
    assert not acl_allows(record("p", "secret", visibility="private"), access)
    assert not acl_allows(record("t", "tenant data", tenant_id="other"), access)
    assert not acl_allows(record("d", "department data", visibility="department"), access)
    assert acl_allows(record("shared", "public data"), access)
