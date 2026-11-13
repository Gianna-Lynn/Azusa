from __future__ import annotations

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
