"""Tests for validated natural-language execution against the exact catalog."""

from pathlib import Path

import pytest

from author_corpus import ExactCatalogStatus, QueryRouteDecision, execute_catalog_query, route_query
from author_corpus.catalog import CorpusCatalog
from author_corpus.ingestion import load_corpus


@pytest.fixture
def catalog(tmp_path: Path) -> CorpusCatalog:
    """Build a small mixed-author, mixed-type exact catalog."""
    works = (
        (
            "first.md",
            """\
---
title: First Article
authors: [Avery Stone, Jordan Vale]
document_type: article
source_uri: https://example.test/first
---

First body.
""",
        ),
        (
            "second.md",
            """\
---
title: Second Article
author: Avery Stone
document_type: article
source_uri: https://example.test/second
---

Second body.
""",
        ),
        (
            "paper.md",
            """\
---
title: Research Paper
author: Jordan Vale
document_type: paper
source_uri: https://example.test/paper
---

Paper body.
""",
        ),
    )
    for filename, content in works:
        (tmp_path / filename).write_text(content, encoding="utf-8")
    result = load_corpus([tmp_path])
    result.raise_for_errors()
    value = CorpusCatalog(tmp_path / "generated" / "catalog.sqlite3")
    value.rebuild(result.documents)
    return value


def test_executes_explicit_author_count_with_supporting_sources(catalog: CorpusCatalog) -> None:
    """Resolve exact known filters and retain every supporting document record."""
    decision = route_query("How many articles has Avery Stone written here?")

    result = execute_catalog_query(catalog, decision)

    assert result.status is ExactCatalogStatus.COMPLETED
    assert result.arguments.author == "Avery Stone"
    assert result.arguments.document_type == "article"
    assert result.count == 2
    assert len(result.documents) == 2
    assert all(document.source_uris for document in result.documents)
    assert result.coverage is not None
    assert result.coverage.corpus_documents == 3
    assert result.coverage.matched_documents == 2


def test_resolves_generic_author_only_from_validated_private_default(catalog: CorpusCatalog) -> None:
    """Use a configured focal author for generic wording after exact validation."""
    decision = route_query("How many articles has the author written here?")

    result = execute_catalog_query(catalog, decision, default_author="avery stone")

    assert result.status is ExactCatalogStatus.COMPLETED
    assert result.arguments.author == "Avery Stone"
    assert result.count == 2


@pytest.mark.parametrize(
    "query",
    [
        "How many articles has Avery written here?",
        "How many articles has she written here?",
    ],
)
def test_resolves_unique_short_name_or_pronoun_without_guessing(catalog: CorpusCatalog, query: str) -> None:
    """Resolve a unique name component or a pronoun backed by the private default."""
    decision = route_query(query)

    result = execute_catalog_query(catalog, decision, default_author="Avery Stone")

    assert result.status is ExactCatalogStatus.COMPLETED
    assert result.arguments.author == "Avery Stone"
    assert result.count == 2


def test_unknown_explicit_author_never_falls_back_to_default(catalog: CorpusCatalog) -> None:
    """Do not turn an unknown named author into an unfiltered or default count."""
    decision = route_query("How many articles has Rowan Pierce written here?")

    result = execute_catalog_query(catalog, decision, default_author="Avery Stone")

    assert result.status is ExactCatalogStatus.NEEDS_CLARIFICATION
    assert result.count is None
    assert "Rowan Pierce" in result.message
    assert result.coverage is None


def test_ambiguous_short_name_requests_clarification(tmp_path: Path) -> None:
    """Never choose between multiple catalog authors sharing a name component."""
    for index, author in enumerate(("Avery Stone", "Avery Reed")):
        (tmp_path / f"work-{index}.md").write_text(
            f"---\nauthor: {author}\ndocument_type: article\n---\n\nSynthetic body {index}.",
            encoding="utf-8",
        )
    loaded = load_corpus([tmp_path])
    loaded.raise_for_errors()
    ambiguous_catalog = CorpusCatalog(tmp_path / "generated" / "catalog.sqlite3")
    ambiguous_catalog.rebuild(loaded.documents)
    decision = route_query("How many articles has Avery written here?")

    result = execute_catalog_query(ambiguous_catalog, decision, default_author="Avery Stone")

    assert result.status is ExactCatalogStatus.NEEDS_CLARIFICATION
    assert "ambiguous" in result.message
    assert "Avery Reed" in result.message
    assert "Avery Stone" in result.message


def test_invalid_default_author_reports_configuration_error(catalog: CorpusCatalog) -> None:
    """Reject a stale or misspelled private focal-author setting."""
    decision = route_query("Show the sole-authored and coauthored writing credits.")

    result = execute_catalog_query(catalog, decision, default_author="Avery Stones")

    assert result.status is ExactCatalogStatus.NEEDS_CLARIFICATION
    assert "not in the catalog" in result.message


def test_other_authors_excludes_validated_focal_author(catalog: CorpusCatalog) -> None:
    """Interpret other authors relative to a validated focal author."""
    decision = route_query("List every other writer in the collection.")

    result = execute_catalog_query(catalog, decision, default_author="Avery Stone")

    assert result.status is ExactCatalogStatus.COMPLETED
    assert result.arguments.exclude_author == "Avery Stone"
    assert [(author.name, author.document_count) for author in result.authors] == [("Jordan Vale", 2)]
    assert [document.title for document in result.documents] == ["First Article", "Research Paper"]


def test_unknown_document_type_does_not_report_misleading_zero(catalog: CorpusCatalog) -> None:
    """Request clarification when metadata has no requested document type."""
    decision = route_query("How many posts are in the corpus?")

    result = execute_catalog_query(catalog, decision)

    assert result.status is ExactCatalogStatus.NEEDS_CLARIFICATION
    assert "Available types: article, paper" in result.message
    assert result.count is None


def test_total_works_counts_every_document_without_inventing_filters(catalog: CorpusCatalog) -> None:
    """Treat generic works as all logical documents rather than a document type."""
    decision = route_query("What is the total number of works in this corpus?")

    result = execute_catalog_query(catalog, decision)

    assert result.status is ExactCatalogStatus.COMPLETED
    assert result.arguments.author is None
    assert result.arguments.document_type is None
    assert result.count == 3


def test_source_inventory_for_all_documents_is_not_mistaken_for_an_author(catalog: CorpusCatalog) -> None:
    """Do not parse the preposition in a source inventory as a person filter."""
    decision = route_query("Give me the complete source URL inventory for all documents.")

    result = execute_catalog_query(catalog, decision, default_author="Avery Stone")

    assert result.status is ExactCatalogStatus.COMPLETED
    assert result.arguments.author is None
    assert len(result.documents) == 3


def test_unknown_author_in_document_filter_does_not_return_every_document(catalog: CorpusCatalog) -> None:
    """Reject an unknown person in a document-list filter instead of dropping it."""
    decision = route_query("List all articles for Rowan Pierce.")

    result = execute_catalog_query(catalog, decision, default_author="Avery Stone")

    assert result.status is ExactCatalogStatus.NEEDS_CLARIFICATION
    assert "Rowan Pierce" in result.message
    assert result.documents == ()


def test_semantic_decision_cannot_be_executed_as_exact(catalog: CorpusCatalog) -> None:
    """Protect the executor boundary from semantic routing decisions."""
    decision: QueryRouteDecision = route_query("What themes recur across the corpus?")

    with pytest.raises(ValueError, match="Only exact catalog"):
        execute_catalog_query(catalog, decision)
