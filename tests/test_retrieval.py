"""Tests for inspectable semantic corpus retrieval."""

import hashlib
import json

import pytest
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from author_corpus.retrieval import SemanticCorpusSearch
from author_corpus.scope import AuthorScope


class SyntheticRetriever(BaseRetriever):
    """Return fixed synthetic candidates for deterministic tests."""

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Return candidates without invoking an embedding model."""
        assert query_bundle.query_str
        return [
            _candidate(
                document_id="work-one",
                title="First Work",
                text="The most relevant synthetic passage.",
                score=0.91,
                document_author_fraction=0.2,
            ),
            _candidate(
                document_id="work-one",
                title="First Work",
                text="Another chunk from the same work.",
                score=0.89,
                document_author_fraction=0.95,
            ),
            _candidate(
                document_id="work-two",
                title="Second Work",
                text="A passage from another synthetic work.",
                score=0.82,
                document_author_fraction=1.0,
            ),
        ]


def test_search_returns_diverse_passages_with_provenance() -> None:
    """Cap repeated documents while retaining normalized source metadata."""
    search = SemanticCorpusSearch(SyntheticRetriever(), default_limit=2)

    result = search.search("synthetic topic")

    assert result.exhaustive is False
    assert result.strategy == "dense"
    assert result.score_kind == "cosine_similarity"
    assert result.inspected_candidates == 3
    assert [passage.document_id for passage in result.passages] == ["work-one", "work-two"]
    assert [passage.rank for passage in result.passages] == [1, 2]
    assert result.passages[0].authors == ("Avery Stone",)
    assert result.passages[0].source_uris == ("https://example.test/first",)
    assert result.passages[0].canonical_source_uri == "https://example.test/first"
    assert result.passages[0].text == "The most relevant synthetic passage."
    assert result.passages[0].score_kind == "cosine_similarity"
    assert result.passages[0].retrieval_contributions[0].method == "dense"
    assert result.passages[0].retrieval_contributions[0].rank == 1


def test_search_can_return_multiple_passages_per_document() -> None:
    """Allow callers to retain multiple chunks when a question needs them."""
    search = SemanticCorpusSearch(
        SyntheticRetriever(),
        default_limit=3,
        max_passages_per_document=2,
    )

    result = search.search("synthetic topic")

    assert [passage.document_id for passage in result.passages] == [
        "work-one",
        "work-one",
        "work-two",
    ]


def test_search_can_require_predominantly_document_author_voice() -> None:
    """Filter quoted candidates before enforcing per-document diversity."""
    search = SemanticCorpusSearch(SyntheticRetriever(), default_limit=2)

    result = search.search("authorial advice", minimum_document_author_fraction=0.8)

    assert [passage.text for passage in result.passages] == [
        "Another chunk from the same work.",
        "A passage from another synthetic work.",
    ]
    assert result.discarded_by_voice_filter == 1


def test_search_hard_filters_authors_and_retains_coauthored_documents() -> None:
    """Exclude unrelated credits while retaining a selected author's shared work."""
    candidates = (
        _candidate(
            document_id="avery-only",
            title="Avery Work",
            text="Avery-only passage.",
            score=0.95,
            document_author_fraction=1.0,
            authors=("Avery Stone",),
        ),
        _candidate(
            document_id="shared",
            title="Shared Work",
            text="Jointly credited passage.",
            score=0.90,
            document_author_fraction=1.0,
            authors=("Avery Stone", "Jamie River"),
        ),
        _candidate(
            document_id="jamie-only",
            title="Jamie Work",
            text="Jamie-only passage.",
            score=0.85,
            document_author_fraction=1.0,
            authors=("Jamie River",),
        ),
    )

    class MultiAuthorRetriever(BaseRetriever):
        """Return a mixed-author ranking."""

        def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
            """Return all candidates in deterministic rank order."""
            assert query_bundle.query_str
            return list(candidates)

    result = SemanticCorpusSearch(MultiAuthorRetriever(), default_limit=3).search(
        "training",
        author_scope=AuthorScope.for_author("Jamie River"),
    )

    assert [passage.document_id for passage in result.passages] == ["shared", "jamie-only"]
    assert result.discarded_by_author_filter == 1
    assert result.represented_authors == ("Jamie River",)
    assert result.missing_scoped_authors == ()


def test_search_reports_missing_comparison_author_without_relaxing_scope() -> None:
    """Expose incomplete comparison coverage instead of admitting another author."""
    result = SemanticCorpusSearch(SyntheticRetriever(), default_limit=3).search(
        "compare approaches",
        author_scope=AuthorScope(kind="comparison", authors=("Avery Stone", "Jamie River")),
    )

    assert result.represented_authors == ("Avery Stone",)
    assert result.missing_scoped_authors == ("Jamie River",)
    assert all(passage.authors == ("Avery Stone",) for passage in result.passages)


def test_search_rejects_corrupt_internal_contribution_metadata() -> None:
    """Fail closed instead of presenting fabricated component provenance."""
    candidate = _candidate(
        document_id="work-one",
        title="First Work",
        text="Evidence.",
        score=0.9,
        document_author_fraction=1.0,
    )
    candidate.node.metadata["_retrieval_contributions"] = "not-json"

    class CorruptRetriever(BaseRetriever):
        """Return one candidate with invalid internal retrieval metadata."""

        def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
            """Return the corrupt candidate."""
            assert query_bundle.query_str
            return [candidate]

    with pytest.raises(ValueError, match="Invalid internal retrieval-contribution metadata"):
        SemanticCorpusSearch(CorruptRetriever()).search("question")


def test_search_reconstructs_exact_versioned_source_span() -> None:
    """Carry absolute source offsets from index metadata into inspected evidence."""
    text = "The exact source passage."
    candidate = _candidate(
        document_id="work-one",
        title="First Work",
        text=text,
        score=0.9,
        document_author_fraction=1.0,
    )
    candidate.node.metadata.update(
        {
            "source_span_version": "source-spans-v1",
            "source_span_id": "exact-span",
            "source_start_char": 17,
            "source_end_char": 17 + len(text),
            "source_text_hash": hashlib.sha256(text.encode()).hexdigest(),
            "document_content_hash": "document-version",
        }
    )

    class ExactSpanRetriever(BaseRetriever):
        """Return one candidate with exact source provenance."""

        def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
            """Return the versioned candidate."""
            assert query_bundle.query_str
            return [candidate]

    passage = SemanticCorpusSearch(ExactSpanRetriever()).search("question").passages[0]

    assert passage.evidence_span is not None
    assert passage.evidence_span.span_id == "exact-span"
    assert passage.evidence_span.start_char == 17
    assert passage.evidence_span.text == text
    assert passage.evidence_span.source_uris == ("https://example.test/first",)


def test_search_rejects_partial_evidence_span_metadata() -> None:
    """Fail closed when an index claims exact provenance without a complete range."""
    candidate = _candidate(
        document_id="work-one",
        title="First Work",
        text="Evidence.",
        score=0.9,
        document_author_fraction=1.0,
    )
    candidate.node.metadata["source_span_id"] = "incomplete"

    class PartialSpanRetriever(BaseRetriever):
        """Return one candidate with partial source provenance."""

        def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
            """Return the corrupt candidate."""
            assert query_bundle.query_str
            return [candidate]

    with pytest.raises(ValueError, match="Incomplete internal evidence-span metadata"):
        SemanticCorpusSearch(PartialSpanRetriever()).search("question")


def _candidate(
    *,
    document_id: str,
    title: str,
    text: str,
    score: float,
    document_author_fraction: float,
    authors: tuple[str, ...] = ("Avery Stone",),
) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(
            text=text,
            metadata={
                "document_id": document_id,
                "title": title,
                "authors": json.dumps(authors),
                "published_at": "2026-01-01",
                "document_type": "article",
                "source_uris": f'["https://example.test/{title.split()[0].lower()}"]',
                "canonical_source_uri": f"https://example.test/{title.split()[0].lower()}",
                "passage_voice": "mixed" if document_author_fraction < 1.0 else "document_author",
                "document_author_fraction": document_author_fraction,
                "quoted_speech_fraction": 1.0 - document_author_fraction,
            },
        ),
        score=score,
    )
