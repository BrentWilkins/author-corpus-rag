"""Tests for inspectable semantic corpus retrieval."""

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from author_corpus.retrieval import SemanticCorpusSearch


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
    assert result.inspected_candidates == 3
    assert [passage.document_id for passage in result.passages] == ["work-one", "work-two"]
    assert [passage.rank for passage in result.passages] == [1, 2]
    assert result.passages[0].authors == ("Avery Stone",)
    assert result.passages[0].source_uris == ("https://example.test/first",)
    assert result.passages[0].canonical_source_uri == "https://example.test/first"
    assert result.passages[0].text == "The most relevant synthetic passage."


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


def _candidate(
    *,
    document_id: str,
    title: str,
    text: str,
    score: float,
    document_author_fraction: float,
) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(
            text=text,
            metadata={
                "document_id": document_id,
                "title": title,
                "authors": '["Avery Stone"]',
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
