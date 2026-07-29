"""Tests for single-retrieval query execution and bounded conversation context."""

from pathlib import Path

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from author_corpus import (
    ConversationMessage,
    ConversationRole,
    CorpusQueryService,
    ExactCatalogStatus,
    ask_conversational,
    resolve_conversation_query,
)
from author_corpus.catalog import CorpusCatalog
from author_corpus.hybrid import RetrievalProfiles
from author_corpus.ingestion import load_corpus
from author_corpus.local_llm import LocalModelSettings
from author_corpus.persistence import corpus_fingerprint
from author_corpus.retrieval import SemanticCorpusSearch
from author_corpus.service import QueryTraceContext
from author_corpus.tracing import QueryTraceStore


class CountingRetriever(BaseRetriever):
    """Return one source-bearing passage and count retrieval calls."""

    def __init__(self) -> None:
        """Initialize with no calls."""
        self.queries: list[str] = []
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Record the effective query and return deterministic evidence."""
        self.queries.append(query_bundle.query_str)
        return [
            NodeWithScore(
                node=TextNode(
                    text="The source recommends a gradual synthetic approach.",
                    metadata={
                        "document_id": "synthetic-work",
                        "title": "Synthetic Work",
                        "authors": '["Avery Stone"]',
                        "document_type": "article",
                        "canonical_source_uri": "https://example.test/work",
                        "passage_voice": "document_author",
                        "document_author_fraction": 1.0,
                        "quoted_speech_fraction": 0.0,
                        "uncertain_voice_fraction": 0.0,
                    },
                ),
                score=0.9,
            )
        ]


def _service(tmp_path: Path, retriever: CountingRetriever, *, with_generation: bool) -> CorpusQueryService:
    source = tmp_path / "work.md"
    source.write_text(
        "---\ntitle: Synthetic Work\nauthor: Avery Stone\ndocument_type: article\n---\n\nSynthetic body.",
        encoding="utf-8",
    )
    loaded = load_corpus([source])
    loaded.raise_for_errors()
    catalog = CorpusCatalog(tmp_path / "catalog.sqlite3")
    catalog.rebuild(loaded.documents)
    profiles = RetrievalProfiles(
        discovery=SemanticCorpusSearch(retriever, strategy="dense_discovery"),
        dense_evidence=SemanticCorpusSearch(retriever, strategy="dense_evidence"),
        lexical_evidence=SemanticCorpusSearch(retriever, strategy="lexical_evidence", score_kind="bm25"),
        hybrid_evidence=SemanticCorpusSearch(
            retriever,
            strategy="hybrid_evidence",
            score_kind="reciprocal_rank_fusion",
        ),
    )
    if with_generation:
        model_settings = LocalModelSettings(model_id="synthetic-model", reasoning_effort="none")
        return CorpusQueryService(
            catalog,
            profiles,
            complete=lambda prompt: "The source recommends a gradual approach [1].",
            model_id=model_settings.model_id,
            default_author="Avery Stone",
            trace_context=QueryTraceContext(
                store=QueryTraceStore(tmp_path / "query-traces.sqlite3"),
                corpus_fingerprint=corpus_fingerprint(loaded.documents),
                document_content_hashes={document.document_id: document.content_hash for document in loaded.documents},
                generation_settings=model_settings,
            ),
        )
    return CorpusQueryService(catalog, profiles, default_author="Avery Stone")


def test_exact_question_bypasses_retrieval_and_generation(tmp_path: Path) -> None:
    """Keep counts entirely on the validated exhaustive catalog path."""
    retriever = CountingRetriever()
    service = _service(tmp_path, retriever, with_generation=True)

    result = service.ask("How many articles has Avery written here?")

    assert result.exact is not None
    assert result.exact.status is ExactCatalogStatus.COMPLETED
    assert result.exact.count == 1
    assert result.semantic is None
    assert result.grounded_answer is None
    assert retriever.queries == []
    assert [timing.label for timing in result.timings] == ["routing", "exact catalog"]


def test_semantic_generation_reuses_the_inspected_retrieval(tmp_path: Path) -> None:
    """Generate from the first evidence set instead of silently retrieving twice."""
    retriever = CountingRetriever()
    service = _service(tmp_path, retriever, with_generation=True)

    result = service.ask("What approach does the article recommend?")

    assert len(retriever.queries) == 1
    assert result.semantic is not None
    assert result.semantic.strategy == "hybrid_evidence"
    assert result.grounded_answer is not None
    assert result.grounded_answer.evidence == result.semantic.passages
    assert result.trace_id is not None
    assert service.trace_context is not None
    trace = service.trace_context.store.get(result.trace_id)
    assert trace is not None
    assert trace.user_query == "What approach does the article recommend?"
    assert trace.evidence[0].passage_text == result.semantic.passages[0].text
    assert [timing.label for timing in result.timings] == [
        "routing",
        "retrieval",
        "generation",
        "trace persistence",
    ]


def test_broad_question_selects_discovery_without_generation(tmp_path: Path) -> None:
    """Use the diverse discovery profile when broad exploration is requested."""
    retriever = CountingRetriever()
    service = _service(tmp_path, retriever, with_generation=False)

    result = service.ask("What themes recur across the corpus?")

    assert result.semantic is not None
    assert result.semantic.strategy == "dense_discovery"
    assert result.grounded_answer is None
    assert "Generation is disabled" in result.to_markdown()


def test_semantic_follow_up_uses_only_previous_user_question(tmp_path: Path) -> None:
    """Never recycle generated assistant text as retrieval evidence or context."""
    retriever = CountingRetriever()
    service = _service(tmp_path, retriever, with_generation=False)
    history = (
        ConversationMessage(role=ConversationRole.USER, content="What does the article recommend before a race?"),
        ConversationMessage(role=ConversationRole.ASSISTANT, content="An unsupported assistant draft."),
    )

    turn = ask_conversational(service, "What about after the race?", history, generate=False)

    assert turn.resolution.used_previous_user_query is True
    assert "What does the article recommend before a race?" in turn.resolution.retrieval_query
    assert "unsupported assistant draft" not in turn.resolution.retrieval_query
    assert retriever.queries == [turn.resolution.retrieval_query]


def test_exact_follow_up_is_never_contextualized() -> None:
    """Keep exact argument extraction tied to literal current wording."""
    history = (ConversationMessage(role=ConversationRole.USER, content="What topics recur?"),)

    resolution = resolve_conversation_query("Who are the other authors?", history)

    assert resolution.used_previous_user_query is False
    assert resolution.retrieval_query == "Who are the other authors?"


def test_self_contained_semantic_question_does_not_inherit_history() -> None:
    """Avoid polluting a new topic with stale conversation context."""
    history = (ConversationMessage(role=ConversationRole.USER, content="What topics recur?"),)

    resolution = resolve_conversation_query("How does heat acclimation change plasma volume?", history)

    assert resolution.used_previous_user_query is False
    assert resolution.retrieval_query == resolution.query
