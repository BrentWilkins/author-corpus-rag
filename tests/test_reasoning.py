"""Tests for bounded, claim-verified multi-step reasoning."""

import json
from pathlib import Path

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from author_corpus.audit import EVIDENCE_SPAN_VERSION, EvidenceSpan
from author_corpus.ingestion import load_corpus
from author_corpus.reasoning import (
    BoundedReasoningEngine,
    ConservativeClaimVerifier,
    plan_reasoning,
    requires_multistep_reasoning,
)
from author_corpus.retrieval import SemanticCorpusSearch
from author_corpus.scope import AuthorScope


class StaticEvidenceRetriever(BaseRetriever):
    """Return one exact-span passage while recording retrieval queries."""

    def __init__(self, node: TextNode) -> None:
        """Initialize deterministic evidence."""
        self.node = node
        self.queries: list[str] = []
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Record and return one passage."""
        self.queries.append(query_bundle.query_str)
        return [NodeWithScore(node=self.node, score=0.9)]


def test_plans_compositional_question_with_explicit_scope() -> None:
    """Preserve author scope while decomposing explicit contrasting clauses."""
    scope = AuthorScope.for_author("Avery Stone")
    plan = plan_reasoning("Compare early advice versus later advice", author_scope=scope)

    assert requires_multistep_reasoning(plan.question) is True
    assert plan.author_scope == scope
    assert [step.question for step in plan.steps] == ["Compare early advice", "later advice"]
    assert plan.maximum_rounds == 2


def test_reasoning_returns_only_claims_supported_by_exact_spans(tmp_path: Path) -> None:
    """Retain a supported claim and its original citation after verification."""
    search, retriever = _search(tmp_path)
    engine = BoundedReasoningEngine(
        search,
        lambda prompt: "The route closes during extreme heat [1].",
        model_id="test-model",
        verifier=ConservativeClaimVerifier(),
    )

    result = engine.answer("When does the route close?", author_scope=AuthorScope.for_author("Avery Stone"))

    assert result.grounded_answer.answer == "The route closes during extreme heat. [1]"
    assert result.grounded_answer.cited_evidence_numbers == (1,)
    assert result.reasoning.rounds[0].verifications[0].status == "supported"
    assert len(result.reasoning.rounds) == 1
    assert retriever.queries == ["When does the route close?"]


def test_reasoning_retries_once_then_abstains_on_unsupported_claim(tmp_path: Path) -> None:
    """Bound corrective retrieval and omit a claim that never gains support."""
    search, retriever = _search(tmp_path)
    engine = BoundedReasoningEngine(
        search,
        lambda prompt: "The route is always open [1].",
        model_id="test-model",
    )

    result = engine.answer("Is the route always open?")

    assert result.grounded_answer.answer == "The retrieved evidence is insufficient to answer this question."
    assert result.grounded_answer.cited_evidence_numbers == ()
    assert len(result.reasoning.rounds) == 2
    assert result.reasoning.rounds[-1].verifications[0].may_answer is False
    assert len(retriever.queries) == 2


def _search(tmp_path: Path) -> tuple[SemanticCorpusSearch, StaticEvidenceRetriever]:
    source = tmp_path / "closure.md"
    source.write_text(
        "---\ntitle: Closure Guide\nauthor: Avery Stone\n---\n\nThe route closes during extreme heat.",
        encoding="utf-8",
    )
    document = load_corpus([source]).documents[0]
    span = EvidenceSpan.from_document(document, "The route closes during extreme heat.")
    node = TextNode(
        text=span.text,
        metadata={
            "document_id": document.document_id,
            "title": document.title,
            "authors": json.dumps(document.authors),
            "document_type": document.document_type,
            "source_uris": json.dumps([source.uri for source in document.sources]),
            "passage_voice": "document_author",
            "document_author_fraction": 1.0,
            "source_span_version": EVIDENCE_SPAN_VERSION,
            "source_span_id": span.span_id,
            "source_start_char": span.start_char,
            "source_end_char": span.end_char,
            "source_text_hash": span.text_hash,
            "document_content_hash": span.document_content_hash,
        },
    )
    retriever = StaticEvidenceRetriever(node)
    return SemanticCorpusSearch(retriever, default_limit=5), retriever
