"""Tests for explicit answer review and private export."""

import json
from pathlib import Path

import pytest

from author_corpus.answer_review import (
    AnswerReviewStore,
    export_reviewed_training_examples,
    review_trace_answer,
)
from author_corpus.answering import GroundedAnswer
from author_corpus.audit import EvidenceSpan
from author_corpus.ingestion import load_corpus
from author_corpus.local_llm import LocalModelSettings
from author_corpus.models import CorpusDocument
from author_corpus.persistence import corpus_fingerprint
from author_corpus.retrieval import RetrievedPassage
from author_corpus.scope import AuthorScope
from author_corpus.tracing import GenerationTraceSettings, QueryTrace, QueryTraceStore, RetrievalTraceSettings


def test_accepts_current_exact_answer_and_exports_evidence_in_context(tmp_path: Path) -> None:
    """Export reviewed behavior with evidence instead of relying on memorized facts."""
    document, trace = _trace(tmp_path)
    documents = {document.document_id: document}
    review = review_trace_answer(
        trace,
        documents=documents,
        action="accept",
        reviewer="Reviewer One",
    )
    review_store = AnswerReviewStore(tmp_path / "answer-reviews.sqlite3")
    trace_store = QueryTraceStore(tmp_path / "traces.sqlite3")
    review_store.put(review)
    trace_store.put(trace)
    output = tmp_path / "private" / "reviewed.jsonl"

    count = export_reviewed_training_examples(review_store, trace_store, output)
    record = json.loads(output.read_text(encoding="utf-8"))

    assert count == 1
    assert record["question"] == "When does the route close?"
    assert record["response"] == trace.answer
    assert record["evidence"][0]["text"] == "The route closes during extreme heat."
    assert record["evidence"][0]["authors"] == ["Avery Stone"]
    assert review.author_scope == AuthorScope.for_author("Avery Stone")


def test_revised_answer_requires_valid_citations(tmp_path: Path) -> None:
    """Refuse a behavioral example whose revision drops source grounding."""
    document, trace = _trace(tmp_path)

    with pytest.raises(ValueError, match="requires numbered citations"):
        review_trace_answer(
            trace,
            documents={document.document_id: document},
            action="revise",
            reviewer="Reviewer One",
            revised_answer="The route closes in hot weather.",
        )


def test_accept_refuses_changed_or_unversioned_sources(tmp_path: Path) -> None:
    """Do not approve an answer after its exact evidence becomes stale."""
    document, trace = _trace(tmp_path)
    changed = document.model_copy(update={"content_hash": "changed"})

    with pytest.raises(ValueError, match="current exact source evidence"):
        review_trace_answer(
            trace,
            documents={changed.document_id: changed},
            action="accept",
            reviewer="Reviewer One",
        )


def test_reject_is_auditable_but_not_exported(tmp_path: Path) -> None:
    """Retain rejected reviews while excluding them from training material."""
    document, trace = _trace(tmp_path)
    review = review_trace_answer(
        trace,
        documents={document.document_id: document},
        action="reject",
        reviewer="Reviewer One",
        notes="Unsupported interpretation.",
    )
    review_store = AnswerReviewStore(tmp_path / "answer-reviews.sqlite3")
    trace_store = QueryTraceStore(tmp_path / "traces.sqlite3")
    review_store.put(review)
    trace_store.put(trace)

    count = export_reviewed_training_examples(review_store, trace_store, tmp_path / "reviewed.jsonl")

    assert count == 0
    assert review_store.get(review.review_id) == review


def _trace(tmp_path: Path) -> tuple[CorpusDocument, QueryTrace]:
    source = tmp_path / "article.md"
    source.write_text(
        "---\ntitle: Closures\nauthor: Avery Stone\n---\n\nThe route closes during extreme heat.",
        encoding="utf-8",
    )
    document = load_corpus([source]).documents[0]
    span = EvidenceSpan.from_document(document, "The route closes during extreme heat.")
    passage = RetrievedPassage(
        rank=1,
        document_id=document.document_id,
        title=document.title,
        authors=document.authors,
        document_type=document.document_type,
        source_uris=tuple(item.uri for item in document.sources),
        canonical_source_uri=document.canonical_source.uri if document.canonical_source else None,
        passage_voice="document_author",
        document_author_fraction=1.0,
        text=span.text,
        evidence_span=span,
    )
    answer = GroundedAnswer(
        query="When does the route close?",
        answer="The route closes during extreme heat [1].",
        evidence=(passage,),
        cited_evidence_numbers=(1,),
        model_id="test-model",
        prompt_version="test-v1",
    )
    settings = LocalModelSettings(model_id="test-model", reasoning_effort="none")
    trace = QueryTrace.from_grounded_answer(
        answer,
        corpus_fingerprint=corpus_fingerprint([document]),
        document_content_hashes={document.document_id: document.content_hash},
        retrieval=RetrievalTraceSettings(evidence_limit=1, max_passages_per_document=1),
        generation=GenerationTraceSettings.from_local_settings(settings, prompt_version=answer.prompt_version),
        elapsed_seconds=0.1,
        user_query="When does the route close?",
        author_scope=AuthorScope.for_author("Avery Stone"),
    )
    return document, trace
