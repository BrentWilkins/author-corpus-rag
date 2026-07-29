"""Tests for durable, source-aware grounded-answer traces."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from author_corpus.answering import GroundedAnswer
from author_corpus.audit import EvidenceSpan
from author_corpus.ingestion import load_corpus
from author_corpus.local_llm import LocalModelSettings
from author_corpus.persistence import corpus_fingerprint
from author_corpus.retrieval import RetrievalContribution, RetrievedPassage
from author_corpus.tracing import (
    GenerationTraceSettings,
    QueryTrace,
    QueryTraceStore,
    RetrievalTraceSettings,
)


def test_query_trace_round_trip_and_freshness(tmp_path: Path) -> None:
    """Persist exact evidence and detect a later source-content change."""
    source_path = tmp_path / "work.md"
    source_path.write_text("# Synthetic Work\n\nThe route closes during extreme heat.", encoding="utf-8")
    document = load_corpus([source_path]).documents[0]
    evidence_span = EvidenceSpan.from_document(document, "The route closes during extreme heat.")
    answer = GroundedAnswer(
        query="When does the route close?",
        answer="The route closes during extreme heat [1].",
        evidence=(
            RetrievedPassage(
                rank=1,
                score=0.91,
                score_kind="reciprocal_rank_fusion",
                retrieval_contributions=(
                    RetrievalContribution(
                        method="dense",
                        rank=2,
                        score=0.91,
                        score_kind="cosine_similarity",
                        rrf_contribution=0.016,
                    ),
                ),
                document_id=document.document_id,
                title=document.title,
                document_type=document.document_type,
                source_uris=tuple(source.uri for source in document.sources),
                canonical_source_uri=document.canonical_source.uri if document.canonical_source else None,
                section_path=("Guide", "Closures"),
                passage_voice="document_author",
                document_author_fraction=1.0,
                quoted_speech_fraction=0.0,
                uncertain_voice_fraction=0.0,
                text="The route closes during extreme heat.",
                evidence_span=evidence_span,
            ),
        ),
        cited_evidence_numbers=(1,),
        model_id="synthetic-model",
        prompt_version="synthetic-prompt-v1",
    )
    local_settings = LocalModelSettings(
        model_id="synthetic-model",
        max_tokens=321,
        reasoning_effort="none",
    )
    trace = QueryTrace.from_grounded_answer(
        answer,
        corpus_fingerprint=corpus_fingerprint([document]),
        document_content_hashes={document.document_id: document.content_hash},
        retrieval=RetrievalTraceSettings(
            evidence_limit=5,
            max_passages_per_document=1,
            minimum_document_author_fraction=0.8,
            strategy="hybrid_evidence",
            score_kind="reciprocal_rank_fusion",
        ),
        generation=GenerationTraceSettings.from_local_settings(
            local_settings,
            prompt_version=answer.prompt_version,
        ),
        elapsed_seconds=1.25,
    )

    store = QueryTraceStore(tmp_path / "traces.sqlite3")
    store.put(trace)
    restored = store.get(trace.trace_id)

    assert restored == trace
    assert store.recent() == (trace,)
    assert trace.evidence[0].passage_text == "The route closes during extreme heat."
    assert trace.evidence[0].authors == document.authors
    assert trace.evidence[0].section_path == ("Guide", "Closures")
    assert trace.evidence[0].passage_voice == "document_author"
    assert trace.evidence[0].uncertain_voice_fraction == 0.0
    assert trace.evidence[0].score_kind == "reciprocal_rank_fusion"
    assert trace.evidence[0].retrieval_contributions[0].method == "dense"
    assert trace.retrieval.minimum_document_author_fraction == 0.8
    assert trace.retrieval.strategy == "hybrid_evidence"
    assert trace.user_query is None
    assert trace.author_scope.kind == "corpus"
    assert len(trace.evidence[0].passage_hash) == 64
    assert trace.evidence[0].evidence_span_id == evidence_span.span_id
    assert trace.evidence_spans == (evidence_span,)
    assert trace.cited_evidence_spans == (evidence_span,)
    assert trace.cited_span_coverage == (1, 1)
    assert trace.check_freshness({document.document_id: document}).is_current is True

    changed_document = document.model_copy(update={"content_hash": "changed-content-hash"})
    freshness = trace.check_freshness({document.document_id: changed_document})
    assert freshness.is_current is False
    assert freshness.changed_document_ids == (document.document_id,)
    assert freshness.evidence_issues[0].code == "document_changed"


def test_historical_trace_without_source_ranges_is_explicitly_unversioned(tmp_path: Path) -> None:
    """Load older JSON records while refusing to call passage hashes exact spans."""
    source_path = tmp_path / "legacy.md"
    source_path.write_text("Legacy evidence.", encoding="utf-8")
    document = load_corpus([source_path]).documents[0]
    passage = RetrievedPassage(
        rank=1,
        document_id=document.document_id,
        title=document.title,
        document_type=document.document_type,
        text="Legacy evidence.",
    )
    answer = GroundedAnswer(
        query="What is the evidence?",
        answer="Legacy evidence [1].",
        evidence=(passage,),
        cited_evidence_numbers=(1,),
        model_id="synthetic-model",
        prompt_version="synthetic-prompt-v1",
    )
    settings = LocalModelSettings(model_id="synthetic-model", reasoning_effort="none")
    trace = QueryTrace.from_grounded_answer(
        answer,
        corpus_fingerprint=corpus_fingerprint([document]),
        document_content_hashes={document.document_id: document.content_hash},
        retrieval=RetrievalTraceSettings(evidence_limit=1, max_passages_per_document=1),
        generation=GenerationTraceSettings.from_local_settings(settings, prompt_version=answer.prompt_version),
        elapsed_seconds=0.1,
    )

    serialized = trace.model_dump(mode="json")
    serialized.pop("evidence_spans")
    for item in serialized["evidence"]:
        item.pop("evidence_span_id")
    restored = QueryTrace.model_validate(serialized)
    freshness = restored.check_freshness({document.document_id: document})

    assert restored.evidence_spans == ()
    assert freshness.is_current is False
    assert freshness.unversioned_document_ids == (document.document_id,)


def test_trace_rejects_duplicate_evidence_numbers(tmp_path: Path) -> None:
    """Prevent ambiguous citation-to-passage links in persisted traces."""
    source_path = tmp_path / "duplicate.md"
    source_path.write_text("Evidence.", encoding="utf-8")
    document = load_corpus([source_path]).documents[0]
    passage = RetrievedPassage(
        rank=1,
        document_id=document.document_id,
        title=document.title,
        document_type=document.document_type,
        text="Evidence.",
    )
    answer = GroundedAnswer(
        query="What is the evidence?",
        answer="Evidence [1].",
        evidence=(passage,),
        cited_evidence_numbers=(1,),
        model_id="synthetic-model",
        prompt_version="synthetic-prompt-v1",
    )
    settings = LocalModelSettings(model_id="synthetic-model", reasoning_effort="none")
    trace = QueryTrace.from_grounded_answer(
        answer,
        corpus_fingerprint=corpus_fingerprint([document]),
        document_content_hashes={document.document_id: document.content_hash},
        retrieval=RetrievalTraceSettings(evidence_limit=1, max_passages_per_document=1),
        generation=GenerationTraceSettings.from_local_settings(settings, prompt_version=answer.prompt_version),
        elapsed_seconds=0.1,
    )
    duplicate = trace.evidence[0].model_copy(update={"rank": 2})

    with pytest.raises(ValidationError, match="duplicate evidence numbers"):
        QueryTrace.model_validate(
            {
                **trace.model_dump(),
                "evidence": (trace.evidence[0], duplicate),
            }
        )
