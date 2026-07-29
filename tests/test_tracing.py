"""Tests for durable, source-aware grounded-answer traces."""

from pathlib import Path

from author_corpus.answering import GroundedAnswer
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
    assert trace.evidence[0].section_path == ("Guide", "Closures")
    assert trace.evidence[0].passage_voice == "document_author"
    assert trace.evidence[0].uncertain_voice_fraction == 0.0
    assert trace.evidence[0].score_kind == "reciprocal_rank_fusion"
    assert trace.evidence[0].retrieval_contributions[0].method == "dense"
    assert trace.retrieval.minimum_document_author_fraction == 0.8
    assert trace.retrieval.strategy == "hybrid_evidence"
    assert trace.user_query is None
    assert len(trace.evidence[0].passage_hash) == 64
    assert trace.check_freshness({document.document_id: document}).is_current is True

    changed_document = document.model_copy(update={"content_hash": "changed-content-hash"})
    freshness = trace.check_freshness({document.document_id: changed_document})
    assert freshness.is_current is False
    assert freshness.changed_document_ids == (document.document_id,)
