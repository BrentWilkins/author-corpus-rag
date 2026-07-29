"""Tests for private generated-answer claim evaluation."""

from pathlib import Path

import pytest

from author_corpus.answering import GroundedAnswer
from author_corpus.audit import EvidenceSpan
from author_corpus.ingestion import load_corpus
from author_corpus.local_llm import LocalModelSettings
from author_corpus.models import CorpusDocument
from author_corpus.persistence import corpus_fingerprint
from author_corpus.retrieval import RetrievedPassage
from author_corpus.trace_evaluation import GeneratedClaimEvaluationCase, evaluate_generated_claims
from author_corpus.tracing import GenerationTraceSettings, QueryTrace, QueryTraceStore, RetrievalTraceSettings


def test_evaluates_exact_trace_claim_without_promoting_it(tmp_path: Path) -> None:
    """Measure a human-labeled generated claim against current raw evidence."""
    document, trace = _trace(tmp_path)
    store = QueryTraceStore(tmp_path / "traces.sqlite3")
    store.put(trace)
    from author_corpus.claim_extraction import extract_answer_claims

    candidate = extract_answer_claims(trace).candidates[0]
    evaluation = evaluate_generated_claims(
        (
            GeneratedClaimEvaluationCase(
                name="closure",
                trace_id=trace.trace_id,
                candidate_id=candidate.candidate_id,
                evidence_number=1,
                expected_label="supports",
                category="factual_support",
            ),
        ),
        store,
        documents={document.document_id: document},
    )

    assert evaluation.accuracy == 1.0
    assert evaluation.exact_span_coverage == 1.0
    assert evaluation.false_support_rate is None
    assert evaluation.abstention_rate == 0.0


def test_rejects_legacy_candidate_without_exact_span(tmp_path: Path) -> None:
    """Refuse to evaluate frozen passage text as current source provenance."""
    document, trace = _trace(tmp_path)
    serialized = trace.model_dump(mode="python")
    serialized["evidence_spans"] = ()
    serialized["evidence"][0]["evidence_span_id"] = None
    legacy = QueryTrace.model_validate(serialized)
    store = QueryTraceStore(tmp_path / "legacy.sqlite3")
    store.put(legacy)
    from author_corpus.claim_extraction import extract_answer_claims

    candidate = extract_answer_claims(legacy).candidates[0]
    case = GeneratedClaimEvaluationCase(
        name="legacy",
        trace_id=legacy.trace_id,
        candidate_id=candidate.candidate_id,
        evidence_number=1,
        expected_label="supports",
        category="legacy",
    )

    with pytest.raises(ValueError, match="not tied to an exact"):
        evaluate_generated_claims((case,), store, documents={document.document_id: document})


def _trace(tmp_path: Path) -> tuple[CorpusDocument, QueryTrace]:
    source = tmp_path / "article.md"
    source.write_text("---\ntitle: Closures\nauthor: Avery Stone\n---\n\nThe route closes during extreme heat.", encoding="utf-8")
    document = load_corpus([source]).documents[0]
    span = EvidenceSpan.from_document(document, "The route closes during extreme heat.")
    passage = RetrievedPassage(
        rank=1,
        document_id=document.document_id,
        title=document.title,
        authors=document.authors,
        document_type=document.document_type,
        text=span.text,
        passage_voice="document_author",
        document_author_fraction=1.0,
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
    )
    return document, trace
