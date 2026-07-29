"""Tests for private aggregate claim-verifier evaluation."""

from pathlib import Path

import pytest

from author_corpus.answering import GroundedAnswer
from author_corpus.audit import EvidenceSpan
from author_corpus.claim_extraction import AnswerClaimCandidate, extract_answer_claims
from author_corpus.ingestion import load_corpus
from author_corpus.local_llm import LocalModelSettings
from author_corpus.models import CorpusDocument
from author_corpus.persistence import corpus_fingerprint
from author_corpus.reasoning_models import ClaimVerification, VerificationStatus
from author_corpus.retrieval import RetrievedPassage
from author_corpus.tracing import GenerationTraceSettings, QueryTrace, QueryTraceStore, RetrievalTraceSettings
from author_corpus.verifier_evaluation import VerifierEvaluationCase, evaluate_verifiers, load_verifier_cases


class FixedVerifier:
    """Return one configured status while preserving candidate provenance."""

    def __init__(self, status: VerificationStatus) -> None:
        """Store the aggregate status used for every case."""
        self.status = status

    def verify_many(self, candidates: tuple[AnswerClaimCandidate, ...]) -> tuple[ClaimVerification, ...]:
        """Build deterministic decisions in the supplied order."""
        return tuple(
            ClaimVerification(
                candidate_id=candidate.candidate_id,
                statement=candidate.statement,
                citation_numbers=candidate.citation_numbers,
                status=self.status,
                verifier_id=f"fixed-{self.status}",
                rationale="Synthetic evaluator decision.",
            )
            for candidate in candidates
        )


def test_compares_coverage_and_false_acceptance_on_identical_cases(tmp_path: Path) -> None:
    """Separate abstention coverage from unsafe admission on a human negative."""
    document, trace = _trace(tmp_path)
    store = QueryTraceStore(tmp_path / "traces.sqlite3")
    store.put(trace)
    candidate = extract_answer_claims(trace).candidates[0]
    cases = (
        VerifierEvaluationCase(
            name="unsupported closure",
            trace_id=trace.trace_id,
            candidate_id=candidate.candidate_id,
            expected_status="unsupported",
            category="unsupported",
        ),
    )

    comparison = evaluate_verifiers(
        cases,
        store,
        documents={document.document_id: document},
        verifiers={
            "always support": FixedVerifier("supported"),
            "always abstain": FixedVerifier("uncertain"),
        },
    )

    support, abstain = comparison.evaluations
    assert support.accuracy == 0.0
    assert support.coverage == 1.0
    assert support.selective_accuracy == 0.0
    assert support.false_acceptance_rate == 1.0
    assert abstain.accuracy == 0.0
    assert abstain.coverage == 0.0
    assert abstain.selective_accuracy is None
    assert abstain.false_acceptance_rate == 0.0


def test_loads_private_verifier_cases_and_rejects_duplicate_candidates(tmp_path: Path) -> None:
    """Validate ignored YAML labels before any model calls are made."""
    path = tmp_path / "verifier-cases.yaml"
    path.write_text(
        """\
cases:
  - name: first
    trace_id: trace-1
    candidate_id: candidate-1
    expected_status: supported
    category: entailment
""",
        encoding="utf-8",
    )

    cases = load_verifier_cases(path)

    assert cases[0].category == "entailment"
    path.write_text(
        """\
cases:
  - &case
    name: first
    trace_id: trace-1
    candidate_id: candidate-1
    expected_status: supported
    category: entailment
  - <<: *case
    name: duplicate
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot repeat"):
        load_verifier_cases(path)


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
        query="Is the route always open?",
        answer="The route is always open [1].",
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
