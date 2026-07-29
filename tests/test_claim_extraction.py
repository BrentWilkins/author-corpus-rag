"""Adversarial tests for read-only citation-bound answer-claim extraction."""

from datetime import UTC, datetime
from pathlib import Path

from author_corpus.audit import EvidenceSpan
from author_corpus.claim_extraction import extract_answer_claims
from author_corpus.ingestion import load_corpus
from author_corpus.tracing import (
    GenerationTraceSettings,
    QueryTrace,
    RetrievalTraceSettings,
    TracedEvidence,
)


def test_extracts_each_cited_bullet_sentence_without_promoting_it(tmp_path: Path) -> None:
    """Separate claims by sentence while retaining uncited answer prose."""
    trace = _trace(
        tmp_path,
        answer=(
            "Here are the findings:\n\n"
            "* **Policy:** The bridge closes during extreme heat [1]. "
            "The closure protects visitors [1].\n"
            "* No source establishes a winter closure."
        ),
    )

    extraction = extract_answer_claims(trace)

    assert [candidate.statement for candidate in extraction.candidates] == [
        "Policy: The bridge closes during extreme heat.",
        "The closure protects visitors.",
    ]
    assert extraction.uncited_segments == (
        "Here are the findings:",
        "No source establishes a winter closure.",
    )
    assert extraction.exact_span_coverage == (2, 2)
    assert all(candidate.trace_id == trace.trace_id for candidate in extraction.candidates)


def test_exact_citation_retains_span_and_inspectable_classifier_decision(tmp_path: Path) -> None:
    """Assess generated wording against the exact cited text without making it truth."""
    trace = _trace(
        tmp_path,
        answer="The bridge closes during extreme heat [1].",
    )

    candidate = extract_answer_claims(trace).candidates[0]

    assert candidate.is_source_bound is True
    assert candidate.evidence[0].evidence_span == trace.evidence_spans[0]
    assert candidate.evidence[0].classifier_decision is not None
    assert candidate.evidence[0].classifier_decision.label == "supports"


def test_legacy_trace_remains_visibly_unversioned(tmp_path: Path) -> None:
    """Do not manufacture exact provenance for an older copied passage."""
    trace = _trace(
        tmp_path,
        answer="The bridge closes during extreme heat [1].",
        exact_span=False,
    )

    candidate = extract_answer_claims(trace).candidates[0]

    assert candidate.is_source_bound is False
    assert candidate.missing_exact_span_numbers == (1,)
    assert candidate.evidence[0].classifier_decision is None
    assert extract_answer_claims(trace).exact_span_coverage == (0, 1)


def test_unknown_answer_citation_is_preserved_as_unresolved(tmp_path: Path) -> None:
    """Expose a generated citation number that has no traced retrieval evidence."""
    trace = _trace(
        tmp_path,
        answer="The bridge closes during extreme heat [9].",
    )

    candidate = extract_answer_claims(trace).candidates[0]

    assert candidate.citation_numbers == (9,)
    assert candidate.missing_exact_span_numbers == (9,)
    assert candidate.evidence[0].title is None


def test_duplicate_citation_markers_are_deduplicated_in_first_seen_order(tmp_path: Path) -> None:
    """Avoid pretending repeated markers are independent evidence."""
    trace = _trace(
        tmp_path,
        answer="The bridge closes during extreme heat [1][1].",
    )

    candidate = extract_answer_claims(trace).candidates[0]

    assert candidate.citation_numbers == (1,)
    assert len(candidate.evidence) == 1


def test_common_abbreviation_does_not_split_a_claim(tmp_path: Path) -> None:
    """Keep a cited sentence intact when a title contains a period."""
    trace = _trace(
        tmp_path,
        answer="Dr. Rivera reports that the bridge closes during extreme heat [1].",
    )

    extraction = extract_answer_claims(trace)

    assert len(extraction.candidates) == 1
    assert extraction.candidates[0].statement.startswith("Dr. Rivera")


def test_inline_citation_removes_its_dangling_preposition(tmp_path: Path) -> None:
    """Keep readable prose when a citation occurs inside a parenthetical phrase."""
    trace = _trace(
        tmp_path,
        answer="The coach (identified as Rivera in [1]) grew up near the mountains [1].",
    )

    candidate = extract_answer_claims(trace).candidates[0]

    assert candidate.statement == "The coach (identified as Rivera) grew up near the mountains."


def test_candidate_ids_are_stable_for_the_same_trace(tmp_path: Path) -> None:
    """Make read-only selections stable across UI refreshes."""
    trace = _trace(
        tmp_path,
        answer="The bridge closes during extreme heat [1].",
    )

    first = extract_answer_claims(trace)
    second = extract_answer_claims(trace)

    assert first.candidates[0].candidate_id == second.candidates[0].candidate_id


def _trace(
    tmp_path: Path,
    *,
    answer: str,
    exact_span: bool = True,
) -> QueryTrace:
    source_path = tmp_path / "article.md"
    evidence = "The bridge closes during extreme heat."
    source_path.write_text(evidence, encoding="utf-8")
    document = load_corpus([source_path]).documents[0]
    span = EvidenceSpan.from_document(document, evidence)
    traced = TracedEvidence(
        evidence_number=1,
        rank=1,
        document_id=document.document_id,
        document_content_hash=document.content_hash,
        title=document.title,
        source_uris=tuple(source.uri for source in document.sources),
        canonical_source_uri=document.canonical_source.uri if document.canonical_source is not None else None,
        passage_voice="document_author",
        passage_text=evidence,
        passage_hash=span.text_hash,
        evidence_span_id=span.span_id if exact_span else None,
    )
    return QueryTrace(
        trace_id="synthetic-trace",
        created_at=datetime.now(UTC),
        corpus_fingerprint="synthetic-fingerprint",
        query="When does the bridge close?",
        answer=answer,
        cited_evidence_numbers=(1,),
        evidence=(traced,),
        evidence_spans=(span,) if exact_span else (),
        retrieval=RetrievalTraceSettings(
            evidence_limit=1,
            max_passages_per_document=1,
        ),
        generation=GenerationTraceSettings(
            model_id="synthetic-model",
            prompt_version="synthetic-prompt",
            temperature=0.0,
            max_tokens=100,
            reasoning_effort="none",
        ),
        elapsed_seconds=0.1,
    )
