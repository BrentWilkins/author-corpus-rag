"""Tests for explicit human promotion of claim-classifier suggestions."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from author_corpus.audit import AuditedClaim, EvidenceLedger, EvidenceSpan
from author_corpus.claim_classification import classify_claim_evidence
from author_corpus.ingestion import load_corpus
from author_corpus.persistence import corpus_fingerprint
from author_corpus.review import (
    ClaimReviewProposal,
    ClaimReviewStore,
    apply_claim_review,
    review_claim_proposal,
)


def test_proposal_does_not_create_an_audited_claim() -> None:
    """Keep a classifier suggestion pending until a reviewer acts."""
    proposal, _ = _support_proposal()

    assert proposal.suggested_status == "supported"
    assert not hasattr(proposal, "audited_claim")


def test_explicit_acceptance_promotes_exact_evidence_into_ledger() -> None:
    """Create a resolved claim only after an identified reviewer accepts it."""
    proposal, ledger = _support_proposal()

    review = review_claim_proposal(
        proposal,
        action="accept",
        reviewer="Local reviewer",
        notes="Checked against the cited sentence.",
    )
    updated = apply_claim_review(ledger, review)

    assert review.audited_claim is not None
    assert review.audited_claim.status == "supported"
    assert updated.claims == (review.audited_claim,)
    assert updated.evidence_spans == proposal.evidence_spans
    assert updated.claims[0].notes == "Checked against the cited sentence."


def test_ambiguous_classifier_label_cannot_be_accepted_automatically(tmp_path: Path) -> None:
    """Require revision when a heuristic label has no safe claim-status mapping."""
    source_path = tmp_path / "review.md"
    evidence = "The policy review reached a conclusion."
    source_path.write_text(evidence, encoding="utf-8")
    document = load_corpus([source_path]).documents[0]
    decision = classify_claim_evidence(
        "The policy changed after the safety review.",
        evidence,
        evidence_voice="uncertain",
    )
    span = EvidenceSpan.from_document(document, evidence)
    proposal = ClaimReviewProposal.from_decision(
        decision,
        corpus_fingerprint="fingerprint",
        claim_id="policy-change",
        evidence_spans=(span,),
    )

    assert decision.label == "uncertain"
    assert proposal.suggested_status is None
    with pytest.raises(ValueError, match="requires revision"):
        review_claim_proposal(proposal, action="accept", reviewer="Local reviewer")


def test_reviewer_can_revise_attributed_report_with_named_attribution(tmp_path: Path) -> None:
    """Let a reviewer supply the attribution that a classifier cannot establish."""
    source_path = tmp_path / "report.md"
    evidence = 'A witness said, "The bridge closes during extreme heat."'
    source_path.write_text(evidence, encoding="utf-8")
    document = load_corpus([source_path]).documents[0]
    span = EvidenceSpan.from_document(document, evidence)
    decision = classify_claim_evidence(
        "The bridge closes during extreme heat.",
        evidence,
        evidence_voice="quoted_speech",
    )
    fingerprint = corpus_fingerprint([document])
    proposal = ClaimReviewProposal.from_decision(
        decision,
        corpus_fingerprint=fingerprint,
        claim_id="witness-report",
        evidence_spans=(span,),
    )
    revised_claim = AuditedClaim(
        claim_id="witness-report",
        statement="A witness reported that the bridge closes during extreme heat.",
        status="supported",
        evidence_span_ids=(span.span_id,),
        attribution="A witness",
    )

    review = review_claim_proposal(
        proposal,
        action="revise",
        reviewer="Local reviewer",
        revised_claim=revised_claim,
    )

    assert decision.label == "attributed_report"
    assert review.audited_claim == revised_claim


def test_qualification_requires_reviewer_authored_scope(tmp_path: Path) -> None:
    """Do not promote a scope marker until a reviewer states the qualification."""
    source_path = tmp_path / "policy.md"
    evidence = "The bridge closes only when temperatures exceed the posted threshold."
    source_path.write_text(evidence, encoding="utf-8")
    document = load_corpus([source_path]).documents[0]
    span = EvidenceSpan.from_document(document, evidence)
    decision = classify_claim_evidence(
        "The bridge closes.",
        evidence,
        evidence_voice="document_author",
    )
    proposal = ClaimReviewProposal.from_decision(
        decision,
        corpus_fingerprint=corpus_fingerprint([document]),
        claim_id="bridge-closure",
        evidence_spans=(span,),
    )

    assert decision.label == "qualifies"
    assert proposal.suggested_status is None
    with pytest.raises(ValueError, match="requires revision"):
        review_claim_proposal(proposal, action="accept", reviewer="Local reviewer")

    incomplete_claim = AuditedClaim(
        claim_id="bridge-closure",
        statement="The bridge closes above the posted temperature threshold.",
        status="qualified",
        evidence_span_ids=(span.span_id,),
    )
    with pytest.raises(ValidationError, match="explicit qualifier"):
        review_claim_proposal(
            proposal,
            action="revise",
            reviewer="Local reviewer",
            revised_claim=incomplete_claim,
        )

    revised_claim = AuditedClaim(
        claim_id="bridge-closure",
        statement="The bridge closes above the posted temperature threshold.",
        status="qualified",
        evidence_span_ids=(span.span_id,),
        qualifiers=("Only above the posted temperature threshold.",),
    )
    review = review_claim_proposal(
        proposal,
        action="revise",
        reviewer="Local reviewer",
        revised_claim=revised_claim,
    )
    assert review.audited_claim == revised_claim


def test_rejected_review_cannot_mutate_ledger() -> None:
    """Preserve rejection as an audit event without promoting the suggestion."""
    proposal, ledger = _support_proposal()
    review = review_claim_proposal(proposal, action="reject", reviewer="Local reviewer")

    assert review.audited_claim is None
    with pytest.raises(ValueError, match="rejected review"):
        apply_claim_review(ledger, review)


def test_review_cannot_cross_corpus_fingerprint() -> None:
    """Prevent an accepted claim from being applied to a different corpus snapshot."""
    proposal, ledger = _support_proposal()
    review = review_claim_proposal(proposal, action="accept", reviewer="Local reviewer")
    different_ledger = ledger.model_copy(update={"corpus_fingerprint": "different"})

    with pytest.raises(ValueError, match="different corpus fingerprints"):
        apply_claim_review(different_ledger, review)


def test_review_store_round_trips_complete_human_record(tmp_path: Path) -> None:
    """Persist reviewer identity, action, proposal, evidence, and approved claim."""
    proposal, _ = _support_proposal()
    review = review_claim_proposal(proposal, action="accept", reviewer="Local reviewer")
    store = ClaimReviewStore(tmp_path / "reviews.sqlite3")

    store.put(review)
    store.put(review)

    assert store.get(review.review_id) == review
    assert store.recent() == (review,)


def test_review_store_refuses_to_rewrite_an_audit_record(tmp_path: Path) -> None:
    """Make review persistence append-only while retaining idempotent writes."""
    proposal, _ = _support_proposal()
    review = review_claim_proposal(proposal, action="accept", reviewer="Local reviewer")
    changed = review.model_copy(update={"reviewer": "Different reviewer"})
    store = ClaimReviewStore(tmp_path / "reviews.sqlite3")
    store.put(review)

    with pytest.raises(ValueError, match="immutable"):
        store.put(changed)


def test_review_timestamps_must_identify_an_absolute_time() -> None:
    """Reject ambiguous local timestamps in durable human audit records."""
    proposal, _ = _support_proposal()

    with pytest.raises(ValidationError, match="must include a timezone"):
        ClaimReviewProposal.model_validate(proposal.model_copy(update={"created_at": datetime.now()}).model_dump())

    assert proposal.created_at.tzinfo == UTC


def test_proposal_rejects_evidence_unrelated_to_classifier_input(tmp_path: Path) -> None:
    """Prevent a suggestion from attaching a different passage after classification."""
    source_path = tmp_path / "article.md"
    source_path.write_text("The bridge opens at dawn.", encoding="utf-8")
    document = load_corpus([source_path]).documents[0]
    span = EvidenceSpan.from_document(document, "The bridge opens at dawn.")
    decision = classify_claim_evidence(
        "The bridge closes at dusk.",
        "The bridge closes at dusk.",
        evidence_voice="document_author",
    )

    with pytest.raises(ValidationError, match="must match the classified evidence"):
        ClaimReviewProposal.from_decision(
            decision,
            corpus_fingerprint=corpus_fingerprint([document]),
            claim_id="bridge-hours",
            evidence_spans=(span,),
        )


def _support_proposal() -> tuple[ClaimReviewProposal, EvidenceLedger]:
    source_path = Path(__file__).parent / "fixtures" / "contradictory_article.md"
    document = load_corpus([source_path]).documents[0]
    evidence = "the route was closed during extreme heat"
    span = EvidenceSpan.from_document(document, evidence)
    decision = classify_claim_evidence(
        "The route was closed during extreme heat.",
        evidence,
        evidence_voice="document_author",
    )
    fingerprint = corpus_fingerprint([document])
    proposal = ClaimReviewProposal.from_decision(
        decision,
        corpus_fingerprint=fingerprint,
        claim_id="heat-closure",
        evidence_spans=(span,),
    )
    ledger = EvidenceLedger(
        corpus_fingerprint=fingerprint,
        query="When does the route close?",
        answer="The route closes during extreme heat.",
        claims=(),
        evidence_spans=(),
    )
    return proposal, ledger
