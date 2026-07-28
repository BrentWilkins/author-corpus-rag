"""Adversarial tests for exact evidence and contradictory claims."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from author_corpus.audit import AuditedClaim, ClaimRelation, EvidenceLedger, EvidenceSpan
from author_corpus.ingestion import load_corpus
from author_corpus.persistence import corpus_fingerprint

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "contradictory_article.md"


def test_ledger_preserves_correction_qualifier_and_counterexample() -> None:
    """Keep a correction distinct from a scoped policy and personal quotation."""
    document = load_corpus([FIXTURE_PATH]).documents[0]
    early = EvidenceSpan.from_document(
        document,
        "An early bulletin described the route as safe in all weather.",
        span_id="early-report",
    )
    correction = EvidenceSpan.from_document(
        document,
        "After a later inspection, the route was closed during extreme heat.",
        span_id="later-correction",
    )
    qualifier = EvidenceSpan.from_document(
        document,
        "The updated\nguidance applies only when temperatures exceed the posted threshold.",
        span_id="policy-scope",
    )
    counterexample = EvidenceSpan.from_document(
        document,
        "A participant still described the route as comfortable on a cool morning.",
        span_id="personal-experience",
    )
    ledger = EvidenceLedger(
        corpus_fingerprint=corpus_fingerprint([document]),
        query="Is the route always safe?",
        answer="No. A later inspection introduced a heat-specific closure.",
        claims=(
            AuditedClaim(
                claim_id="original",
                statement="The route was initially reported as safe in all weather.",
                status="contradicted",
                evidence_span_ids=(early.span_id, correction.span_id),
            ),
            AuditedClaim(
                claim_id="current-policy",
                statement="The later closure applies during extreme heat.",
                status="qualified",
                evidence_span_ids=(correction.span_id, qualifier.span_id),
                qualifiers=("Only above the posted temperature threshold.",),
            ),
            AuditedClaim(
                claim_id="personal-report",
                statement="One participant found the route comfortable in cool weather.",
                status="supported",
                evidence_span_ids=(counterexample.span_id,),
                attribution="A participant",
            ),
        ),
        evidence_spans=(early, correction, qualifier, counterexample),
        relations=(
            ClaimRelation(
                source_claim_id="current-policy",
                target_claim_id="original",
                relation="updates",
                rationale="The later inspection replaces the unconditional early bulletin.",
            ),
            ClaimRelation(
                source_claim_id="personal-report",
                target_claim_id="current-policy",
                relation="qualifies",
                rationale="A cool-weather experience does not overturn the heat-specific policy.",
            ),
        ),
    )

    assert ledger.validate_sources({document.document_id: document}) == ()
    assert ledger.claims[1].qualifiers == ("Only above the posted temperature threshold.",)
    assert ledger.relations[0].relation == "updates"


def test_source_validation_detects_changed_document_snapshot() -> None:
    """Mark frozen evidence stale when the underlying document hash changes."""
    document = load_corpus([FIXTURE_PATH]).documents[0]
    span = EvidenceSpan.from_document(document, "the route was closed during extreme heat")
    ledger = EvidenceLedger(
        corpus_fingerprint=corpus_fingerprint([document]),
        query="What changed?",
        answer="The route was closed during extreme heat.",
        claims=(
            AuditedClaim(
                claim_id="closure",
                statement="The route was closed during extreme heat.",
                status="supported",
                evidence_span_ids=(span.span_id,),
            ),
        ),
        evidence_spans=(span,),
    )
    changed = document.model_copy(update={"content_hash": "changed"})

    issues = ledger.validate_sources({document.document_id: changed})

    assert len(issues) == 1
    assert issues[0].code == "document_changed"


def test_resolved_claim_cannot_reference_missing_evidence() -> None:
    """Reject a fluent support label that has no inspectable source span."""
    with pytest.raises(ValidationError, match="must cite at least one evidence span"):
        AuditedClaim(
            claim_id="unsupported-label",
            statement="A claim with no actual support.",
            status="supported",
        )
