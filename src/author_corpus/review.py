"""Explicit human review between heuristic claim suggestions and audited truth."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from author_corpus.audit import AuditedClaim, EvidenceLedger, EvidenceSpan
from author_corpus.claim_classification import ClaimEvidenceDecision, ClaimEvidenceLabel

ClaimReviewAction = Literal["accept", "revise", "reject"]
ResolvedClaimStatus = Literal["supported", "qualified", "contradicted", "unsupported"]

_SUGGESTED_STATUSES: dict[ClaimEvidenceLabel, ResolvedClaimStatus] = {
    "supports": "supported",
    "contradicts": "contradicted",
    "insufficient": "unsupported",
}


class ClaimReviewProposal(BaseModel):
    """A classifier suggestion awaiting an explicit human decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    proposal_id: str = Field(min_length=1)
    created_at: datetime
    corpus_fingerprint: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    classifier_decision: ClaimEvidenceDecision
    evidence_spans: tuple[EvidenceSpan, ...] = Field(min_length=1)
    suggested_status: ResolvedClaimStatus | None = None

    @field_validator("created_at")
    @classmethod
    def require_timezone_aware_created_at(cls, value: datetime) -> datetime:
        """Reject proposal timestamps whose absolute time is ambiguous."""
        return _timezone_aware(value, name="created_at")

    @model_validator(mode="after")
    def validate_classifier_provenance(self) -> Self:
        """Require the proposal to preserve the classified claim and evidence."""
        if self.classifier_decision.claim != self.statement:
            raise ValueError("The proposal statement must match the classified claim.")
        span_ids = [span.span_id for span in self.evidence_spans]
        if len(set(span_ids)) != len(span_ids):
            raise ValueError("A review proposal cannot contain duplicate evidence span IDs.")
        if not any(
            _required_text(span.text, name="Evidence") == self.classifier_decision.evidence for span in self.evidence_spans
        ):
            raise ValueError("At least one exact evidence span must match the classified evidence.")
        expected_status = _SUGGESTED_STATUSES.get(self.classifier_decision.label)
        if self.suggested_status != expected_status:
            raise ValueError("The suggested status must use the conservative classifier-label mapping.")
        return self

    @classmethod
    def from_decision(
        cls,
        decision: ClaimEvidenceDecision,
        *,
        corpus_fingerprint: str,
        claim_id: str,
        evidence_spans: tuple[EvidenceSpan, ...],
        proposal_id: str | None = None,
    ) -> ClaimReviewProposal:
        """Create a pending proposal without creating an audited claim."""
        resolved_proposal_id = proposal_id or _proposal_id(
            decision,
            corpus_fingerprint=corpus_fingerprint,
            claim_id=claim_id,
            evidence_spans=evidence_spans,
        )
        return cls(
            proposal_id=resolved_proposal_id,
            created_at=datetime.now(UTC),
            corpus_fingerprint=corpus_fingerprint,
            claim_id=claim_id,
            statement=decision.claim,
            classifier_decision=decision,
            evidence_spans=evidence_spans,
            suggested_status=_SUGGESTED_STATUSES.get(decision.label),
        )


class ClaimReviewRecord(BaseModel):
    """A durable human decision and its optional approved audited claim."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    review_id: str = Field(min_length=1)
    proposal: ClaimReviewProposal
    action: ClaimReviewAction
    reviewer: str = Field(min_length=1)
    reviewed_at: datetime
    audited_claim: AuditedClaim | None = None
    notes: str | None = None

    @field_validator("reviewer")
    @classmethod
    def normalize_reviewer(cls, value: str) -> str:
        """Require a non-empty reviewer identity in direct model construction."""
        return _required_text(value, name="Reviewer")

    @field_validator("reviewed_at")
    @classmethod
    def require_timezone_aware_reviewed_at(cls, value: datetime) -> datetime:
        """Reject review timestamps whose absolute time is ambiguous."""
        return _timezone_aware(value, name="reviewed_at")

    @model_validator(mode="after")
    def validate_human_decision(self) -> Self:
        """Keep rejected proposals unpromoted and approved claims source-bound."""
        if self.action == "reject":
            if self.audited_claim is not None:
                raise ValueError("A rejected proposal cannot contain an audited claim.")
            return self
        if self.audited_claim is None:
            raise ValueError("Accepted and revised proposals must contain an audited claim.")
        if self.audited_claim.status == "pending":
            raise ValueError("A reviewed claim must have a resolved status.")
        if not self.audited_claim.evidence_span_ids:
            raise ValueError("A reviewed claim must retain at least one exact evidence span.")
        if self.audited_claim.status == "qualified" and not self.audited_claim.qualifiers:
            raise ValueError("A qualified reviewed claim must state at least one explicit qualifier.")
        if self.audited_claim.claim_id != self.proposal.claim_id:
            raise ValueError("The reviewed claim ID must match its proposal.")
        available_span_ids = {span.span_id for span in self.proposal.evidence_spans}
        unknown_span_ids = set(self.audited_claim.evidence_span_ids) - available_span_ids
        if unknown_span_ids:
            raise ValueError(f"The reviewed claim references evidence outside its proposal: {sorted(unknown_span_ids)}.")
        if self.action == "accept":
            if self.proposal.suggested_status is None:
                raise ValueError("This classifier label has no status that can be accepted without revision.")
            expected_span_ids = tuple(span.span_id for span in self.proposal.evidence_spans)
            if (
                self.audited_claim.statement != self.proposal.statement
                or self.audited_claim.status != self.proposal.suggested_status
                or self.audited_claim.evidence_span_ids != expected_span_ids
            ):
                raise ValueError("Accept must preserve the proposal; use revise to change it.")
        return self


def review_claim_proposal(
    proposal: ClaimReviewProposal,
    *,
    action: ClaimReviewAction,
    reviewer: str,
    revised_claim: AuditedClaim | None = None,
    notes: str | None = None,
) -> ClaimReviewRecord:
    """Record one explicit reviewer action without mutating an evidence ledger."""
    normalized_reviewer = _required_text(reviewer, name="Reviewer")
    normalized_notes = _optional_text(notes)
    if action == "reject":
        if revised_claim is not None:
            raise ValueError("Reject does not accept a revised claim.")
        audited_claim = None
    elif action == "accept":
        if revised_claim is not None:
            raise ValueError("Accept does not accept a revised claim; use revise.")
        if proposal.suggested_status is None:
            raise ValueError("This classifier label requires revision before it can become an audited claim.")
        audited_claim = AuditedClaim(
            claim_id=proposal.claim_id,
            statement=proposal.statement,
            status=proposal.suggested_status,
            evidence_span_ids=tuple(span.span_id for span in proposal.evidence_spans),
            notes=normalized_notes,
        )
    else:
        if revised_claim is None:
            raise ValueError("Revise requires an explicit revised claim.")
        audited_claim = revised_claim
    return ClaimReviewRecord(
        review_id=uuid4().hex,
        proposal=proposal,
        action=action,
        reviewer=normalized_reviewer,
        reviewed_at=datetime.now(UTC),
        audited_claim=audited_claim,
        notes=normalized_notes,
    )


def apply_claim_review(ledger: EvidenceLedger, review: ClaimReviewRecord) -> EvidenceLedger:
    """Return a new ledger containing one explicitly approved review."""
    claim = review.audited_claim
    if claim is None:
        raise ValueError("A rejected review cannot be applied to an evidence ledger.")
    if review.proposal.corpus_fingerprint != ledger.corpus_fingerprint:
        raise ValueError("The review proposal and evidence ledger use different corpus fingerprints.")

    spans_by_id = {span.span_id: span for span in ledger.evidence_spans}
    for span in review.proposal.evidence_spans:
        existing = spans_by_id.get(span.span_id)
        if existing is not None and existing != span:
            raise ValueError(f"Evidence span ID {span.span_id!r} resolves to conflicting content.")
        spans_by_id[span.span_id] = span

    claims = list(ledger.claims)
    matching_indexes = [index for index, existing in enumerate(claims) if existing.claim_id == claim.claim_id]
    if matching_indexes:
        claims[matching_indexes[0]] = claim
    else:
        claims.append(claim)
    return EvidenceLedger(
        corpus_fingerprint=ledger.corpus_fingerprint,
        query=ledger.query,
        answer=ledger.answer,
        claims=tuple(claims),
        evidence_spans=tuple(spans_by_id.values()),
        relations=ledger.relations,
    )


class ClaimReviewStore:
    """Persist human claim-review records in a local SQLite audit log."""

    def __init__(self, path: str | Path) -> None:
        """Initialize the review store at a privacy-controlled local path."""
        self.path = Path(path)

    def put(self, review: ClaimReviewRecord) -> None:
        """Append one review, allowing only an identical idempotent retry."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record_json = review.model_dump_json()
        with self._connect() as connection:
            _create_schema(connection)
            existing = connection.execute(
                "SELECT record_json FROM claim_reviews WHERE review_id = ?",
                (review.review_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["record_json"]) != record_json:
                    raise ValueError(f"Claim review {review.review_id!r} is immutable and already contains different data.")
                return
            connection.execute(
                """
                INSERT INTO claim_reviews (
                    review_id,
                    proposal_id,
                    reviewed_at,
                    action,
                    record_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    review.review_id,
                    review.proposal.proposal_id,
                    review.reviewed_at.isoformat(),
                    review.action,
                    record_json,
                ),
            )

    def get(self, review_id: str) -> ClaimReviewRecord | None:
        """Return one review by ID, or ``None`` when it is absent."""
        if not self.path.exists():
            return None
        with self._connect() as connection:
            _create_schema(connection)
            row = connection.execute(
                "SELECT record_json FROM claim_reviews WHERE review_id = ?",
                (review_id,),
            ).fetchone()
        if row is None:
            return None
        return ClaimReviewRecord.model_validate_json(str(row["record_json"]))

    def recent(
        self,
        *,
        limit: int = 20,
        corpus_fingerprint: str | None = None,
    ) -> tuple[ClaimReviewRecord, ...]:
        """Return recent reviews, optionally restricted to one corpus snapshot."""
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        if not self.path.exists():
            return ()
        with self._connect() as connection:
            _create_schema(connection)
            rows = connection.execute(
                """
                SELECT record_json
                FROM claim_reviews
                ORDER BY reviewed_at DESC, review_id DESC
                """
            ).fetchall()
        reviews = tuple(ClaimReviewRecord.model_validate_json(str(row["record_json"])) for row in rows)
        if corpus_fingerprint is not None:
            reviews = tuple(review for review in reviews if review.proposal.corpus_fingerprint == corpus_fingerprint)
        return reviews[:limit]

    def for_proposal(self, proposal_id: str) -> tuple[ClaimReviewRecord, ...]:
        """Return all durable reviews recorded for one proposal."""
        if not self.path.exists():
            return ()
        with self._connect() as connection:
            _create_schema(connection)
            rows = connection.execute(
                """
                SELECT record_json
                FROM claim_reviews
                WHERE proposal_id = ?
                ORDER BY reviewed_at DESC, review_id DESC
                """,
                (proposal_id,),
            ).fetchall()
        return tuple(ClaimReviewRecord.model_validate_json(str(row["record_json"])) for row in rows)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection


@dataclass(frozen=True, slots=True)
class ClaimReviewWorkspace:
    """Source-bound review proposals plus their append-only local audit store."""

    proposals: tuple[ClaimReviewProposal, ...]
    store: ClaimReviewStore

    def __post_init__(self) -> None:
        """Reject ambiguous proposal and claim identifiers."""
        proposal_ids = [proposal.proposal_id for proposal in self.proposals]
        claim_ids = [proposal.claim_id for proposal in self.proposals]
        if len(set(proposal_ids)) != len(proposal_ids):
            raise ValueError("A review workspace cannot contain duplicate proposal IDs.")
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("A review workspace cannot contain duplicate claim IDs.")
        if len({proposal.corpus_fingerprint for proposal in self.proposals}) > 1:
            raise ValueError("A review workspace cannot mix corpus fingerprints.")

    @property
    def corpus_fingerprint(self) -> str | None:
        """Return the queue's single corpus fingerprint, if it has proposals."""
        return self.proposals[0].corpus_fingerprint if self.proposals else None

    def get(self, proposal_id: str) -> ClaimReviewProposal:
        """Return one proposal or fail without silently selecting another."""
        for proposal in self.proposals:
            if proposal.proposal_id == proposal_id:
                return proposal
        raise ValueError(f"Unknown claim-review proposal: {proposal_id!r}.")

    def reviews_for(self, proposal_id: str) -> tuple[ClaimReviewRecord, ...]:
        """Return durable decisions for one known proposal."""
        self.get(proposal_id)
        return self.store.for_proposal(proposal_id)

    def record(
        self,
        proposal_id: str,
        *,
        action: ClaimReviewAction,
        reviewer: str,
        revised_claim: AuditedClaim | None = None,
        notes: str | None = None,
    ) -> ClaimReviewRecord:
        """Persist the first explicit human decision for one proposal."""
        proposal = self.get(proposal_id)
        if self.store.for_proposal(proposal_id):
            raise ValueError("This proposal already has a durable human review.")
        review = review_claim_proposal(
            proposal,
            action=action,
            reviewer=reviewer,
            revised_claim=revised_claim,
            notes=notes,
        )
        self.store.put(review)
        return review


def _required_text(value: str, *, name: str) -> str:
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.strip().split())
    return normalized or None


def _proposal_id(
    decision: ClaimEvidenceDecision,
    *,
    corpus_fingerprint: str,
    claim_id: str,
    evidence_spans: tuple[EvidenceSpan, ...],
) -> str:
    payload = "\0".join(
        (
            corpus_fingerprint,
            claim_id,
            decision.model_dump_json(),
            *(span.span_id for span in evidence_spans),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _timezone_aware(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone.")
    return value


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS claim_reviews (
            review_id TEXT PRIMARY KEY,
            proposal_id TEXT NOT NULL,
            reviewed_at TEXT NOT NULL,
            action TEXT NOT NULL,
            record_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS claim_reviews_reviewed_at
        ON claim_reviews (reviewed_at DESC);

        CREATE INDEX IF NOT EXISTS claim_reviews_proposal_id
        ON claim_reviews (proposal_id);
        """
    )
