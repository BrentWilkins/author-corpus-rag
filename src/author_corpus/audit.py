"""Exact source spans and contradiction-aware claim-ledger models."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from author_corpus.models import CorpusDocument

EVIDENCE_SPAN_VERSION = "source-spans-v1"

ClaimStatus = Literal["pending", "supported", "qualified", "contradicted", "unsupported"]
ClaimRelationKind = Literal["supports", "qualifies", "contradicts", "updates"]
EvidenceIssueCode = Literal[
    "document_missing",
    "document_changed",
    "span_out_of_bounds",
    "span_text_changed",
    "span_hash_invalid",
]


class EvidenceSpan(BaseModel):
    """An exact character range copied from one versioned source document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    span_id: str
    document_id: str
    document_content_hash: str
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    text: str
    text_hash: str
    source_uris: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_range_and_hash(self) -> Self:
        """Reject invalid ranges and evidence text whose hash is inconsistent."""
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char.")
        if self.end_char - self.start_char != len(self.text):
            raise ValueError("Evidence range length must equal evidence text length.")
        if self.text_hash != _text_hash(self.text):
            raise ValueError("text_hash does not match evidence text.")
        return self

    @classmethod
    def from_document(
        cls,
        document: CorpusDocument,
        text: str,
        *,
        occurrence: int = 1,
        span_id: str | None = None,
    ) -> EvidenceSpan:
        """Locate one exact occurrence and freeze its offsets and hashes."""
        if occurrence < 1:
            raise ValueError("occurrence must be at least 1.")
        start_char = _occurrence_start(document.content, text, occurrence=occurrence)
        resolved_span_id = span_id or _span_id(document.document_id, start_char, text)
        return cls(
            span_id=resolved_span_id,
            document_id=document.document_id,
            document_content_hash=document.content_hash,
            start_char=start_char,
            end_char=start_char + len(text),
            text=text,
            text_hash=_text_hash(text),
            source_uris=tuple(source.uri for source in document.sources),
        )

    @classmethod
    def from_range(
        cls,
        document: CorpusDocument,
        *,
        start_char: int,
        end_char: int,
        span_id: str | None = None,
    ) -> EvidenceSpan:
        """Freeze an already-resolved half-open range in one source version."""
        if start_char < 0:
            raise ValueError("start_char must not be negative.")
        if end_char <= start_char:
            raise ValueError("end_char must be greater than start_char.")
        if end_char > len(document.content):
            raise ValueError("Evidence range extends beyond the source document.")
        text = document.content[start_char:end_char]
        resolved_span_id = span_id or _span_id(document.document_id, start_char, text)
        return cls(
            span_id=resolved_span_id,
            document_id=document.document_id,
            document_content_hash=document.content_hash,
            start_char=start_char,
            end_char=end_char,
            text=text,
            text_hash=_text_hash(text),
            source_uris=tuple(source.uri for source in document.sources),
        )


class AuditedClaim(BaseModel):
    """One answer assertion with explicit support status and qualifiers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_id: str
    statement: str
    status: ClaimStatus = "pending"
    evidence_span_ids: tuple[str, ...] = ()
    attribution: str | None = None
    qualifiers: tuple[str, ...] = ()
    notes: str | None = None

    @model_validator(mode="after")
    def require_evidence_for_resolved_claims(self) -> Self:
        """Require evidence for every resolution except unsupported."""
        if self.status in {"supported", "qualified", "contradicted"} and not self.evidence_span_ids:
            raise ValueError(f"A {self.status} claim must cite at least one evidence span.")
        return self


class ClaimRelation(BaseModel):
    """A directional semantic relationship between two audited claims."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_claim_id: str
    target_claim_id: str
    relation: ClaimRelationKind
    rationale: str

    @model_validator(mode="after")
    def reject_self_relation(self) -> Self:
        """Reject a claim relating to itself."""
        if self.source_claim_id == self.target_claim_id:
            raise ValueError("A claim relation must connect two different claims.")
        return self


class EvidenceValidationIssue(BaseModel):
    """A reason that frozen evidence no longer resolves to current source text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    span_id: str
    document_id: str
    code: EvidenceIssueCode
    message: str


class EvidenceLedger(BaseModel):
    """Claims, exact source spans, and relationships for one answer audit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    corpus_fingerprint: str
    query: str
    answer: str
    claims: tuple[AuditedClaim, ...]
    evidence_spans: tuple[EvidenceSpan, ...]
    relations: tuple[ClaimRelation, ...] = ()

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        """Reject duplicate identifiers and dangling claim or evidence links."""
        claim_ids = [claim.claim_id for claim in self.claims]
        span_ids = [span.span_id for span in self.evidence_spans]
        duplicate_claims = _duplicates(claim_ids)
        duplicate_spans = _duplicates(span_ids)
        if duplicate_claims:
            raise ValueError(f"Duplicate claim IDs: {duplicate_claims}.")
        if duplicate_spans:
            raise ValueError(f"Duplicate evidence span IDs: {duplicate_spans}.")
        known_claims = set(claim_ids)
        known_spans = set(span_ids)
        dangling_spans = {span_id for claim in self.claims for span_id in claim.evidence_span_ids if span_id not in known_spans}
        if dangling_spans:
            raise ValueError(f"Unknown evidence span IDs: {sorted(dangling_spans)}.")
        dangling_claims = {
            claim_id
            for relation in self.relations
            for claim_id in (relation.source_claim_id, relation.target_claim_id)
            if claim_id not in known_claims
        }
        if dangling_claims:
            raise ValueError(f"Unknown related claim IDs: {sorted(dangling_claims)}.")
        return self

    def validate_sources(self, documents: Mapping[str, CorpusDocument]) -> tuple[EvidenceValidationIssue, ...]:
        """Check every frozen span against the current raw corpus snapshot."""
        return validate_evidence_spans(self.evidence_spans, documents)


def validate_evidence_spans(
    spans: Sequence[EvidenceSpan],
    documents: Mapping[str, CorpusDocument],
) -> tuple[EvidenceValidationIssue, ...]:
    """Check frozen spans against the exact text of a current corpus snapshot."""
    issues: list[EvidenceValidationIssue] = []
    for span in spans:
        document = documents.get(span.document_id)
        if document is None:
            issues.append(_evidence_issue(span, "document_missing", "The source document is not in the current corpus."))
            continue
        if document.content_hash != span.document_content_hash:
            issues.append(_evidence_issue(span, "document_changed", "The source document content hash changed."))
            continue
        if span.end_char > len(document.content):
            issues.append(_evidence_issue(span, "span_out_of_bounds", "The source span is outside the current document."))
            continue
        current_text = document.content[span.start_char : span.end_char]
        if current_text != span.text:
            issues.append(_evidence_issue(span, "span_text_changed", "The text at the recorded source range changed."))
        elif _text_hash(current_text) != span.text_hash:
            issues.append(_evidence_issue(span, "span_hash_invalid", "The source span hash is invalid."))
    return tuple(issues)


def _occurrence_start(content: str, text: str, *, occurrence: int) -> int:
    if not text:
        raise ValueError("Evidence text must not be empty.")
    start = -1
    search_from = 0
    for _ in range(occurrence):
        start = content.find(text, search_from)
        if start < 0:
            raise ValueError(f"Evidence text occurrence {occurrence} was not found in the document.")
        search_from = start + 1
    return start


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _span_id(document_id: str, start_char: int, text: str) -> str:
    payload = f"{document_id}\0{start_char}\0{text}".encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def _duplicates(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def _evidence_issue(span: EvidenceSpan, code: EvidenceIssueCode, message: str) -> EvidenceValidationIssue:
    return EvidenceValidationIssue(
        span_id=span.span_id,
        document_id=span.document_id,
        code=code,
        message=message,
    )
