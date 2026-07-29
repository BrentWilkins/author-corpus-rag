"""Deterministic, read-only extraction of citation-bound generated-answer claims."""

from __future__ import annotations

import hashlib
import re
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from author_corpus.answering import GroundedAnswer
from author_corpus.audit import EvidenceSpan
from author_corpus.claim_classification import ClaimEvidenceDecision, EvidenceVoice, classify_claim_evidence
from author_corpus.tracing import QueryTrace, TracedEvidence

_CITATION_PATTERN = re.compile(r"\[(\d+)]")
_INLINE_CITATION_PREPOSITION_PATTERN = re.compile(
    r"\s+\b(?:from|in|see)\s+(?:\[\d+])+(?=\s*[,.;:)])",
    re.IGNORECASE,
)
_LIST_PREFIX_PATTERN = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
_LINK_PATTERN = re.compile(r"\[([^\]]+)]\([^)]+\)")
_MARKDOWN_MARKER_PATTERN = re.compile(r"[*_`~]")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_PUNCTUATION_SPACING_PATTERN = re.compile(r"\s+([,.;:!?])")
_ABBREVIATIONS = frozenset({"dr", "e.g", "i.e", "mr", "mrs", "ms", "prof", "st", "u.s", "vs"})


class ClaimEvidenceAssessment(BaseModel):
    """One answer citation resolved to optional exact evidence and a heuristic role."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_number: int = Field(ge=1)
    title: str | None = None
    source_uris: tuple[str, ...] = ()
    evidence_span: EvidenceSpan | None = None
    classifier_decision: ClaimEvidenceDecision | None = None

    @model_validator(mode="after")
    def require_decision_only_with_exact_evidence(self) -> Self:
        """Keep heuristic decisions inseparable from the exact text they assessed."""
        if (self.evidence_span is None) != (self.classifier_decision is None):
            raise ValueError("Exact evidence and its claim-classifier decision must occur together.")
        if (
            self.evidence_span is not None
            and self.classifier_decision is not None
            and _normalize(self.evidence_span.text) != self.classifier_decision.evidence
        ):
            raise ValueError("The classifier decision does not assess the recorded exact evidence span.")
        return self


class AnswerClaimCandidate(BaseModel):
    """One generated sentence retaining every citation and source-resolution result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    ordinal: int = Field(ge=1)
    statement: str = Field(min_length=1)
    citation_numbers: tuple[int, ...] = Field(min_length=1)
    evidence: tuple[ClaimEvidenceAssessment, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_one_assessment_per_citation(self) -> Self:
        """Preserve citation order without silently dropping unresolved numbers."""
        if tuple(item.evidence_number for item in self.evidence) != self.citation_numbers:
            raise ValueError("Evidence assessments must match citation numbers in order.")
        if len(set(self.citation_numbers)) != len(self.citation_numbers):
            raise ValueError("Claim citations must be unique.")
        return self

    @property
    def is_source_bound(self) -> bool:
        """Return whether every citation resolves to an exact versioned span."""
        return all(item.evidence_span is not None for item in self.evidence)

    @property
    def missing_exact_span_numbers(self) -> tuple[int, ...]:
        """Return citations that cannot support a source-bound review proposal."""
        return tuple(item.evidence_number for item in self.evidence if item.evidence_span is None)


class AnswerClaimExtraction(BaseModel):
    """All cited candidates and uncited prose found in one generated answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    candidates: tuple[AnswerClaimCandidate, ...] = ()
    uncited_segments: tuple[str, ...] = ()

    @property
    def exact_span_coverage(self) -> tuple[int, int]:
        """Return source-bound candidate coverage as ``(covered, total)``."""
        return sum(candidate.is_source_bound for candidate in self.candidates), len(self.candidates)


def extract_answer_claims(trace: QueryTrace) -> AnswerClaimExtraction:
    """Extract cited answer sentences without creating proposals or audit records."""
    evidence_by_number = {item.evidence_number: item for item in trace.evidence}
    spans_by_id = {span.span_id: span for span in trace.evidence_spans}
    return _extract_claims(
        trace.answer,
        trace_id=trace.trace_id,
        query=trace.user_query or trace.query,
        evidence_by_number=evidence_by_number,
        spans_by_id=spans_by_id,
    )


def extract_grounded_answer_claims(
    answer: GroundedAnswer,
    *,
    extraction_id: str = "unpersisted",
) -> AnswerClaimExtraction:
    """Extract claims directly from a grounded draft before trace persistence."""
    evidence = tuple(
        TracedEvidence.from_passage(
            passage,
            evidence_number=number,
            document_content_hash=(passage.evidence_span.document_content_hash if passage.evidence_span is not None else None),
        )
        for number, passage in enumerate(answer.evidence, start=1)
    )
    spans = tuple(passage.evidence_span for passage in answer.evidence if passage.evidence_span is not None)
    return _extract_claims(
        answer.answer,
        trace_id=extraction_id,
        query=answer.query,
        evidence_by_number={item.evidence_number: item for item in evidence},
        spans_by_id={span.span_id: span for span in spans},
    )


def _extract_claims(
    answer: str,
    *,
    trace_id: str,
    query: str,
    evidence_by_number: dict[int, TracedEvidence],
    spans_by_id: dict[str, EvidenceSpan],
) -> AnswerClaimExtraction:
    candidates: list[AnswerClaimCandidate] = []
    uncited: list[str] = []
    ordinal = 0
    for segment in _answer_segments(answer):
        citations = _citation_numbers(segment)
        statement = _clean_statement(segment)
        if not statement:
            continue
        if not citations:
            uncited.append(statement)
            continue
        ordinal += 1
        assessments = tuple(
            _assess_citation(
                statement,
                evidence_number,
                evidence_by_number=evidence_by_number,
                spans_by_id=spans_by_id,
            )
            for evidence_number in citations
        )
        candidates.append(
            AnswerClaimCandidate(
                candidate_id=_candidate_id(trace_id, ordinal, statement, assessments),
                trace_id=trace_id,
                ordinal=ordinal,
                statement=statement,
                citation_numbers=citations,
                evidence=assessments,
            )
        )
    return AnswerClaimExtraction(
        trace_id=trace_id,
        query=query,
        candidates=tuple(candidates),
        uncited_segments=tuple(uncited),
    )


def _answer_segments(answer: str) -> tuple[str, ...]:
    segments: list[str] = []
    for raw_line in answer.splitlines():
        line = _LIST_PREFIX_PATTERN.sub("", raw_line.strip())
        if not line or line.startswith("#") or set(line) <= {"-", "*", "_"}:
            continue
        segments.extend(_split_sentences(line))
    return tuple(segment for segment in segments if segment.strip())


def _split_sentences(text: str) -> tuple[str, ...]:
    sentences: list[str] = []
    start = 0
    for index, character in enumerate(text):
        if character not in ".!?":
            continue
        following = text[index + 1 :]
        whitespace = len(following) - len(following.lstrip())
        if whitespace == 0:
            continue
        next_index = index + 1 + whitespace
        if next_index >= len(text):
            continue
        next_character = text[next_index]
        if next_character not in '"“ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789*_`':
            continue
        token = text[start:index].rstrip().split()[-1].casefold().rstrip(".") if text[start:index].strip() else ""
        if token in _ABBREVIATIONS:
            continue
        sentences.append(text[start : index + 1].strip())
        start = next_index
    remainder = text[start:].strip()
    if remainder:
        sentences.append(remainder)
    return tuple(sentences)


def _citation_numbers(segment: str) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(match) for match in _CITATION_PATTERN.findall(segment)))


def _clean_statement(segment: str) -> str:
    without_inline_reference = _INLINE_CITATION_PREPOSITION_PATTERN.sub("", segment)
    without_citations = _CITATION_PATTERN.sub("", without_inline_reference)
    without_links = _LINK_PATTERN.sub(r"\1", without_citations)
    without_markers = _MARKDOWN_MARKER_PATTERN.sub("", without_links)
    normalized = _WHITESPACE_PATTERN.sub(" ", without_markers).strip()
    return _PUNCTUATION_SPACING_PATTERN.sub(r"\1", normalized)


def _assess_citation(
    statement: str,
    evidence_number: int,
    *,
    evidence_by_number: dict[int, TracedEvidence],
    spans_by_id: dict[str, EvidenceSpan],
) -> ClaimEvidenceAssessment:
    traced = evidence_by_number.get(evidence_number)
    if traced is None:
        return ClaimEvidenceAssessment(evidence_number=evidence_number)
    span = spans_by_id.get(traced.evidence_span_id) if traced.evidence_span_id is not None else None
    decision = (
        None
        if span is None
        else classify_claim_evidence(
            statement,
            span.text,
            evidence_voice=_evidence_voice(traced.passage_voice),
        )
    )
    return ClaimEvidenceAssessment(
        evidence_number=evidence_number,
        title=traced.title,
        source_uris=traced.source_uris,
        evidence_span=span,
        classifier_decision=decision,
    )


def _evidence_voice(value: str) -> EvidenceVoice:
    if value == "document_author":
        return "document_author"
    if value == "quoted_speech":
        return "quoted_speech"
    if value == "mixed":
        return "mixed"
    if value == "uncertain":
        return "uncertain"
    return "unknown"


def _candidate_id(
    trace_id: str,
    ordinal: int,
    statement: str,
    assessments: tuple[ClaimEvidenceAssessment, ...],
) -> str:
    payload = "\0".join(
        (
            trace_id,
            str(ordinal),
            statement,
            *(
                f"{item.evidence_number}:{item.evidence_span.span_id if item.evidence_span is not None else ''}"
                for item in assessments
            ),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _normalize(value: str) -> str:
    return " ".join(value.strip().split())
