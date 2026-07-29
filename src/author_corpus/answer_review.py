"""Explicit whole-answer review and private behavioral-training export."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from author_corpus.claim_extraction import extract_answer_claims
from author_corpus.models import CorpusDocument
from author_corpus.scope import AuthorScope
from author_corpus.tracing import QueryTrace, QueryTraceStore

AnswerReviewAction = Literal["accept", "revise", "reject"]
AnswerClaimReviewAction = Literal["accept", "revise", "reject"]
_CITATION_PATTERN = re.compile(r"\[(\d+)]")


class AnswerClaimReview(BaseModel):
    """An optional human decision about one extracted answer claim."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(min_length=1)
    action: AnswerClaimReviewAction
    revised_statement: str | None = None

    @model_validator(mode="after")
    def validate_revision(self) -> Self:
        """Require revised text only for an explicit revision."""
        if self.action == "revise" and not _optional_text(self.revised_statement):
            raise ValueError("A revised answer claim requires revised_statement.")
        if self.action != "revise" and self.revised_statement is not None:
            raise ValueError("Only a revised answer claim may contain revised_statement.")
        return self


class AnswerReviewRecord(BaseModel):
    """An append-only human decision about one complete generated answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    review_id: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    corpus_fingerprint: str = Field(min_length=1)
    original_answer_hash: str = Field(min_length=64, max_length=64)
    action: AnswerReviewAction
    reviewer: str = Field(min_length=1)
    reviewed_at: datetime
    reviewed_answer: str | None = None
    claim_reviews: tuple[AnswerClaimReview, ...] = ()
    exact_span_coverage: tuple[int, int]
    author_scope: AuthorScope = Field(default_factory=AuthorScope)
    notes: str | None = None

    @field_validator("reviewer")
    @classmethod
    def normalize_reviewer(cls, value: str) -> str:
        """Require a stable non-empty reviewer identity."""
        return _required_text(value, name="Reviewer")

    @field_validator("reviewed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Reject ambiguous review timestamps."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("reviewed_at must include a timezone.")
        return value

    @model_validator(mode="after")
    def validate_action(self) -> Self:
        """Keep rejected answers out of accepted training material."""
        covered, total = self.exact_span_coverage
        if covered < 0 or total < 0 or covered > total:
            raise ValueError("Invalid exact-span coverage.")
        if self.action == "reject" and self.reviewed_answer is not None:
            raise ValueError("A rejected answer cannot contain reviewed answer text.")
        if self.action != "reject" and not _optional_text(self.reviewed_answer):
            raise ValueError("Accepted and revised reviews require answer text.")
        candidate_ids = [review.candidate_id for review in self.claim_reviews]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("An answer review cannot repeat a candidate decision.")
        return self


class TrainingEvidence(BaseModel):
    """One source passage exported with a reviewed behavioral example."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_number: int = Field(ge=1)
    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    authors: tuple[str, ...] = ()
    source_uris: tuple[str, ...] = ()
    passage_voice: str
    text: str = Field(min_length=1)
    evidence_span_id: str = Field(min_length=1)


class ReviewedTrainingExample(BaseModel):
    """A private evidence-in, reviewed-answer-out behavioral example."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    review_id: str
    trace_id: str
    question: str
    response: str
    author_scope: AuthorScope
    evidence: tuple[TrainingEvidence, ...]


class AnswerReviewStore:
    """Persist immutable whole-answer reviews in SQLite."""

    def __init__(self, path: str | Path) -> None:
        """Initialize the answer-review store."""
        self.path = Path(path)

    def put(self, review: AnswerReviewRecord) -> None:
        """Append one review, allowing only an identical idempotent retry."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = review.model_dump_json()
        with self._connect() as connection:
            _create_schema(connection)
            existing = connection.execute(
                "SELECT record_json FROM answer_reviews WHERE review_id = ?",
                (review.review_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["record_json"]) != record:
                    raise ValueError(f"Answer review {review.review_id!r} is immutable.")
                return
            connection.execute(
                """
                INSERT INTO answer_reviews (
                    review_id,
                    trace_id,
                    corpus_fingerprint,
                    reviewed_at,
                    action,
                    record_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    review.review_id,
                    review.trace_id,
                    review.corpus_fingerprint,
                    review.reviewed_at.isoformat(),
                    review.action,
                    record,
                ),
            )

    def get(self, review_id: str) -> AnswerReviewRecord | None:
        """Return one answer review by ID."""
        if not self.path.exists():
            return None
        with self._connect() as connection:
            _create_schema(connection)
            row = connection.execute(
                "SELECT record_json FROM answer_reviews WHERE review_id = ?",
                (review_id,),
            ).fetchone()
        return None if row is None else AnswerReviewRecord.model_validate_json(str(row["record_json"]))

    def for_trace(self, trace_id: str) -> tuple[AnswerReviewRecord, ...]:
        """Return answer reviews for one trace in reverse chronological order."""
        return tuple(review for review in self.recent(limit=10_000) if review.trace_id == trace_id)

    def recent(
        self,
        *,
        limit: int = 20,
        corpus_fingerprint: str | None = None,
    ) -> tuple[AnswerReviewRecord, ...]:
        """Return recent answer reviews, optionally for one corpus snapshot."""
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        if not self.path.exists():
            return ()
        with self._connect() as connection:
            _create_schema(connection)
            rows = connection.execute(
                """
                SELECT record_json
                FROM answer_reviews
                ORDER BY reviewed_at DESC, review_id DESC
                """
            ).fetchall()
        reviews = tuple(AnswerReviewRecord.model_validate_json(str(row["record_json"])) for row in rows)
        if corpus_fingerprint is not None:
            reviews = tuple(review for review in reviews if review.corpus_fingerprint == corpus_fingerprint)
        return reviews[:limit]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection


def review_trace_answer(
    trace: QueryTrace,
    *,
    documents: Mapping[str, CorpusDocument],
    action: AnswerReviewAction,
    reviewer: str,
    revised_answer: str | None = None,
    claim_reviews: tuple[AnswerClaimReview, ...] = (),
    notes: str | None = None,
) -> AnswerReviewRecord:
    """Create one explicit review after checking trace and source freshness."""
    normalized_reviewer = _required_text(reviewer, name="Reviewer")
    extraction = extract_answer_claims(trace)
    known_candidate_ids = {candidate.candidate_id for candidate in extraction.candidates}
    unknown_candidates = {review.candidate_id for review in claim_reviews} - known_candidate_ids
    if unknown_candidates:
        raise ValueError(f"Answer review references unknown candidates: {sorted(unknown_candidates)}.")
    freshness = trace.check_freshness(documents)
    if action != "reject" and not freshness.is_current:
        raise ValueError("Accepted or revised answers require current exact source evidence.")
    covered, total = trace.cited_span_coverage
    if action != "reject" and covered != total:
        raise ValueError("Accepted or revised answers require exact spans for every citation.")

    if action == "accept":
        if revised_answer is not None:
            raise ValueError("Accept preserves the generated answer; use revise to change it.")
        reviewed_answer = trace.answer
    elif action == "revise":
        reviewed_answer = _required_text(revised_answer or "", name="Revised answer")
        _validate_citations(reviewed_answer, available=len(trace.evidence))
    else:
        if revised_answer is not None:
            raise ValueError("Reject does not accept revised answer text.")
        reviewed_answer = None

    return AnswerReviewRecord(
        review_id=uuid4().hex,
        trace_id=trace.trace_id,
        corpus_fingerprint=trace.corpus_fingerprint,
        original_answer_hash=_text_hash(trace.answer),
        action=action,
        reviewer=normalized_reviewer,
        reviewed_at=datetime.now(UTC),
        reviewed_answer=reviewed_answer,
        claim_reviews=claim_reviews,
        exact_span_coverage=(covered, total),
        author_scope=trace.author_scope,
        notes=_optional_text(notes),
    )


def export_reviewed_training_examples(
    review_store: AnswerReviewStore,
    trace_store: QueryTraceStore,
    output_path: str | Path,
    *,
    corpus_fingerprint: str | None = None,
) -> int:
    """Write accepted and revised evidence-bound examples to private JSONL."""
    examples: list[ReviewedTrainingExample] = []
    reviews = review_store.recent(limit=10_000, corpus_fingerprint=corpus_fingerprint)
    for review in reversed(reviews):
        if review.action == "reject" or review.reviewed_answer is None:
            continue
        trace = trace_store.get(review.trace_id)
        if trace is None:
            raise ValueError(f"Answer review {review.review_id!r} references a missing trace.")
        if _text_hash(trace.answer) != review.original_answer_hash:
            raise ValueError(f"Answer review {review.review_id!r} no longer matches its trace.")
        evidence = tuple(
            TrainingEvidence(
                evidence_number=item.evidence_number,
                document_id=item.document_id,
                title=item.title,
                authors=item.authors,
                source_uris=item.source_uris,
                passage_voice=item.passage_voice,
                text=item.passage_text,
                evidence_span_id=item.evidence_span_id,
            )
            for item in trace.evidence
            if item.evidence_span_id is not None
        )
        examples.append(
            ReviewedTrainingExample(
                review_id=review.review_id,
                trace_id=trace.trace_id,
                question=trace.user_query or trace.query,
                response=review.reviewed_answer,
                author_scope=review.author_scope,
                evidence=evidence,
            )
        )
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(example.model_dump(mode="json"), ensure_ascii=False) + "\n" for example in examples),
        encoding="utf-8",
    )
    return len(examples)


def _validate_citations(answer: str, *, available: int) -> None:
    citations = tuple(int(value) for value in _CITATION_PATTERN.findall(answer))
    if available and not citations:
        raise ValueError("A revised evidence-based answer requires numbered citations.")
    invalid = tuple(number for number in citations if number < 1 or number > available)
    if invalid:
        raise ValueError(f"Revised answer cites unavailable evidence: {sorted(set(invalid))}.")


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


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS answer_reviews (
            review_id TEXT PRIMARY KEY,
            trace_id TEXT NOT NULL,
            corpus_fingerprint TEXT NOT NULL,
            reviewed_at TEXT NOT NULL,
            action TEXT NOT NULL,
            record_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS answer_reviews_trace
        ON answer_reviews (trace_id);

        CREATE INDEX IF NOT EXISTS answer_reviews_fingerprint
        ON answer_reviews (corpus_fingerprint, reviewed_at DESC);
        """
    )
