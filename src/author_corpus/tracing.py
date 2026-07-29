"""Durable, source-aware traces for grounded corpus answers."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from author_corpus.answering import AnswerStatus, GenerationAttempt, GroundedAnswer
from author_corpus.audit import EvidenceSpan, EvidenceValidationIssue, validate_evidence_spans
from author_corpus.local_llm import LocalModelSettings, ReasoningEffort
from author_corpus.models import CorpusDocument
from author_corpus.reasoning_models import ReasoningTrace
from author_corpus.retrieval import RetrievalContribution, RetrievedPassage, ScoreKind
from author_corpus.scope import AuthorScope


class RetrievalTraceSettings(BaseModel):
    """Retrieval settings that materially affect an answer's evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_limit: int = Field(ge=1)
    max_passages_per_document: int = Field(ge=1)
    minimum_document_author_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    strategy: str = "dense"
    score_kind: ScoreKind = "unknown"


class GenerationTraceSettings(BaseModel):
    """Generation settings needed to interpret or reproduce an answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    prompt_version: str
    temperature: float
    max_tokens: int = Field(ge=1)
    reasoning_effort: ReasoningEffort

    @classmethod
    def from_local_settings(
        cls,
        settings: LocalModelSettings,
        *,
        prompt_version: str,
    ) -> GenerationTraceSettings:
        """Copy non-secret local generation settings into a trace."""
        return cls(
            model_id=settings.model_id,
            prompt_version=prompt_version,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            reasoning_effort=settings.reasoning_effort,
        )


class TracedEvidence(BaseModel):
    """One retrieved passage frozen as it appeared when an answer was made."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_number: int = Field(ge=1)
    rank: int = Field(ge=1)
    score: float | None = None
    score_kind: ScoreKind = "unknown"
    retrieval_contributions: tuple[RetrievalContribution, ...] = ()
    document_id: str
    document_content_hash: str | None = None
    title: str
    authors: tuple[str, ...] = ()
    source_uris: tuple[str, ...] = ()
    canonical_source_uri: str | None = None
    section_path: tuple[str, ...] = ()
    passage_voice: str = "unknown"
    document_author_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    quoted_speech_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertain_voice_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    attributed_speakers: tuple[str, ...] = ()
    passage_text: str
    passage_hash: str
    evidence_span_id: str | None = None

    @classmethod
    def from_passage(
        cls,
        passage: RetrievedPassage,
        *,
        evidence_number: int,
        document_content_hash: str | None,
    ) -> TracedEvidence:
        """Freeze a retrieved passage and its current source content hash."""
        return cls(
            evidence_number=evidence_number,
            rank=passage.rank,
            score=passage.score,
            score_kind=passage.score_kind,
            retrieval_contributions=passage.retrieval_contributions,
            document_id=passage.document_id,
            document_content_hash=(
                document_content_hash
                or (passage.evidence_span.document_content_hash if passage.evidence_span is not None else None)
            ),
            title=passage.title,
            authors=passage.authors,
            source_uris=passage.source_uris,
            canonical_source_uri=passage.canonical_source_uri,
            section_path=passage.section_path,
            passage_voice=passage.passage_voice,
            document_author_fraction=passage.document_author_fraction,
            quoted_speech_fraction=passage.quoted_speech_fraction,
            uncertain_voice_fraction=passage.uncertain_voice_fraction,
            attributed_speakers=passage.attributed_speakers,
            passage_text=passage.text,
            passage_hash=_text_hash(passage.text),
            evidence_span_id=passage.evidence_span.span_id if passage.evidence_span is not None else None,
        )


class TraceFreshness(BaseModel):
    """Whether the source documents behind a historical trace are unchanged."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_current: bool
    missing_document_ids: tuple[str, ...] = ()
    changed_document_ids: tuple[str, ...] = ()
    unversioned_document_ids: tuple[str, ...] = ()
    evidence_issues: tuple[EvidenceValidationIssue, ...] = ()


class QueryTrace(BaseModel):
    """A reproducible record of one grounded question and answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_id: str
    created_at: datetime
    corpus_fingerprint: str
    query: str
    user_query: str | None = None
    author_scope: AuthorScope = Field(default_factory=AuthorScope)
    answer: str
    answer_status: AnswerStatus = "answered"
    generation_attempts: tuple[GenerationAttempt, ...] = ()
    reasoning: ReasoningTrace | None = None
    cited_evidence_numbers: tuple[int, ...]
    evidence: tuple[TracedEvidence, ...]
    evidence_spans: tuple[EvidenceSpan, ...] = ()
    retrieval: RetrievalTraceSettings
    generation: GenerationTraceSettings
    elapsed_seconds: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_citations(self) -> Self:
        """Require citations and optional source-span references to resolve."""
        ordinals = tuple(attempt.ordinal for attempt in self.generation_attempts)
        if ordinals != tuple(range(1, len(self.generation_attempts) + 1)):
            raise ValueError("Query-trace generation attempts must use contiguous one-based ordinals.")
        if self.answer_status == "citation_failure":
            if len(self.generation_attempts) != 2:
                raise ValueError("A citation-failure trace requires both bounded generation attempts.")
            if self.cited_evidence_numbers or any(attempt.valid_citation_numbers for attempt in self.generation_attempts):
                raise ValueError("A citation-failure trace cannot contain valid citations.")
        available = {item.evidence_number for item in self.evidence}
        if len(available) != len(self.evidence):
            raise ValueError("Query trace contains duplicate evidence numbers.")
        if len(set(self.cited_evidence_numbers)) != len(self.cited_evidence_numbers):
            raise ValueError("Query trace contains duplicate citation numbers.")
        unavailable = set(self.cited_evidence_numbers) - available
        if unavailable:
            raise ValueError(f"Citations do not resolve to evidence: {sorted(unavailable)}.")
        spans_by_id = {span.span_id: span for span in self.evidence_spans}
        if len(spans_by_id) != len(self.evidence_spans):
            raise ValueError("Query trace contains duplicate evidence span IDs.")
        unresolved = {
            item.evidence_span_id
            for item in self.evidence
            if item.evidence_span_id is not None and item.evidence_span_id not in spans_by_id
        }
        if unresolved:
            raise ValueError(f"Traced evidence references unknown source spans: {sorted(unresolved)}.")
        for item in self.evidence:
            if item.evidence_span_id is None:
                continue
            span = spans_by_id[item.evidence_span_id]
            if item.document_id != span.document_id:
                raise ValueError("Traced evidence and its exact source span identify different documents.")
            if item.passage_text != span.text or item.passage_hash != span.text_hash:
                raise ValueError("Traced passage text does not match its exact source span.")
            if item.document_content_hash is not None and item.document_content_hash != span.document_content_hash:
                raise ValueError("Traced evidence and its exact source span identify different document versions.")
        return self

    @property
    def cited_evidence_spans(self) -> tuple[EvidenceSpan, ...]:
        """Return exact source spans reached by the answer's citation markers."""
        cited_numbers = set(self.cited_evidence_numbers)
        span_ids = {
            item.evidence_span_id
            for item in self.evidence
            if item.evidence_number in cited_numbers and item.evidence_span_id is not None
        }
        return tuple(span for span in self.evidence_spans if span.span_id in span_ids)

    @property
    def cited_span_coverage(self) -> tuple[int, int]:
        """Return exact-span coverage as ``(covered citations, total citations)``."""
        evidence_by_number = {item.evidence_number: item for item in self.evidence}
        covered = sum(
            evidence_by_number[number].evidence_span_id is not None
            for number in self.cited_evidence_numbers
            if number in evidence_by_number
        )
        return covered, len(self.cited_evidence_numbers)

    @classmethod
    def from_grounded_answer(
        cls,
        answer: GroundedAnswer,
        *,
        corpus_fingerprint: str,
        document_content_hashes: Mapping[str, str],
        retrieval: RetrievalTraceSettings,
        generation: GenerationTraceSettings,
        elapsed_seconds: float,
        user_query: str | None = None,
        author_scope: AuthorScope | None = None,
        reasoning: ReasoningTrace | None = None,
    ) -> QueryTrace:
        """Create a trace from an answer and the corpus snapshot used for it."""
        if generation.model_id != answer.model_id:
            raise ValueError("Generation settings and answer model IDs differ.")
        if generation.prompt_version != answer.prompt_version:
            raise ValueError("Generation settings and answer prompt versions differ.")
        evidence = tuple(
            TracedEvidence.from_passage(
                passage,
                evidence_number=number,
                document_content_hash=document_content_hashes.get(passage.document_id),
            )
            for number, passage in enumerate(answer.evidence, start=1)
        )
        spans_by_id = {
            passage.evidence_span.span_id: passage.evidence_span
            for passage in answer.evidence
            if passage.evidence_span is not None
        }
        return cls(
            trace_id=uuid4().hex,
            created_at=datetime.now(UTC),
            corpus_fingerprint=corpus_fingerprint,
            query=answer.query,
            user_query=user_query,
            author_scope=author_scope or AuthorScope(),
            answer=answer.answer,
            answer_status=answer.status,
            generation_attempts=answer.generation_attempts,
            reasoning=reasoning,
            cited_evidence_numbers=answer.cited_evidence_numbers,
            evidence=evidence,
            evidence_spans=tuple(spans_by_id.values()),
            retrieval=retrieval,
            generation=generation,
            elapsed_seconds=elapsed_seconds,
        )

    def check_freshness(self, documents: Mapping[str, CorpusDocument]) -> TraceFreshness:
        """Validate exact source ranges against a current corpus snapshot."""
        missing: list[str] = []
        changed: list[str] = []
        unversioned: list[str] = []
        for item in self.evidence:
            document = documents.get(item.document_id)
            if document is None:
                missing.append(item.document_id)
            elif item.evidence_span_id is None:
                unversioned.append(item.document_id)
                if item.document_content_hash is not None and document.content_hash != item.document_content_hash:
                    changed.append(item.document_id)
            elif item.document_content_hash is not None and document.content_hash != item.document_content_hash:
                changed.append(item.document_id)
        evidence_issues = validate_evidence_spans(self.evidence_spans, documents)
        missing.extend(issue.document_id for issue in evidence_issues if issue.code == "document_missing")
        changed.extend(issue.document_id for issue in evidence_issues if issue.code == "document_changed")
        return TraceFreshness(
            is_current=not missing and not changed and not unversioned and not evidence_issues,
            missing_document_ids=tuple(sorted(set(missing))),
            changed_document_ids=tuple(sorted(set(changed))),
            unversioned_document_ids=tuple(sorted(set(unversioned))),
            evidence_issues=evidence_issues,
        )


class QueryTraceStore:
    """Persist complete query traces in a small SQLite audit database."""

    def __init__(self, path: str | Path) -> None:
        """Initialize the store at an ignored local path."""
        self.path = Path(path)

    def put(self, trace: QueryTrace) -> None:
        """Insert or replace one trace without discarding its evidence text."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            _create_schema(connection)
            connection.execute(
                """
                INSERT INTO query_traces (
                    trace_id,
                    created_at,
                    corpus_fingerprint,
                    query,
                    record_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (trace_id) DO UPDATE SET
                    created_at = excluded.created_at,
                    corpus_fingerprint = excluded.corpus_fingerprint,
                    query = excluded.query,
                    record_json = excluded.record_json
                """,
                (
                    trace.trace_id,
                    trace.created_at.isoformat(),
                    trace.corpus_fingerprint,
                    trace.query,
                    trace.model_dump_json(),
                ),
            )

    def get(self, trace_id: str) -> QueryTrace | None:
        """Return one trace by ID, or ``None`` when it is absent."""
        if not self.path.exists():
            return None
        with self._connect() as connection:
            _create_schema(connection)
            row = connection.execute(
                "SELECT record_json FROM query_traces WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
        if row is None:
            return None
        return QueryTrace.model_validate_json(str(row["record_json"]))

    def recent(self, *, limit: int = 20) -> tuple[QueryTrace, ...]:
        """Return the most recent traces in reverse chronological order."""
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        if not self.path.exists():
            return ()
        with self._connect() as connection:
            _create_schema(connection)
            rows = connection.execute(
                """
                SELECT record_json
                FROM query_traces
                ORDER BY created_at DESC, trace_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return tuple(QueryTrace.model_validate_json(str(row["record_json"])) for row in rows)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS query_traces (
            trace_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            corpus_fingerprint TEXT NOT NULL,
            query TEXT NOT NULL,
            record_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS query_traces_created_at
        ON query_traces (created_at DESC);

        CREATE INDEX IF NOT EXISTS query_traces_corpus_fingerprint
        ON query_traces (corpus_fingerprint);
        """
    )
