"""Evidence-bound corpus synthesis over resolved claim ledgers."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from author_corpus.audit import AuditedClaim, ClaimRelation, EvidenceLedger, EvidenceSpan, validate_evidence_spans
from author_corpus.models import CorpusDocument
from author_corpus.scope import AuthorScope

EVIDENCE_SYNTHESIS_VERSION = "evidence-synthesis-v1"


class SynthesisClaimGroup(BaseModel):
    """A topic label and the resolved claims assigned to it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str = Field(min_length=1)
    claim_ids: tuple[str, ...] = Field(min_length=1)


class SynthesisCoverage(BaseModel):
    """Exact document coverage without implying confidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    documents_inspected: int = Field(ge=0)
    documents_contributing: int = Field(ge=0)
    contributing_document_ids: tuple[str, ...] = ()
    unresolved_document_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_counts(self) -> SynthesisCoverage:
        """Require counts to match their inspectable document identifiers."""
        if self.documents_contributing != len(self.contributing_document_ids):
            raise ValueError("Contributing document count must match its identifiers.")
        if self.documents_contributing > self.documents_inspected:
            raise ValueError("Contributing documents cannot exceed inspected documents.")
        if self.documents_inspected != self.documents_contributing + len(self.unresolved_document_ids):
            raise ValueError("Every inspected document must be contributing or unresolved.")
        return self


class EvidenceBoundSynthesis(BaseModel):
    """A deterministic corpus view whose factual claims retain exact evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    synthesis_id: str = Field(min_length=1)
    version: str = EVIDENCE_SYNTHESIS_VERSION
    corpus_fingerprint: str = Field(min_length=1)
    author_scope: AuthorScope = Field(default_factory=AuthorScope)
    claims: tuple[AuditedClaim, ...]
    evidence_spans: tuple[EvidenceSpan, ...]
    relations: tuple[ClaimRelation, ...] = ()
    groups: tuple[SynthesisClaimGroup, ...] = ()
    coverage: SynthesisCoverage

    @model_validator(mode="after")
    def validate_references(self) -> EvidenceBoundSynthesis:
        """Reject pending claims, missing evidence, and dangling group links."""
        claim_ids = {claim.claim_id for claim in self.claims}
        span_ids = {span.span_id for span in self.evidence_spans}
        if len(claim_ids) != len(self.claims):
            raise ValueError("Evidence synthesis cannot contain duplicate claim IDs.")
        if len(span_ids) != len(self.evidence_spans):
            raise ValueError("Evidence synthesis cannot contain duplicate span IDs.")
        for claim in self.claims:
            if claim.status == "pending":
                raise ValueError("Pending claims cannot enter evidence-bound synthesis.")
            if claim.status != "unsupported" and not claim.evidence_span_ids:
                raise ValueError("Resolved synthesis claims must retain exact evidence.")
            if set(claim.evidence_span_ids) - span_ids:
                raise ValueError(f"Synthesis claim {claim.claim_id!r} references unknown evidence.")
        grouped_ids = [claim_id for group in self.groups for claim_id in group.claim_ids]
        if set(grouped_ids) - claim_ids:
            raise ValueError("Synthesis groups reference unknown claims.")
        if len(set(grouped_ids)) != len(grouped_ids):
            raise ValueError("A synthesis claim can appear in only one topic group.")
        return self

    def validate_sources(
        self,
        documents: Mapping[str, CorpusDocument],
    ) -> tuple[str, ...]:
        """Return human-readable source validation failures."""
        return tuple(issue.message for issue in validate_evidence_spans(self.evidence_spans, documents))

    def to_markdown(self) -> str:
        """Render grouped claims with exact source links and coverage."""
        claim_by_id = {claim.claim_id: claim for claim in self.claims}
        spans_by_id = {span.span_id: span for span in self.evidence_spans}
        lines = ["## Evidence-bound corpus synthesis", ""]
        groups = self.groups or (
            SynthesisClaimGroup(topic="Resolved claims", claim_ids=tuple(claim.claim_id for claim in self.claims)),
        )
        citation_number = 0
        for group in groups:
            lines.extend((f"### {group.topic}", ""))
            for claim_id in group.claim_ids:
                claim = claim_by_id[claim_id]
                citations: list[str] = []
                for span_id in claim.evidence_span_ids:
                    citation_number += 1
                    span = spans_by_id[span_id]
                    source = span.source_uris[0] if span.source_uris else f"document:{span.document_id}"
                    citations.append(f"[{citation_number}]({source})")
                qualifier = f" — {'; '.join(claim.qualifiers)}" if claim.qualifiers else ""
                lines.append(f"- **{claim.status}:** {claim.statement}{qualifier} {' '.join(citations)}")
            lines.append("")
        lines.extend(
            (
                "### Coverage",
                "",
                f"- Documents inspected: {self.coverage.documents_inspected}",
                f"- Documents contributing evidence: {self.coverage.documents_contributing}",
                f"- Documents unresolved: {len(self.coverage.unresolved_document_ids)}",
            )
        )
        return "\n".join(lines)


class EvidenceSynthesisStore:
    """Persist immutable evidence-bound syntheses in SQLite."""

    def __init__(self, path: str | Path) -> None:
        """Initialize the synthesis store."""
        self.path = Path(path)

    def put(self, synthesis: EvidenceBoundSynthesis) -> None:
        """Insert one immutable synthesis, allowing identical retries."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = synthesis.model_dump_json()
        with self._connect() as connection:
            _create_schema(connection)
            existing = connection.execute(
                "SELECT record_json FROM evidence_syntheses WHERE synthesis_id = ?",
                (synthesis.synthesis_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["record_json"]) != record:
                    raise ValueError(f"Evidence synthesis {synthesis.synthesis_id!r} is immutable.")
                return
            connection.execute(
                """
                INSERT INTO evidence_syntheses (
                    synthesis_id,
                    corpus_fingerprint,
                    scope_key,
                    record_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    synthesis.synthesis_id,
                    synthesis.corpus_fingerprint,
                    synthesis.author_scope.cache_key,
                    record,
                ),
            )

    def get(self, synthesis_id: str) -> EvidenceBoundSynthesis | None:
        """Return one synthesis by ID."""
        if not self.path.exists():
            return None
        with self._connect() as connection:
            _create_schema(connection)
            row = connection.execute(
                "SELECT record_json FROM evidence_syntheses WHERE synthesis_id = ?",
                (synthesis_id,),
            ).fetchone()
        return None if row is None else EvidenceBoundSynthesis.model_validate_json(str(row["record_json"]))

    def latest(
        self,
        *,
        corpus_fingerprint: str,
        author_scope: AuthorScope,
    ) -> EvidenceBoundSynthesis | None:
        """Return the latest deterministic synthesis for one corpus and scope."""
        if not self.path.exists():
            return None
        with self._connect() as connection:
            _create_schema(connection)
            row = connection.execute(
                """
                SELECT record_json
                FROM evidence_syntheses
                WHERE corpus_fingerprint = ? AND scope_key = ?
                ORDER BY rowid DESC
                LIMIT 1
                """,
                (corpus_fingerprint, author_scope.cache_key),
            ).fetchone()
        return None if row is None else EvidenceBoundSynthesis.model_validate_json(str(row["record_json"]))

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection


def build_evidence_bound_synthesis(
    documents: Sequence[CorpusDocument],
    ledgers: Sequence[EvidenceLedger],
    *,
    corpus_fingerprint: str,
    author_scope: AuthorScope | None = None,
    topics: Mapping[str, str] | None = None,
) -> EvidenceBoundSynthesis:
    """Combine current resolved ledgers without generating or smoothing claims."""
    scope = author_scope or AuthorScope()
    scoped_documents = tuple(document for document in documents if _document_in_scope(document, scope))
    documents_by_id = {document.document_id: document for document in scoped_documents}
    claims_by_id: dict[str, AuditedClaim] = {}
    spans_by_id: dict[str, EvidenceSpan] = {}
    relations_by_key: dict[tuple[str, str, str], ClaimRelation] = {}
    for ledger in ledgers:
        if ledger.corpus_fingerprint != corpus_fingerprint:
            raise ValueError("Every synthesis ledger must match the current corpus fingerprint.")
        ledger_spans = {span.span_id: span for span in ledger.evidence_spans if span.document_id in documents_by_id}
        issues = validate_evidence_spans(tuple(ledger_spans.values()), documents_by_id)
        if issues:
            raise ValueError(f"Cannot synthesize stale evidence: {issues[0].message}")
        for claim in ledger.claims:
            if claim.status == "pending":
                continue
            relevant_span_ids = tuple(span_id for span_id in claim.evidence_span_ids if span_id in ledger_spans)
            if claim.evidence_span_ids and not relevant_span_ids:
                continue
            if not claim.evidence_span_ids and scope.kind != "corpus":
                continue
            scoped_claim = claim.model_copy(update={"evidence_span_ids": relevant_span_ids})
            existing = claims_by_id.get(claim.claim_id)
            if existing is not None and existing != scoped_claim:
                raise ValueError(f"Claim ID {claim.claim_id!r} has conflicting resolved content.")
            claims_by_id[claim.claim_id] = scoped_claim
            for span_id in relevant_span_ids:
                span = ledger_spans[span_id]
                existing_span = spans_by_id.get(span_id)
                if existing_span is not None and existing_span != span:
                    raise ValueError(f"Evidence span ID {span_id!r} has conflicting content.")
                spans_by_id[span_id] = span
        for relation in ledger.relations:
            relations_by_key[(relation.source_claim_id, relation.target_claim_id, relation.relation)] = relation

    claims = tuple(sorted(claims_by_id.values(), key=lambda claim: claim.claim_id))
    known_claims = {claim.claim_id for claim in claims}
    relations = tuple(
        relation
        for relation in relations_by_key.values()
        if relation.source_claim_id in known_claims and relation.target_claim_id in known_claims
    )
    contributing_ids = tuple(sorted({span.document_id for span in spans_by_id.values()}))
    unresolved_ids = tuple(sorted(set(documents_by_id) - set(contributing_ids)))
    groups = _groups(claims, topics or {})
    synthesis_id = _synthesis_id(
        corpus_fingerprint,
        scope,
        claims,
        tuple(spans_by_id.values()),
        relations,
        groups,
    )
    return EvidenceBoundSynthesis(
        synthesis_id=synthesis_id,
        corpus_fingerprint=corpus_fingerprint,
        author_scope=scope,
        claims=claims,
        evidence_spans=tuple(sorted(spans_by_id.values(), key=lambda span: span.span_id)),
        relations=relations,
        groups=groups,
        coverage=SynthesisCoverage(
            documents_inspected=len(scoped_documents),
            documents_contributing=len(contributing_ids),
            contributing_document_ids=contributing_ids,
            unresolved_document_ids=unresolved_ids,
        ),
    )


def _document_in_scope(document: CorpusDocument, scope: AuthorScope) -> bool:
    if scope.kind == "corpus":
        return True
    scoped = {author.casefold() for author in scope.authors}
    return bool(scoped.intersection(author.casefold() for author in document.authors))


def _groups(
    claims: tuple[AuditedClaim, ...],
    topics: Mapping[str, str],
) -> tuple[SynthesisClaimGroup, ...]:
    grouped: dict[str, list[str]] = {}
    for claim in claims:
        topic = " ".join(topics.get(claim.claim_id, "Ungrouped").strip().split()) or "Ungrouped"
        grouped.setdefault(topic, []).append(claim.claim_id)
    return tuple(SynthesisClaimGroup(topic=topic, claim_ids=tuple(claim_ids)) for topic, claim_ids in sorted(grouped.items()))


def _synthesis_id(
    corpus_fingerprint: str,
    scope: AuthorScope,
    claims: tuple[AuditedClaim, ...],
    spans: tuple[EvidenceSpan, ...],
    relations: tuple[ClaimRelation, ...],
    groups: tuple[SynthesisClaimGroup, ...],
) -> str:
    payload = json.dumps(
        {
            "version": EVIDENCE_SYNTHESIS_VERSION,
            "corpus_fingerprint": corpus_fingerprint,
            "scope": scope.model_dump(mode="json"),
            "claims": [claim.model_dump(mode="json") for claim in claims],
            "spans": [span.model_dump(mode="json") for span in spans],
            "relations": [relation.model_dump(mode="json") for relation in relations],
            "groups": [group.model_dump(mode="json") for group in groups],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS evidence_syntheses (
            synthesis_id TEXT PRIMARY KEY,
            corpus_fingerprint TEXT NOT NULL,
            scope_key TEXT NOT NULL,
            record_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS evidence_syntheses_lookup
        ON evidence_syntheses (corpus_fingerprint, scope_key);
        """
    )
