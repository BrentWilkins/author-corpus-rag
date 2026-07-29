"""Tests for evidence-bound corpus synthesis."""

from pathlib import Path

import pytest

from author_corpus.audit import AuditedClaim, ClaimRelation, EvidenceLedger, EvidenceSpan
from author_corpus.ingestion import load_corpus
from author_corpus.models import CorpusDocument
from author_corpus.persistence import corpus_fingerprint
from author_corpus.scope import AuthorScope
from author_corpus.synthesis import EvidenceSynthesisStore, build_evidence_bound_synthesis


def test_builds_current_source_bound_synthesis_with_coverage(tmp_path: Path) -> None:
    """Combine resolved claims while retaining exact sources and unresolved documents."""
    documents = _documents(tmp_path)
    fingerprint = corpus_fingerprint(documents)
    span = EvidenceSpan.from_document(documents[0], "Heat training expands plasma volume.")
    ledger = EvidenceLedger(
        corpus_fingerprint=fingerprint,
        query="What changes?",
        answer="Heat training expands plasma volume.",
        claims=(
            AuditedClaim(
                claim_id="plasma-volume",
                statement="Heat training expands plasma volume.",
                status="supported",
                evidence_span_ids=(span.span_id,),
            ),
        ),
        evidence_spans=(span,),
    )

    synthesis = build_evidence_bound_synthesis(
        documents,
        (ledger,),
        corpus_fingerprint=fingerprint,
        author_scope=AuthorScope.for_author("Avery Stone"),
        topics={"plasma-volume": "Heat adaptation"},
    )

    assert synthesis.coverage.documents_inspected == 2
    assert synthesis.coverage.documents_contributing == 1
    assert synthesis.coverage.unresolved_document_ids == (documents[1].document_id,)
    assert synthesis.groups[0].topic == "Heat adaptation"
    assert "Heat training expands plasma volume." in synthesis.to_markdown()
    assert synthesis.validate_sources({document.document_id: document for document in documents}) == ()


def test_preserves_contradiction_relations_instead_of_smoothing(tmp_path: Path) -> None:
    """Retain both resolved claims and their explicit disagreement."""
    documents = _documents(tmp_path)
    fingerprint = corpus_fingerprint(documents)
    first_span = EvidenceSpan.from_document(documents[0], "Heat training expands plasma volume.")
    second_span = EvidenceSpan.from_document(documents[1], "Heat training does not replace hydration.")
    claims = (
        AuditedClaim(
            claim_id="expands",
            statement="Heat training expands plasma volume.",
            status="supported",
            evidence_span_ids=(first_span.span_id,),
        ),
        AuditedClaim(
            claim_id="replacement",
            statement="Heat training replaces hydration.",
            status="contradicted",
            evidence_span_ids=(second_span.span_id,),
        ),
    )
    relation = ClaimRelation(
        source_claim_id="replacement",
        target_claim_id="expands",
        relation="qualifies",
        rationale="Physiological adaptation does not remove hydration needs.",
    )
    ledger = EvidenceLedger(
        corpus_fingerprint=fingerprint,
        query="How should these findings be reconciled?",
        answer="",
        claims=claims,
        evidence_spans=(first_span, second_span),
        relations=(relation,),
    )

    synthesis = build_evidence_bound_synthesis(documents, (ledger,), corpus_fingerprint=fingerprint)

    assert synthesis.claims == claims
    assert synthesis.relations == (relation,)


def test_synthesis_store_is_immutable_and_scope_aware(tmp_path: Path) -> None:
    """Cache evidence synthesis by fingerprint and explicit author scope."""
    documents = _documents(tmp_path)
    fingerprint = corpus_fingerprint(documents)
    synthesis = build_evidence_bound_synthesis(documents, (), corpus_fingerprint=fingerprint)
    store = EvidenceSynthesisStore(tmp_path / "syntheses.sqlite3")

    store.put(synthesis)
    store.put(synthesis)

    assert store.get(synthesis.synthesis_id) == synthesis
    assert store.latest(corpus_fingerprint=fingerprint, author_scope=AuthorScope()) == synthesis
    with pytest.raises(ValueError, match="immutable"):
        store.put(synthesis.model_copy(update={"coverage": synthesis.coverage.model_copy(update={"documents_inspected": 0})}))


def test_rejects_stale_source_evidence(tmp_path: Path) -> None:
    """Stop synthesis when a dependent source version changed."""
    documents = _documents(tmp_path)
    fingerprint = corpus_fingerprint(documents)
    span = EvidenceSpan.from_document(documents[0], "Heat training expands plasma volume.")
    ledger = EvidenceLedger(
        corpus_fingerprint=fingerprint,
        query="What changes?",
        answer="",
        claims=(
            AuditedClaim(
                claim_id="claim",
                statement=span.text,
                status="supported",
                evidence_span_ids=(span.span_id,),
            ),
        ),
        evidence_spans=(span,),
    )
    changed = documents[0].model_copy(update={"content_hash": "changed"})

    with pytest.raises(ValueError, match="stale evidence"):
        build_evidence_bound_synthesis((changed, documents[1]), (ledger,), corpus_fingerprint=fingerprint)


def _documents(tmp_path: Path) -> tuple[CorpusDocument, ...]:
    (tmp_path / "first.md").write_text(
        "---\ntitle: First\nauthor: Avery Stone\n---\n\nHeat training expands plasma volume.",
        encoding="utf-8",
    )
    (tmp_path / "second.md").write_text(
        "---\ntitle: Second\nauthor: Avery Stone\n---\n\nHeat training does not replace hydration.",
        encoding="utf-8",
    )
    return load_corpus([tmp_path]).documents
