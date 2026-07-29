"""Tests for content-addressed generated artifact paths."""

from pathlib import Path

from author_corpus.ingestion import load_corpus
from author_corpus.persistence import CacheLayout, corpus_fingerprint, read_manifest, write_manifest


def test_fingerprint_changes_with_content_or_index_options(tmp_path: Path) -> None:
    """Invalidate generated artifacts when content or relevant options change."""
    path = tmp_path / "work.md"
    path.write_text("# Work\n\nFirst version.", encoding="utf-8")
    first = load_corpus([path]).documents
    first_fingerprint = corpus_fingerprint(first, options={"chunk_size": 512})

    path.write_text("# Work\n\nSecond version.", encoding="utf-8")
    second = load_corpus([path]).documents

    assert corpus_fingerprint(second, options={"chunk_size": 512}) != first_fingerprint
    assert corpus_fingerprint(first, options={"chunk_size": 1024}) != first_fingerprint


def test_manifest_excludes_content_and_author_names(tmp_path: Path) -> None:
    """Keep generated manifests useful without duplicating private text."""
    path = tmp_path / "work.md"
    path.write_text(
        """\
---
author: "Avery Stone"
---

Private synthetic prose.
""",
        encoding="utf-8",
    )
    documents = load_corpus([path]).documents
    layout = CacheLayout(tmp_path / ".cache", corpus_fingerprint(documents))

    write_manifest(layout, documents, options={"chunk_size": 1024})

    manifest_text = layout.manifest_path.read_text(encoding="utf-8")
    assert "Avery Stone" not in manifest_text
    assert "Private synthetic prose" not in manifest_text
    manifest = read_manifest(layout)
    assert manifest is not None
    assert manifest["document_count"] == 1


def test_cross_fingerprint_audit_store_paths_are_stable(tmp_path: Path) -> None:
    """Keep generated-answer traces and human reviews outside one corpus build."""
    layout = CacheLayout(tmp_path / ".cache", "fingerprint")

    assert layout.query_trace_path == tmp_path / ".cache" / "query_traces.sqlite3"
    assert layout.claim_review_path == tmp_path / ".cache" / "claim_reviews.sqlite3"
