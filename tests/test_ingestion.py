"""Tests for tolerant, source-independent corpus ingestion."""

from pathlib import Path

from author_corpus.ingestion import load_corpus, load_corpus_config


def test_loads_standard_multi_author_front_matter(tmp_path: Path) -> None:
    """Parse plural authors and retain all publication sources."""
    path = tmp_path / "work.md"
    path.write_text(
        """\
---
title: "Shared Work"
authors:
  - "Avery Stone"
  - "Jordan Vale"
date: "2026-04-09"
sources:
  - uri: "https://example.test/original"
    source_type: webpage
    is_canonical: true
  - uri: "https://mirror.example.test/shared-work"
    source_type: webpage
---

The document body.
""",
        encoding="utf-8",
    )

    result = load_corpus([path])

    assert not result.errors
    document = result.documents[0]
    assert document.title == "Shared Work"
    assert document.authors == ("Avery Stone", "Jordan Vale")
    assert document.content == "The document body."
    assert len(document.sources) == 3
    assert document.canonical_source is not None
    assert document.canonical_source.uri == "https://example.test/original"


def test_recovers_common_malformed_multi_author_value(tmp_path: Path) -> None:
    """Recover quoted author names while warning about malformed YAML."""
    path = tmp_path / "legacy.md"
    path.write_text(
        """\
---
title: "Recovered Work"
author: "Avery Stone" and "Jordan Vale"
---

Recovered body.
""",
        encoding="utf-8",
    )

    result = load_corpus([path])

    assert not result.errors
    assert result.documents[0].authors == ("Avery Stone", "Jordan Vale")
    assert [issue.code for issue in result.warnings] == ["front_matter_recovered"]


def test_missing_authors_warns_but_does_not_fail(tmp_path: Path) -> None:
    """Allow semantic indexing when exact author metadata is unavailable."""
    path = tmp_path / "unknown.md"
    path.write_text("# Untagged Work\n\nUseful content.", encoding="utf-8")

    result = load_corpus([path])

    assert not result.errors
    assert result.documents[0].authors == ()
    assert "authors_missing" in {issue.code for issue in result.warnings}


def test_config_resolves_relative_inputs_and_applies_metadata(tmp_path: Path) -> None:
    """Apply trusted local overrides without embedding them in public code."""
    source_dir = tmp_path / "private"
    source_dir.mkdir()
    (source_dir / "paper.txt").write_text("A synthetic paper.", encoding="utf-8")
    config = tmp_path / "corpus.local.yaml"
    config.write_text(
        """\
name: Private Local Corpus
inputs:
  - path: private
    metadata:
      document_type: paper
      authors:
        - Avery Stone
""",
        encoding="utf-8",
    )

    result = load_corpus_config(config)

    result.raise_for_errors()
    assert result.name == "Private Local Corpus"
    assert result.documents[0].document_type == "paper"
    assert result.documents[0].authors == ("Avery Stone",)
    assert result.documents[0].metadata_provenance["authors"] == "corpus_config"
