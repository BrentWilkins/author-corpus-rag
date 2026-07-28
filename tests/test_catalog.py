"""Tests for exact catalog queries over synthetic authors."""

import json
from pathlib import Path

from author_corpus.catalog import CorpusCatalog
from author_corpus.ingestion import load_corpus
from author_corpus.querying import build_catalog_tools


def test_catalog_counts_documents_and_coauthors(tmp_path: Path) -> None:
    """Count works once while representing multiple authors independently."""
    first = tmp_path / "first.md"
    first.write_text(
        """\
---
title: "First Work"
authors:
  - "Avery Stone"
  - "Jordan Vale"
document_type: article
---

First synthetic document.
""",
        encoding="utf-8",
    )
    second = tmp_path / "second.md"
    second.write_text(
        """\
---
title: "Second Work"
author: "Avery Stone"
document_type: article
---

Second synthetic document.
""",
        encoding="utf-8",
    )

    result = load_corpus([tmp_path])
    result.raise_for_errors()
    catalog = CorpusCatalog(tmp_path / "generated" / "catalog.sqlite3")
    catalog.rebuild(result.documents)

    assert catalog.count_documents() == 2
    assert catalog.count_documents(author="avery stone") == 2
    assert catalog.count_documents(author="Jordan Vale") == 1
    assert catalog.author_counts() == [("Avery Stone", 2), ("Jordan Vale", 1)]
    assert catalog.coauthor_counts("Avery Stone") == [("Jordan Vale", 1)]
    author_stats = catalog.author_document_stats("Avery Stone")
    assert author_stats.credited_documents == 2
    assert author_stats.sole_authored_documents == 1
    assert author_stats.coauthored_documents == 1

    entries = catalog.list_documents(author="Avery Stone")
    assert [entry.title for entry in entries] == ["First Work", "Second Work"]
    assert entries[0].authors == ("Avery Stone", "Jordan Vale")


def test_catalog_reports_documents_without_authors(tmp_path: Path) -> None:
    """Report missing authors without preventing exact catalog construction."""
    path = tmp_path / "anonymous.txt"
    path.write_text("A synthetic unattributed document.", encoding="utf-8")

    result = load_corpus([path])
    result.raise_for_errors()
    catalog = CorpusCatalog(tmp_path / "catalog.sqlite3")
    catalog.rebuild(result.documents)

    stats = catalog.stats()
    assert stats.total_documents == 1
    assert stats.documents_with_authors == 0
    assert stats.documents_without_authors == 1
    assert stats.distinct_authors == 0


def test_catalog_count_tool_reports_exhaustive_coverage(tmp_path: Path) -> None:
    """Return a computed count and state how much of the corpus was inspected."""
    path = tmp_path / "work.md"
    path.write_text(
        """\
---
author: "Avery Stone"
---

Synthetic prose.
""",
        encoding="utf-8",
    )
    result = load_corpus([path])
    catalog = CorpusCatalog(tmp_path / "catalog.sqlite3")
    catalog.rebuild(result.documents)
    tools = {tool.metadata.name: tool for tool in build_catalog_tools(catalog)}

    output = tools["count_corpus_documents"].call(author="Avery Stone")
    payload = json.loads(str(output))

    assert payload["count"] == 1
    assert payload["coverage"] == {
        "corpus_documents": 1,
        "exhaustive": True,
        "inspected_documents": 1,
    }

    stats_output = tools["get_author_document_stats"].call(author="Avery Stone")
    stats_payload = json.loads(str(stats_output))
    assert stats_payload == {
        "author": "Avery Stone",
        "coauthored_documents": 0,
        "coverage": {
            "corpus_documents": 1,
            "exhaustive": True,
            "inspected_documents": 1,
        },
        "credited_documents": 1,
        "sole_authored_documents": 1,
    }
