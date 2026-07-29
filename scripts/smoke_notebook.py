"""Execute the notebook in memory with expensive operations forced off."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import nbformat
from nbclient import NotebookClient

_SKIP_TAG = "skip-ci-execution"


def main() -> None:
    """Run the public notebook path without private data or persisted output."""
    project_root = Path(__file__).resolve().parents[1]
    notebook_path = project_root / "notebooks" / "author_corpus_analyzer.ipynb"

    with TemporaryDirectory(prefix="author-corpus-rag-notebook-") as temporary_directory:
        temporary_root = Path(temporary_directory)
        config_path = _write_synthetic_corpus(temporary_root)
        os.environ.update(
            {
                "AUTHOR_CORPUS_ALLOW_INDEX_BUILD": "",
                "AUTHOR_CORPUS_CACHE_DIR": str(temporary_root / "cache"),
                "AUTHOR_CORPUS_CLAIM_EVAL": str(project_root / "evaluation" / "claim-classification.yaml"),
                "AUTHOR_CORPUS_CONFIG": str(config_path),
                "AUTHOR_CORPUS_DEFAULT_AUTHOR": "",
                "AUTHOR_CORPUS_DEFAULT_AUTHOR_ALIASES": "",
                "AUTHOR_CORPUS_GENERATED_CLAIM_EVAL": "",
                "AUTHOR_CORPUS_NOTEBOOK_QUERY": "How many documents are in this corpus?",
                "AUTHOR_CORPUS_RETRIEVAL_EVAL": "",
                "AUTHOR_CORPUS_ROUTING_EVAL": str(project_root / "evaluation" / "query-routing.yaml"),
                "AUTHOR_CORPUS_RUN_GROUNDED_ANSWER": "",
                "EMBEDDING_MODEL": "disabled-for-public-smoke-test",
                "OLLAMA_MODEL": "",
            }
        )

        notebook = nbformat.read(notebook_path, as_version=4)
        started_at = perf_counter()
        client = NotebookClient(
            notebook,
            timeout=300,
            interrupt_on_timeout=True,
            resources={"metadata": {"path": str(project_root)}},
            skip_cells_with_tag=_SKIP_TAG,
        )
        client.execute()
        elapsed_seconds = perf_counter() - started_at

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    skipped_cells = [cell for cell in code_cells if _SKIP_TAG in cell.metadata.get("tags", [])]
    print(
        f"Notebook smoke test passed: {len(code_cells) - len(skipped_cells)} executed code cells, "
        f"{len(skipped_cells)} expensive cells skipped, {elapsed_seconds:.3f}s."
    )


def _write_synthetic_corpus(root: Path) -> Path:
    """Write one public synthetic document and its isolated corpus config."""
    article_path = root / "synthetic-article.md"
    article_path.write_text(
        """\
---
title: "A Synthetic Article"
author: "Avery Stone"
document_type: article
sources:
  - uri: "https://example.org/synthetic-article"
    source_type: webpage
    is_canonical: true
---

This synthetic passage exists only to exercise the public notebook path.
""",
        encoding="utf-8",
    )
    config_path = root / "corpus.yaml"
    config_path.write_text(
        json.dumps(
            {
                "name": "Public Notebook Smoke Corpus",
                "inputs": [{"path": str(article_path)}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return config_path


if __name__ == "__main__":
    main()
