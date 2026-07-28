"""Execute the notebook in memory with expensive operations forced off."""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

import nbformat
from nbclient import NotebookClient


def main() -> None:
    """Run every notebook cell without persisting private execution output."""
    project_root = Path(__file__).resolve().parents[1]
    notebook_path = project_root / "notebooks" / "author_corpus_analyzer.ipynb"

    os.environ["AUTHOR_CORPUS_ALLOW_INDEX_BUILD"] = ""
    os.environ["AUTHOR_CORPUS_RUN_GROUNDED_ANSWER"] = ""

    notebook = nbformat.read(notebook_path, as_version=4)
    started_at = perf_counter()
    client = NotebookClient(
        notebook,
        timeout=300,
        interrupt_on_timeout=True,
        resources={"metadata": {"path": str(project_root)}},
    )
    client.execute()
    elapsed_seconds = perf_counter() - started_at
    code_cell_count = sum(cell.cell_type == "code" for cell in notebook.cells)
    print(f"Notebook smoke test passed: {code_cell_count} code cells in {elapsed_seconds:.3f}s.")


if __name__ == "__main__":
    main()
