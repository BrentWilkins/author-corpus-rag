"""Static safety checks for the interactive notebook."""

from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parents[1] / "notebooks" / "author_corpus_analyzer.ipynb"


def test_notebook_requires_explicit_expensive_operation_flags() -> None:
    """Keep model generation and missing-index builds opt-in."""
    notebook_text = NOTEBOOK_PATH.read_text(encoding="utf-8")

    assert "AUTHOR_CORPUS_ALLOW_INDEX_BUILD" in notebook_text
    assert "AUTHOR_CORPUS_RUN_GROUNDED_ANSWER" in notebook_text


def test_notebook_cannot_resume_broad_synthesis() -> None:
    """Prevent Run All from invoking the unaudited synthesis function."""
    notebook_text = NOTEBOOK_PATH.read_text(encoding="utf-8")

    assert "build_cached_knowledge" not in notebook_text
    assert "knowledge.synthesis" not in notebook_text
    assert "--include-experimental-synthesis" in notebook_text
