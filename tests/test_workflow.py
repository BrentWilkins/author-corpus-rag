"""Tests for safe, explicit notebook execution switches."""

import pytest

from author_corpus.workflow import NotebookRunOptions


def test_notebook_expensive_operations_are_disabled_by_default() -> None:
    """Keep index building and generation off without explicit environment flags."""
    options = NotebookRunOptions.from_environment({})

    assert options.allow_index_build is False
    assert options.run_grounded_answer is False


def test_notebook_accepts_explicit_true_flags() -> None:
    """Enable an expensive operation only for a recognized true value."""
    options = NotebookRunOptions.from_environment(
        {
            "AUTHOR_CORPUS_ALLOW_INDEX_BUILD": "yes",
            "AUTHOR_CORPUS_RUN_GROUNDED_ANSWER": "1",
        }
    )

    assert options.allow_index_build is True
    assert options.run_grounded_answer is True


def test_notebook_rejects_ambiguous_flag_values() -> None:
    """Reject typos instead of silently changing execution behavior."""
    with pytest.raises(ValueError, match="AUTHOR_CORPUS_ALLOW_INDEX_BUILD"):
        NotebookRunOptions.from_environment({"AUTHOR_CORPUS_ALLOW_INDEX_BUILD": "sometimes"})
