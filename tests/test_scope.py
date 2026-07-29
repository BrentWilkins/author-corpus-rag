"""Tests for explicit multi-author-ready query scopes."""

import pytest
from pydantic import ValidationError

from author_corpus.scope import AuthorScope


def test_single_author_scope_normalizes_display_name() -> None:
    """Normalize configured names without fuzzy identity inference."""
    scope = AuthorScope.for_author("  Avery   Stone ")

    assert scope.kind == "authors"
    assert scope.authors == ("Avery Stone",)
    assert scope.cache_key == "authors:avery stone"


def test_comparison_scope_requires_two_distinct_authors() -> None:
    """Reject incomplete or duplicate comparison scopes."""
    with pytest.raises(ValidationError, match="at least two"):
        AuthorScope(kind="comparison", authors=("Avery Stone",))
    with pytest.raises(ValidationError, match="duplicate"):
        AuthorScope(kind="comparison", authors=("Avery Stone", "avery stone"))


def test_corpus_scope_cannot_silently_filter_authors() -> None:
    """Keep whole-corpus and author-filtered operations distinct."""
    with pytest.raises(ValidationError, match="cannot name"):
        AuthorScope(kind="corpus", authors=("Avery Stone",))
