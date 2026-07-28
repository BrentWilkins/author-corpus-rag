"""Tests for validated public corpus models."""

import pytest
from pydantic import ValidationError

from author_corpus.models import CorpusDocument, SourceReference


def test_document_rejects_duplicate_normalized_authors() -> None:
    """Reject duplicate author identities before catalog construction."""
    with pytest.raises(ValidationError, match="must not be repeated"):
        CorpusDocument(
            document_id="synthetic-work",
            title="Synthetic Work",
            authors=("Avery Stone", "avery stone"),
            content="Synthetic prose.",
            content_hash="synthetic-hash",
            sources=(
                SourceReference(
                    uri="https://example.test/work",
                    source_type="webpage",
                ),
            ),
        )


def test_source_rejects_blank_uri() -> None:
    """Reject a source record that cannot identify its location."""
    with pytest.raises(ValidationError, match="must not be empty"):
        SourceReference(uri=" ", source_type="webpage")
