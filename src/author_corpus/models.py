"""Normalized models shared by every corpus input and query layer."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

IssueSeverity = Literal["warning", "error"]


class SourceReference(BaseModel):
    """One location or manifestation of a logical work."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str
    source_type: str
    is_canonical: bool = False
    publisher: str | None = None
    retrieved_at: str | None = None
    content_hash: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("uri", "source_type")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Reject empty source identifiers and types."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Source fields must not be empty.")
        return stripped

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible source representation."""
        return cast(dict[str, object], self.model_dump(mode="json"))


class CorpusDocument(BaseModel):
    """A logical work, independent of file format and publication location."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    title: str
    authors: tuple[str, ...]
    content: str
    content_hash: str
    sources: tuple[SourceReference, ...]
    published_at: str | None = None
    document_type: str = "document"
    raw_metadata: dict[str, object] = Field(default_factory=dict)
    metadata_provenance: dict[str, str] = Field(default_factory=dict)

    @field_validator(
        "document_id",
        "title",
        "content",
        "content_hash",
        "document_type",
    )
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Reject empty required document fields."""
        if not value.strip():
            raise ValueError("Required document fields must not be empty.")
        return value

    @field_validator("authors")
    @classmethod
    def validate_authors(cls, authors: tuple[str, ...]) -> tuple[str, ...]:
        """Reject empty or repeated normalized author names."""
        if any(not author.strip() for author in authors):
            raise ValueError("Author names must not be empty.")
        if len({author.casefold() for author in authors}) != len(authors):
            raise ValueError("Author names must not be repeated.")
        return authors

    @property
    def canonical_source(self) -> SourceReference | None:
        """Return the canonical source, or the first source as a fallback."""
        return next(
            (source for source in self.sources if source.is_canonical),
            self.sources[0] if self.sources else None,
        )

    def as_dict(self, *, include_content: bool = True) -> dict[str, object]:
        """Return a JSON-compatible document representation."""
        result = cast(dict[str, object], self.model_dump(mode="json"))
        if not include_content:
            result.pop("content")
        return result


class LoadIssue(BaseModel):
    """A recoverable warning or blocking ingestion error."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    severity: IssueSeverity
    code: str
    message: str
    path: Path | None = None

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible issue representation."""
        return cast(dict[str, object], self.model_dump(mode="json"))


class CorpusLoadResult(BaseModel):
    """Documents plus explicit information about imperfect inputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    documents: tuple[CorpusDocument, ...]
    issues: tuple[LoadIssue, ...] = ()
    name: str | None = None

    @property
    def errors(self) -> tuple[LoadIssue, ...]:
        """Return only blocking ingestion issues."""
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[LoadIssue, ...]:
        """Return only recoverable ingestion issues."""
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def raise_for_errors(self) -> None:
        """Raise one exception containing every blocking ingestion issue."""
        if not self.errors:
            return
        details = "\n".join(f"- {issue.path or '<corpus>'}: {issue.message}" for issue in self.errors)
        raise ValueError(f"Corpus ingestion failed:\n{details}")
