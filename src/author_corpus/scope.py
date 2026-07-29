"""Explicit corpus and author scopes shared by retrieval and derived knowledge."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

ScopeKind = Literal["corpus", "authors", "comparison"]


class AuthorScope(BaseModel):
    """A normalized, inspectable scope for one corpus operation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ScopeKind = "corpus"
    authors: tuple[str, ...] = ()

    @field_validator("authors")
    @classmethod
    def normalize_authors(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Normalize author display names and reject duplicates."""
        authors = tuple(" ".join(value.strip().split()) for value in values)
        if any(not author for author in authors):
            raise ValueError("Scoped author names must not be blank.")
        if len({author.casefold() for author in authors}) != len(authors):
            raise ValueError("An author scope cannot contain duplicate names.")
        return authors

    @model_validator(mode="after")
    def validate_kind(self) -> Self:
        """Require an author count appropriate for the selected scope kind."""
        if self.kind == "corpus" and self.authors:
            raise ValueError("Corpus-wide scope cannot name individual authors.")
        if self.kind == "authors" and not self.authors:
            raise ValueError("Author scope must name at least one author.")
        if self.kind == "comparison" and len(self.authors) < 2:
            raise ValueError("Comparison scope must name at least two authors.")
        return self

    @classmethod
    def for_author(cls, author: str) -> AuthorScope:
        """Return a single-author scope."""
        return cls(kind="authors", authors=(author,))

    @property
    def cache_key(self) -> str:
        """Return a stable human-readable cache-key component."""
        if self.kind == "corpus":
            return "corpus"
        return f"{self.kind}:{'|'.join(author.casefold() for author in self.authors)}"
