"""Conservative author-identity resolution for semantic retrieval queries."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

_GENERIC_AUTHOR_REFERENCE = re.compile(
    r"\b(?:(?:the\s+)?(?:main|corpus)\s+(?:author|writer)|(?:the|this)\s+(?:author|writer))"
    r"(?P<possessive>['’]s)?\b",
    re.IGNORECASE,
)
_PROFILE_SIGNAL = re.compile(
    r"\b(?:biograph\w*|personal|family|siblings?|brothers?|sisters?|children|sons?|daughters?|"
    r"grew up|childhood|hobbies)\b|"
    r"\b(?:favorite|favourite)\s+(?:things?|activities|hobbies|foods?|books?|places?)\b",
    re.IGNORECASE,
)
_INTEREST_SIGNAL = re.compile(
    r"\b(?:favorite|favourite|hobbies|interests?|preferences?)\b",
    re.IGNORECASE,
)
_FAMILY_SIGNAL = re.compile(
    r"\b(?:family|siblings?|brothers?|sisters?|children|sons?|daughters?)\b",
    re.IGNORECASE,
)
_REDUNDANT_AUTHORSHIP_CLAUSE = re.compile(
    r"\b(?:the\s+)?corpus author\s+is\s+(?:(?:the\s+)?author|(?:the\s+)?corpus author)\s+of\s+"
    r"(?:all\s+of\s+)?(?:the\s+)?(?:articles|documents|posts|works|pieces)\s*[.;:]?\s*",
    re.IGNORECASE,
)
_GENERIC_IDENTITY_TERMS = {
    "author",
    "writer",
    "the author",
    "the writer",
    "main author",
    "corpus author",
    "she",
    "her",
    "he",
    "him",
    "they",
    "them",
}


class AuthorIdentity(BaseModel):
    """One canonical corpus author and explicitly trusted aliases."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_name: str = Field(min_length=1)
    aliases: tuple[str, ...] = ()

    @field_validator("canonical_name")
    @classmethod
    def normalize_canonical_name(cls, value: str) -> str:
        """Strip redundant whitespace from the canonical display name."""
        return _display(value)

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, aliases: tuple[str, ...]) -> tuple[str, ...]:
        """Reject blank, generic, and repeated aliases."""
        normalized: set[str] = set()
        cleaned: list[str] = []
        for alias in aliases:
            display = _display(alias)
            key = _key(display)
            if not key:
                raise ValueError("Author aliases must not be blank.")
            if key in _GENERIC_IDENTITY_TERMS:
                raise ValueError(f"Author alias {display!r} is too generic.")
            if key in normalized:
                raise ValueError(f"Author alias {display!r} is repeated.")
            normalized.add(key)
            cleaned.append(display)
        return tuple(cleaned)

    @classmethod
    def from_catalog(
        cls,
        *,
        default_author: str,
        aliases: Iterable[str] = (),
        catalog_authors: Iterable[str],
    ) -> Self:
        """Resolve one default author and reject aliases colliding with another credit."""
        authors = tuple(_display(author) for author in catalog_authors)
        canonical_key = _key(default_author)
        canonical = next((author for author in authors if _key(author) == canonical_key), None)
        if canonical is None:
            raise ValueError(f"The configured default author {default_author!r} is not in the catalog.")

        other_author_terms = {
            term
            for author in authors
            if _key(author) != canonical_key
            for term in (_key(author), *(_key(part) for part in author.split()))
        }
        cleaned_aliases: list[str] = []
        seen = {canonical_key}
        for alias in aliases:
            display = _display(alias)
            key = _key(display)
            if key in other_author_terms:
                raise ValueError(f"Author alias {display!r} collides with another catalog author.")
            if key not in seen:
                cleaned_aliases.append(display)
                seen.add(key)
        return cls(canonical_name=canonical, aliases=tuple(cleaned_aliases))


class AuthorQueryResolution(BaseModel):
    """An inspectable, conservative author-reference substitution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    original_query: str = Field(min_length=1)
    retrieval_query: str = Field(min_length=1)
    grounding_question: str = Field(min_length=1)
    canonical_author: str | None = None
    matched_references: tuple[str, ...] = ()
    added_profile_context: bool = False

    @property
    def changed(self) -> bool:
        """Return whether retrieval sees wording different from the caller's query."""
        return self.retrieval_query != self.original_query


def parse_author_aliases(value: str | None) -> tuple[str, ...]:
    """Parse an optional JSON array from the private environment boundary."""
    if value is None or not value.strip():
        return ()
    try:
        decoded: object = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("AUTHOR_CORPUS_DEFAULT_AUTHOR_ALIASES must be a JSON array of strings.") from exc
    if not isinstance(decoded, list) or any(not isinstance(item, str) for item in decoded):
        raise ValueError("AUTHOR_CORPUS_DEFAULT_AUTHOR_ALIASES must be a JSON array of strings.")
    return tuple(_display(item) for item in decoded)


def resolve_author_query(query: str, identity: AuthorIdentity | None) -> AuthorQueryResolution:
    """Replace only configured identity references with a semantic corpus role.

    This function never guesses nicknames or uses fuzzy name similarity. Alias
    links must cross the private configuration boundary explicitly.
    """
    original = query.strip()
    if not original:
        raise ValueError("Query must not be empty.")
    if identity is None:
        return AuthorQueryResolution(
            original_query=original,
            retrieval_query=original,
            grounding_question=original,
        )

    matched = [match.group() for match in _GENERIC_AUTHOR_REFERENCE.finditer(original)]

    def generic_replacement(match: re.Match[str]) -> str:
        return "the corpus author's" if match.group("possessive") else "the corpus author"

    def canonical_replacement(match: re.Match[str]) -> str:
        suffix = "'s" if match.group("possessive") else ""
        return f"{identity.canonical_name}{suffix}"

    retrieval_query = _GENERIC_AUTHOR_REFERENCE.sub(generic_replacement, original)
    grounding_question = _GENERIC_AUTHOR_REFERENCE.sub(canonical_replacement, original)
    identity_terms = (identity.canonical_name, *identity.aliases)
    alternatives = "|".join(re.escape(term) for term in sorted(identity_terms, key=len, reverse=True))
    identity_pattern = re.compile(
        rf"(?<!\w)(?:{alternatives})(?P<possessive>['’]s)?(?!\w)",
        re.IGNORECASE,
    )

    def identity_replacement(match: re.Match[str]) -> str:
        matched.append(match.group())
        return "the corpus author's" if match.group("possessive") else "the corpus author"

    retrieval_query = identity_pattern.sub(identity_replacement, retrieval_query)
    grounding_question = identity_pattern.sub(canonical_replacement, grounding_question)

    if not matched:
        return AuthorQueryResolution(
            original_query=original,
            retrieval_query=original,
            grounding_question=original,
        )

    retrieval_query = re.sub(
        r"\b(?:the\s+author\s+)?the corpus author\b", "the corpus author", retrieval_query, flags=re.IGNORECASE
    )
    retrieval_query = _REDUNDANT_AUTHORSHIP_CLAUSE.sub("Regarding the corpus author, ", retrieval_query)
    retrieval_query = " ".join(retrieval_query.split())

    added_profile_context = bool(_PROFILE_SIGNAL.search(original))
    if added_profile_context:
        profile_terms = ["corpus author", "personal facts", "biography"]
        if _INTEREST_SIGNAL.search(original):
            profile_terms.extend(("interests", "preferences"))
        if _FAMILY_SIGNAL.search(original):
            profile_terms.extend(("family", "siblings", "brothers", "sisters"))
        retrieval_query = " ".join(profile_terms)

    return AuthorQueryResolution(
        original_query=original,
        retrieval_query=retrieval_query,
        grounding_question=grounding_question,
        canonical_author=identity.canonical_name if matched else None,
        matched_references=tuple(dict.fromkeys(matched)),
        added_profile_context=added_profile_context,
    )


def _display(value: str) -> str:
    return " ".join(value.strip().split())


def _key(value: str) -> str:
    return _display(value).casefold()
