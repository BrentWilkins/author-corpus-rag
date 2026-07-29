"""Conservative author-identity resolution for semantic retrieval queries."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

from author_corpus.scope import AuthorScope

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
    r"\b(?:the\s+)?(?:corpus|selected) author\s+is\s+"
    r"(?:(?:the\s+)?author|(?:the\s+)?(?:corpus|selected) author)\s+of\s+"
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
    canonical_authors: tuple[str, ...] = ()
    author_scope: AuthorScope = Field(default_factory=AuthorScope)
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


def resolve_author_query(
    query: str,
    identity: AuthorIdentity | None,
    *,
    catalog_authors: Iterable[str] = (),
    requested_scope: AuthorScope | None = None,
) -> AuthorQueryResolution:
    """Replace only configured identity references with a semantic corpus role.

    This function never guesses nicknames or uses fuzzy name similarity. Alias
    links must cross the private configuration boundary explicitly.
    """
    original = query.strip()
    if not original:
        raise ValueError("Query must not be empty.")
    canonical_by_key = {_key(author): _display(author) for author in catalog_authors}
    if identity is not None:
        canonical_by_key.setdefault(_key(identity.canonical_name), identity.canonical_name)
    scope = _canonical_scope(requested_scope, canonical_by_key)
    matched: list[str] = []
    matched_authors: list[str] = []
    retrieval_query = original
    grounding_question = original
    generic_matches = tuple(_GENERIC_AUTHOR_REFERENCE.finditer(original))
    generic_author = _generic_author(identity, scope)
    if generic_matches and generic_author is None:
        raise ValueError("A generic author reference is ambiguous for the selected author scope.")
    matched.extend(match.group() for match in generic_matches)
    if generic_author is not None and generic_matches:
        matched_authors.append(generic_author)

    def generic_replacement(match: re.Match[str]) -> str:
        return "the selected author's" if match.group("possessive") else "the selected author"

    def generic_grounding_replacement(match: re.Match[str]) -> str:
        suffix = "'s" if match.group("possessive") else ""
        return f"{generic_author}{suffix}"

    if generic_matches:
        retrieval_query = _GENERIC_AUTHOR_REFERENCE.sub(generic_replacement, retrieval_query)
        grounding_question = _GENERIC_AUTHOR_REFERENCE.sub(generic_grounding_replacement, grounding_question)

    terms = dict(canonical_by_key)
    if identity is not None:
        terms.update({_key(alias): identity.canonical_name for alias in identity.aliases})
    identity_pattern = _author_pattern(terms)

    def retrieval_identity_replacement(match: re.Match[str]) -> str:
        canonical = terms[_key(match.group("name"))]
        matched.append(match.group())
        matched_authors.append(canonical)
        return "the selected author's" if match.group("possessive") else "the selected author"

    def grounding_identity_replacement(match: re.Match[str]) -> str:
        canonical = terms[_key(match.group("name"))]
        suffix = "'s" if match.group("possessive") else ""
        return f"{canonical}{suffix}"

    if identity_pattern is not None:
        retrieval_query = identity_pattern.sub(retrieval_identity_replacement, retrieval_query)
        grounding_question = identity_pattern.sub(grounding_identity_replacement, grounding_question)

    if matched:
        retrieval_query = re.sub(
            r"\b(?:the\s+author\s+)?the selected author\b",
            "the selected author",
            retrieval_query,
            flags=re.IGNORECASE,
        )
        retrieval_query = _REDUNDANT_AUTHORSHIP_CLAUSE.sub("Regarding the selected author, ", retrieval_query)
        retrieval_query = " ".join(retrieval_query.split())
    inferred_authors = tuple(dict.fromkeys(matched_authors))
    if scope is None:
        if len(inferred_authors) > 1:
            scope = AuthorScope(kind="comparison", authors=inferred_authors)
        elif inferred_authors:
            scope = AuthorScope.for_author(inferred_authors[0])
        elif identity is not None:
            scope = AuthorScope.for_author(identity.canonical_name)
        else:
            scope = AuthorScope()
    elif inferred_authors and not _scope_contains(scope, inferred_authors):
        raise ValueError("The query names an author outside the explicitly selected author scope.")

    added_profile_context = bool(_PROFILE_SIGNAL.search(original))
    if added_profile_context and matched:
        profile_terms = ["selected author", "personal facts", "biography"]
        if _INTEREST_SIGNAL.search(original):
            profile_terms.extend(("interests", "preferences"))
        if _FAMILY_SIGNAL.search(original):
            profile_terms.extend(("family", "siblings", "brothers", "sisters"))
        retrieval_query = " ".join(profile_terms)

    return AuthorQueryResolution(
        original_query=original,
        retrieval_query=retrieval_query,
        grounding_question=grounding_question,
        canonical_author=inferred_authors[0] if len(inferred_authors) == 1 else None,
        canonical_authors=inferred_authors,
        author_scope=scope,
        matched_references=tuple(dict.fromkeys(matched)),
        added_profile_context=added_profile_context,
    )


def _canonical_scope(
    scope: AuthorScope | None,
    canonical_by_key: dict[str, str],
) -> AuthorScope | None:
    if scope is None or scope.kind == "corpus":
        return scope
    authors: list[str] = []
    for author in scope.authors:
        canonical = canonical_by_key.get(_key(author))
        if canonical is None:
            raise ValueError(f"Unknown author in requested scope: {author!r}.")
        authors.append(canonical)
    if len(authors) > 1:
        return AuthorScope(kind="comparison", authors=tuple(authors))
    return AuthorScope(kind="authors", authors=tuple(authors))


def _generic_author(identity: AuthorIdentity | None, scope: AuthorScope | None) -> str | None:
    if scope is not None:
        return scope.authors[0] if len(scope.authors) == 1 else None
    return identity.canonical_name if identity is not None else None


def _author_pattern(terms: dict[str, str]) -> re.Pattern[str] | None:
    if not terms:
        return None
    alternatives = "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True))
    return re.compile(
        rf"(?<!\w)(?P<name>{alternatives})(?P<possessive>['’]s)?(?!\w)",
        re.IGNORECASE,
    )


def _scope_contains(scope: AuthorScope, authors: tuple[str, ...]) -> bool:
    if scope.kind == "corpus":
        return True
    selected = {author.casefold() for author in scope.authors}
    return all(author.casefold() in selected for author in authors)


def _display(value: str) -> str:
    return " ".join(value.strip().split())


def _key(value: str) -> str:
    return _display(value).casefold()
