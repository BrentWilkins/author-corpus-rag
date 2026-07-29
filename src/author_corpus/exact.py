"""Validated execution of exact natural-language catalog questions."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from author_corpus.catalog import AuthorDocumentStats, CatalogEntry, CorpusCatalog
from author_corpus.routing import CatalogTool, QueryRoute, QueryRouteDecision


class ExactCatalogStatus(StrEnum):
    """Possible outcomes of an exact catalog execution attempt."""

    COMPLETED = "completed"
    NEEDS_CLARIFICATION = "needs_clarification"


class ExactCatalogArguments(BaseModel):
    """Catalog filters resolved and validated against known values."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    author: str | None = None
    document_type: str | None = None
    exclude_author: str | None = None


class CatalogCoverage(BaseModel):
    """Coverage behind one completed exact result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    corpus_documents: int = Field(ge=0)
    matched_documents: int | None = Field(default=None, ge=0)
    exhaustive: bool = True


class CatalogAuthorCount(BaseModel):
    """One author and their exact distinct-document count."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    document_count: int = Field(ge=0)


class CatalogCoauthorCount(BaseModel):
    """One coauthor and their exact shared-document count."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    shared_document_count: int = Field(ge=0)


class CatalogDocumentResult(BaseModel):
    """One source-bearing logical document supporting an exact result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    title: str
    authors: tuple[str, ...]
    published_at: str | None = None
    document_type: str
    source_uris: tuple[str, ...]


class CatalogAuthorshipResult(BaseModel):
    """Exact sole-author and coauthored counts for one author."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    author: str
    credited_documents: int = Field(ge=0)
    sole_authored_documents: int = Field(ge=0)
    coauthored_documents: int = Field(ge=0)


class ExactCatalogResult(BaseModel):
    """An executed catalog answer or an explicit request for clarification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    tool: CatalogTool
    status: ExactCatalogStatus
    arguments: ExactCatalogArguments
    message: str
    coverage: CatalogCoverage | None = None
    count: int | None = Field(default=None, ge=0)
    authors: tuple[CatalogAuthorCount, ...] = ()
    coauthors: tuple[CatalogCoauthorCount, ...] = ()
    authorship: CatalogAuthorshipResult | None = None
    documents: tuple[CatalogDocumentResult, ...] = ()

    @model_validator(mode="after")
    def validate_status_contract(self) -> ExactCatalogResult:
        """Require coverage only when an exact operation completed."""
        if self.status is ExactCatalogStatus.COMPLETED and self.coverage is None:
            raise ValueError("Completed exact catalog results require coverage.")
        if self.status is ExactCatalogStatus.NEEDS_CLARIFICATION and self.coverage is not None:
            raise ValueError("Unresolved catalog requests cannot claim coverage.")
        return self

    def to_markdown(self, *, source_limit: int = 12) -> str:
        """Render a concise answer while retaining every source record in the model."""
        if self.status is ExactCatalogStatus.NEEDS_CLARIFICATION:
            return f"**Clarification needed:** {self.message}"

        lines = [self.message]
        if self.authors:
            lines.extend(f"- {item.name}: {item.document_count}" for item in self.authors)
        if self.coauthors:
            lines.extend(f"- {item.name}: {item.shared_document_count} shared" for item in self.coauthors)
        if self.authorship is not None:
            lines.extend(
                (
                    f"- Credited documents: {self.authorship.credited_documents}",
                    f"- Sole-authored documents: {self.authorship.sole_authored_documents}",
                    f"- Coauthored documents: {self.authorship.coauthored_documents}",
                )
            )

        source_documents = tuple(document for document in self.documents if document.source_uris)
        if source_documents and source_limit > 0:
            shown = source_documents[:source_limit]
            lines.append(f"\nSources shown: {len(shown)} of {len(source_documents)} matching documents.")
            lines.extend(f"- [{document.title}]({document.source_uris[0]})" for document in shown)
            if len(source_documents) > len(shown):
                lines.append(f"- …and {len(source_documents) - len(shown)} more retained in `exact_result.documents`.")

        if self.coverage is not None:
            matched = "" if self.coverage.matched_documents is None else f"; {self.coverage.matched_documents} matched"
            lines.append(f"\nCoverage: all {self.coverage.corpus_documents} catalog documents inspected{matched}.")
        return "\n".join(lines)


_GENERIC_AUTHOR_REFERENCES = re.compile(
    r"\b(the author|this author|an author|the writer|this writer|their|they|them|she|her|he|his)\b|\bauthor['’]s\b"
)
_AUTHORSHIP_VERBS = re.compile(r"\b(written|wrote|write|authored|author|published|contributed)\b")
_OTHER_AUTHORS = re.compile(r"\b(other|remaining)\s+(authors?|writers?|contributors?)\b|\bbesides\b")
_SPECIFIC_DOCUMENT_NOUNS = re.compile(r"\b(articles?|papers?|posts?|essays?|stories?)\b")
_EXPLICIT_AUTHOR_PATTERNS = (
    re.compile(r"\bhas\s+(.+?)\s+(?:written|authored|published|contributed)\b", re.IGNORECASE),
    re.compile(r"\bdid\s+(.+?)\s+(?:write|author|publish|contribute)\b", re.IGNORECASE),
    re.compile(r"\b(?:by|besides)\s+([\w][\w .'-]*?)(?:[?.,]|$)", re.IGNORECASE),
    re.compile(
        r"\b(?:documents?|works?|articles?|papers?|posts?|stats|statistics|credits|breakdown)\s+for\s+"
        r"([\w][\w .'-]*?)(?:[?.,]|$)",
        re.IGNORECASE,
    ),
    re.compile(r"\b([\w][\w .'-]*?)['’]s\s+(?:coauthors?|co-authors?|documents?|works?|articles?)\b", re.IGNORECASE),
)


def execute_catalog_query(
    catalog: CorpusCatalog,
    decision: QueryRouteDecision,
    *,
    default_author: str | None = None,
) -> ExactCatalogResult:
    """Resolve safe arguments and execute one exact routing decision.

    Names and document types must match values already present in the catalog.
    Ambiguous or unknown filters produce a clarification result instead of a
    misleading zero or an unfiltered corpus total.

    Args:
        catalog: Exhaustive SQLite metadata catalog.
        decision: An exact decision produced by the query router.
        default_author: Optional private focal author for phrases such as
            ``the author``. The value is validated against the catalog.

    Returns:
        A typed completed result or a typed clarification request.
    """
    if decision.route is not QueryRoute.EXACT_CATALOG or decision.catalog_tool is None:
        raise ValueError("Only exact catalog routing decisions can be executed by the catalog executor.")

    authors = tuple(name for name, _ in catalog.author_counts())
    document_types = tuple(name for name, _ in catalog.document_type_counts())
    arguments, error = _resolve_arguments(
        decision,
        authors=authors,
        document_types=document_types,
        default_author=default_author,
    )
    if error is not None:
        return ExactCatalogResult(
            query=decision.query,
            tool=decision.catalog_tool,
            status=ExactCatalogStatus.NEEDS_CLARIFICATION,
            arguments=arguments,
            message=error,
        )
    return _execute(catalog, decision, arguments)


def _resolve_arguments(
    decision: QueryRouteDecision,
    *,
    authors: tuple[str, ...],
    document_types: tuple[str, ...],
    default_author: str | None,
) -> tuple[ExactCatalogArguments, str | None]:
    query = decision.query
    normalized = _normalize(query)
    matched_authors = _values_in_query(query, authors)
    matched_types = _document_types_in_query(query, document_types)

    if len(matched_authors) > 1:
        return ExactCatalogArguments(), f"More than one catalog author was named: {', '.join(matched_authors)}."
    if len(matched_types) > 1:
        return ExactCatalogArguments(), f"More than one catalog document type was named: {', '.join(matched_types)}."

    author = matched_authors[0] if matched_authors else None
    document_type = matched_types[0] if matched_types else None
    unknown_author = _unknown_explicit_author(query, authors)
    if author is None and unknown_author is not None and not _is_generic_author_phrase(unknown_author):
        alias_matches = _author_alias_matches(unknown_author, authors)
        if len(alias_matches) == 1:
            author = alias_matches[0]
        elif len(alias_matches) > 1:
            return ExactCatalogArguments(document_type=document_type), (
                f"The author reference {unknown_author!r} is ambiguous: {', '.join(alias_matches)}."
            )
        else:
            return ExactCatalogArguments(document_type=document_type), f"No catalog author matches {unknown_author!r}."

    explicit_document_noun = _SPECIFIC_DOCUMENT_NOUNS.search(normalized)
    if explicit_document_noun is not None and document_type is None:
        available = ", ".join(document_types) or "<none>"
        return ExactCatalogArguments(author=author), (
            f"The requested document type {explicit_document_noun.group()!r} is not present in the catalog. "
            f"Available types: {available}."
        )

    tool = decision.catalog_tool
    needs_author = tool in {CatalogTool.LIST_COAUTHORS, CatalogTool.AUTHOR_DOCUMENT_STATS}
    needs_author = needs_author or (
        tool in {CatalogTool.COUNT_DOCUMENTS, CatalogTool.LIST_DOCUMENTS}
        and (_GENERIC_AUTHOR_REFERENCES.search(normalized) is not None or _AUTHORSHIP_VERBS.search(normalized) is not None)
    )
    exclude_author: str | None = None
    if tool is CatalogTool.LIST_AUTHORS and _OTHER_AUTHORS.search(normalized):
        exclude_author = author or _validated_default_author(default_author, authors)
        if exclude_author is None:
            if default_author is not None and default_author.strip():
                return ExactCatalogArguments(), f"The configured default author {default_author!r} is not in the catalog."
            return ExactCatalogArguments(), "Which author should be excluded from the list of other authors?"
        author = None
    elif author is None and needs_author:
        author = _validated_default_author(default_author, authors)
        if author is None:
            if default_author is not None and default_author.strip():
                return ExactCatalogArguments(document_type=document_type), (
                    f"The configured default author {default_author!r} is not in the catalog."
                )
            return ExactCatalogArguments(document_type=document_type), (
                "Which catalog author do you mean? Name one explicitly or set AUTHOR_CORPUS_DEFAULT_AUTHOR."
            )

    return ExactCatalogArguments(author=author, document_type=document_type, exclude_author=exclude_author), None


def _execute(
    catalog: CorpusCatalog,
    decision: QueryRouteDecision,
    arguments: ExactCatalogArguments,
) -> ExactCatalogResult:
    tool = decision.catalog_tool
    if tool is None:
        raise RuntimeError("An exact catalog decision unexpectedly lacked a tool.")
    total = catalog.stats().total_documents

    if tool is CatalogTool.COUNT_DOCUMENTS:
        entries = catalog.list_documents(author=arguments.author, document_type=arguments.document_type)
        count = catalog.count_documents(author=arguments.author, document_type=arguments.document_type)
        if count != len(entries):
            raise RuntimeError("Catalog count and supporting document records disagree.")
        description = _document_description(count, arguments)
        return ExactCatalogResult(
            query=decision.query,
            tool=tool,
            status=ExactCatalogStatus.COMPLETED,
            arguments=arguments,
            message=f"The exact catalog contains **{count} {description}**.",
            coverage=CatalogCoverage(corpus_documents=total, matched_documents=count),
            count=count,
            documents=tuple(_document(entry) for entry in entries),
        )

    if tool is CatalogTool.LIST_AUTHORS:
        author_values = tuple(
            CatalogAuthorCount(name=name, document_count=count)
            for name, count in catalog.author_counts()
            if arguments.exclude_author is None or _normalize(name) != _normalize(arguments.exclude_author)
        )
        qualifier = "other " if arguments.exclude_author is not None else ""
        entries = catalog.list_documents()
        if arguments.exclude_author is not None:
            excluded_key = _normalize(arguments.exclude_author)
            entries = [entry for entry in entries if any(_normalize(author) != excluded_key for author in entry.authors)]
        return ExactCatalogResult(
            query=decision.query,
            tool=tool,
            status=ExactCatalogStatus.COMPLETED,
            arguments=arguments,
            message=f"The catalog contains **{len(author_values)} {qualifier}authors**:",
            coverage=CatalogCoverage(corpus_documents=total, matched_documents=len(entries)),
            authors=author_values,
            documents=tuple(_document(entry) for entry in entries),
        )

    if tool is CatalogTool.LIST_COAUTHORS:
        if arguments.author is None:
            raise RuntimeError("Validated coauthor execution unexpectedly lacked an author.")
        coauthor_values = tuple(
            CatalogCoauthorCount(name=name, shared_document_count=count)
            for name, count in catalog.coauthor_counts(arguments.author)
        )
        author_key = _normalize(arguments.author)
        entries = [
            entry
            for entry in catalog.list_documents(author=arguments.author)
            if any(_normalize(author) != author_key for author in entry.authors)
        ]
        return ExactCatalogResult(
            query=decision.query,
            tool=tool,
            status=ExactCatalogStatus.COMPLETED,
            arguments=arguments,
            message=f"**{arguments.author} has {len(coauthor_values)} catalog coauthors**:",
            coverage=CatalogCoverage(corpus_documents=total, matched_documents=len(entries)),
            coauthors=coauthor_values,
            documents=tuple(_document(entry) for entry in entries),
        )

    if tool is CatalogTool.AUTHOR_DOCUMENT_STATS:
        if arguments.author is None:
            raise RuntimeError("Validated authorship execution unexpectedly lacked an author.")
        stats = catalog.author_document_stats(arguments.author)
        return ExactCatalogResult(
            query=decision.query,
            tool=tool,
            status=ExactCatalogStatus.COMPLETED,
            arguments=arguments,
            message=f"Exact authorship statistics for **{arguments.author}**:",
            coverage=CatalogCoverage(corpus_documents=total, matched_documents=stats.credited_documents),
            authorship=_authorship(stats),
            documents=tuple(_document(entry) for entry in catalog.list_documents(author=arguments.author)),
        )

    entries = catalog.list_documents(author=arguments.author, document_type=arguments.document_type)
    return ExactCatalogResult(
        query=decision.query,
        tool=tool,
        status=ExactCatalogStatus.COMPLETED,
        arguments=arguments,
        message=f"The exact catalog contains **{len(entries)} matching documents**.",
        coverage=CatalogCoverage(corpus_documents=total, matched_documents=len(entries)),
        documents=tuple(_document(entry) for entry in entries),
    )


def _values_in_query(query: str, values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = _normalize(query)
    matches = [value for value in values if re.search(rf"(?<!\w){re.escape(_normalize(value))}(?!\w)", normalized)]
    return tuple(sorted(matches, key=lambda value: (-len(value), value.casefold())))


def _document_types_in_query(query: str, document_types: tuple[str, ...]) -> tuple[str, ...]:
    normalized = _normalize(query)
    matches: list[str] = []
    for document_type in document_types:
        singular = _normalize(document_type.replace("_", " "))
        aliases = {singular, f"{singular}s"}
        if singular.endswith("y"):
            aliases.add(f"{singular[:-1]}ies")
        if any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized) for alias in aliases):
            matches.append(document_type)
    return tuple(matches)


def _unknown_explicit_author(query: str, authors: tuple[str, ...]) -> str | None:
    if _values_in_query(query, authors):
        return None
    for pattern in _EXPLICIT_AUTHOR_PATTERNS:
        match = pattern.search(query)
        if match is not None:
            candidate = " ".join(match.group(1).strip(" ?.,'").split())
            return candidate or None
    return None


def _validated_default_author(default_author: str | None, authors: tuple[str, ...]) -> str | None:
    if default_author is None or not default_author.strip():
        return None
    key = _normalize(default_author)
    return next((author for author in authors if _normalize(author) == key), None)


def _is_generic_author_phrase(value: str) -> bool:
    return _normalize(value) in {
        "author",
        "the author",
        "this author",
        "writer",
        "the writer",
        "this writer",
        "they",
        "them",
        "their",
        "she",
        "her",
        "he",
        "his",
    }


def _author_alias_matches(candidate: str, authors: tuple[str, ...]) -> tuple[str, ...]:
    key = _normalize(candidate)
    if not key or " " in key:
        return ()
    return tuple(author for author in authors if key in _normalize(author).split())


def _document_description(count: int, arguments: ExactCatalogArguments) -> str:
    noun = arguments.document_type or "logical document"
    if count != 1:
        noun = f"{noun}s"
    if arguments.author is not None:
        return f"{noun} credited to {arguments.author}"
    return noun


def _document(entry: CatalogEntry) -> CatalogDocumentResult:
    return CatalogDocumentResult(
        document_id=entry.document_id,
        title=entry.title,
        authors=entry.authors,
        published_at=entry.published_at,
        document_type=entry.document_type,
        source_uris=entry.source_uris,
    )


def _authorship(stats: AuthorDocumentStats) -> CatalogAuthorshipResult:
    return CatalogAuthorshipResult(
        author=stats.author,
        credited_documents=stats.credited_documents,
        sole_authored_documents=stats.sole_authored_documents,
        coauthored_documents=stats.coauthored_documents,
    )


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())
