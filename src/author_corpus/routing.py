"""Auditable routing between exact catalog and semantic query paths."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class QueryRoute(StrEnum):
    """Stable execution paths available to a corpus question."""

    EXACT_CATALOG = "exact_catalog"
    BROAD_DISCOVERY = "broad_discovery"
    FOCUSED_EVIDENCE = "focused_evidence"


class CatalogTool(StrEnum):
    """Exact catalog tools a routing decision may recommend."""

    COUNT_DOCUMENTS = "count_corpus_documents"
    LIST_AUTHORS = "list_corpus_authors"
    LIST_COAUTHORS = "list_author_coauthors"
    AUTHOR_DOCUMENT_STATS = "get_author_document_stats"
    LIST_DOCUMENTS = "list_corpus_documents"


class QueryRouteDecision(BaseModel):
    """One inspectable routing decision without an invented confidence score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=1)
    route: QueryRoute
    rule_id: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    matched_signals: tuple[str, ...] = ()
    catalog_tool: CatalogTool | None = None
    exhaustive: bool = False

    @model_validator(mode="after")
    def validate_catalog_route(self) -> QueryRouteDecision:
        """Require an exact tool only for exhaustive catalog decisions."""
        if self.route is QueryRoute.EXACT_CATALOG:
            if self.catalog_tool is None or not self.exhaustive:
                raise ValueError("Exact catalog routes require a catalog tool and exhaustive coverage.")
        elif self.catalog_tool is not None or self.exhaustive:
            raise ValueError("Semantic routes must not claim a catalog tool or exhaustive coverage.")
        return self


class RoutingCase(BaseModel):
    """One labeled natural-language question for router evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_route: QueryRoute
    expected_catalog_tool: CatalogTool | None = None

    @model_validator(mode="after")
    def validate_expected_tool(self) -> RoutingCase:
        """Keep exact route labels paired with a concrete catalog operation."""
        if self.expected_route is QueryRoute.EXACT_CATALOG and self.expected_catalog_tool is None:
            raise ValueError("Exact routing cases require expected_catalog_tool.")
        if self.expected_route is not QueryRoute.EXACT_CATALOG and self.expected_catalog_tool is not None:
            raise ValueError("Semantic routing cases cannot expect a catalog tool.")
        return self


class RoutingCaseResult(BaseModel):
    """Predicted route and correctness for one labeled question."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    query: str
    expected_route: QueryRoute
    predicted_route: QueryRoute
    expected_catalog_tool: CatalogTool | None = None
    predicted_catalog_tool: CatalogTool | None = None
    rule_id: str

    @property
    def route_correct(self) -> bool:
        """Return whether the execution path is correct."""
        return self.predicted_route is self.expected_route

    @property
    def catalog_tool_correct(self) -> bool:
        """Return whether an exact route also selected the expected tool."""
        return self.route_correct and self.predicted_catalog_tool is self.expected_catalog_tool


class RouteMetrics(BaseModel):
    """Per-route support, precision, and recall."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    route: QueryRoute
    support: int = Field(ge=0)
    predicted: int = Field(ge=0)
    true_positives: int = Field(ge=0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)


class RoutingEvaluation(BaseModel):
    """Aggregate routing quality with individual inspectable decisions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cases: tuple[RoutingCaseResult, ...]
    route_metrics: tuple[RouteMetrics, ...]
    confusion: dict[str, dict[str, int]]

    @property
    def accuracy(self) -> float:
        """Return route accuracy across all labeled questions."""
        return 0.0 if not self.cases else sum(case.route_correct for case in self.cases) / len(self.cases)

    @property
    def exact_tool_accuracy(self) -> float | None:
        """Return exact-tool accuracy among catalog-labeled questions."""
        exact_cases = tuple(case for case in self.cases if case.expected_route is QueryRoute.EXACT_CATALOG)
        return None if not exact_cases else sum(case.catalog_tool_correct for case in exact_cases) / len(exact_cases)


class _RoutingCaseFile(BaseModel):
    """Validated on-disk representation of routing cases."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cases: tuple[RoutingCase, ...] = Field(min_length=1)


QueryRouter = Callable[[str], QueryRouteDecision]

_COUNT_SIGNAL = re.compile(r"\b(how many|count|number of|total(?: number)? of)\b")
_EXHAUSTIVE_SIGNAL = re.compile(r"\b(all|every|complete|entire|exhaustive(?:ly)?|full list)\b")
_EXPLICIT_LIST_SIGNAL = re.compile(r"\b(list|name)\b")
_WHO_ARE_AUTHORS = re.compile(r"\bwho (?:are|were)\b.+\b(authors?|writers?|contributors?)\b")
_DOCUMENT_TERM = re.compile(
    r"\b(article|articles|document|documents|work|works|post|posts|paper|papers|piece|pieces|story|stories)\b"
)
_CONTENT_FILTER = re.compile(r"\b(about|contain|contains|discuss|discusses|mention|mentions|cover|covers|address|addresses)\b")
_AUTHOR_TERM = re.compile(r"\b(author|authors|writer|writers|contributor|contributors|byline|bylines)\b")
_COAUTHOR_TERM = re.compile(r"\b(coauthor|coauthors|co-author|co-authors)\b")
_AUTHORSHIP_BREAKDOWN = re.compile(r"\b(sole[- ]authored|coauthored|co-authored|authorship breakdown|writing credits?)\b")
_SOURCE_INVENTORY = re.compile(
    r"\b(source urls?|source uris?|publication links?|canonical links?|where (?:was|were).+published)\b"
)
_BROAD_DISCOVERY_SIGNALS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "corpus overview",
        re.compile(r"\b(overview|survey|map|explore)\b.+\b(corpus|collection|writing|work|articles|documents)\b"),
    ),
    ("topics or themes", re.compile(r"\b(topic|topics|theme|themes|subject|subjects)\b")),
    ("style across works", re.compile(r"\b(writing style|voice|tone|stylistic|patterns? across|tendenc(?:y|ies))\b")),
    ("broad author coverage", re.compile(r"\bwhat does (?:the|this|an?) author (?:write|talk|care) about\b")),
    ("broad advice", re.compile(r"\b(practical advice|recommendations? across|advice.+readers)\b")),
    ("browse examples", re.compile(r"\b(find|show|recommend|give)\b.+\b(some|several|examples?|articles|posts|pieces)\b")),
    (
        "multiple personal facts",
        re.compile(
            r"\b(?:two|three|four|five|\d+)\s+"
            r"(?:(?:favorite|favourite|personal|biographical)\s+)?"
            r"(?:things?|facts?|details?|interests?|preferences?)\b"
        ),
    ),
)


def route_query(query: str) -> QueryRouteDecision:
    """Route a question using conservative, source-agnostic deterministic rules."""
    normalized = " ".join(query.casefold().split())
    if not normalized:
        raise ValueError("Query must not be empty.")

    coauthor_match = _COAUTHOR_TERM.search(normalized)
    if coauthor_match:
        if _AUTHORSHIP_BREAKDOWN.search(normalized) or _COUNT_SIGNAL.search(normalized):
            return _exact_decision(query, "catalog.authorship_stats", CatalogTool.AUTHOR_DOCUMENT_STATS, coauthor_match.group())
        return _exact_decision(query, "catalog.coauthors", CatalogTool.LIST_COAUTHORS, coauthor_match.group())

    authorship_match = _AUTHORSHIP_BREAKDOWN.search(normalized)
    if authorship_match:
        return _exact_decision(query, "catalog.authorship_stats", CatalogTool.AUTHOR_DOCUMENT_STATS, authorship_match.group())

    author_match = _AUTHOR_TERM.search(normalized)
    count_match = _COUNT_SIGNAL.search(normalized)
    exhaustive_match = _EXHAUSTIVE_SIGNAL.search(normalized)
    explicit_list_match = _EXPLICIT_LIST_SIGNAL.search(normalized)
    who_are_authors_match = _WHO_ARE_AUTHORS.search(normalized)
    document_match = _DOCUMENT_TERM.search(normalized)
    content_match = _CONTENT_FILTER.search(normalized)

    if document_match and content_match and (count_match or exhaustive_match or explicit_list_match or "which" in normalized):
        signals = _matched(document_match, content_match, count_match, exhaustive_match, explicit_list_match)
        return _semantic_decision(
            query,
            QueryRoute.BROAD_DISCOVERY,
            "semantic.content_inventory",
            "The question inventories document content, which metadata cannot answer exhaustively; "
            "use diverse discovery and report that limit.",
            *signals,
        )

    if document_match and count_match:
        return _exact_decision(
            query, "catalog.document_count", CatalogTool.COUNT_DOCUMENTS, document_match.group(), count_match.group()
        )

    if author_match and (count_match or exhaustive_match or explicit_list_match or who_are_authors_match):
        signals = _matched(author_match, count_match, exhaustive_match, explicit_list_match, who_are_authors_match)
        return _exact_decision(query, "catalog.authors", CatalogTool.LIST_AUTHORS, *signals)

    if document_match and (exhaustive_match or explicit_list_match):
        signals = _matched(document_match, exhaustive_match, explicit_list_match)
        return _exact_decision(query, "catalog.document_list", CatalogTool.LIST_DOCUMENTS, *signals)

    source_match = _SOURCE_INVENTORY.search(normalized)
    if source_match:
        return _exact_decision(query, "catalog.source_inventory", CatalogTool.LIST_DOCUMENTS, source_match.group())

    for signal_name, pattern in _BROAD_DISCOVERY_SIGNALS:
        match = pattern.search(normalized)
        if match:
            return _semantic_decision(
                query,
                QueryRoute.BROAD_DISCOVERY,
                f"semantic.discovery.{signal_name.replace(' ', '_')}",
                "The question seeks breadth or representative examples, so use the diverse "
                "one-passage-per-document discovery profile.",
                match.group(),
            )

    return _semantic_decision(
        query,
        QueryRoute.FOCUSED_EVIDENCE,
        "semantic.focused_default",
        "No exact-metadata or broad-discovery rule matched; use hybrid retrieval for focused, source-grounded evidence.",
    )


def load_routing_cases(path: str | Path) -> tuple[RoutingCase, ...]:
    """Load and validate a YAML routing evaluation set."""
    value: object = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    return _RoutingCaseFile.model_validate(value).cases


def evaluate_query_router(cases: tuple[RoutingCase, ...], *, router: QueryRouter = route_query) -> RoutingEvaluation:
    """Evaluate route and catalog-tool decisions over labeled questions."""
    results = tuple(_evaluate_case(case, router(case.query)) for case in cases)
    confusion_counter = Counter((case.expected_route, case.predicted_route) for case in results)
    confusion = {
        expected.value: {predicted.value: confusion_counter[(expected, predicted)] for predicted in QueryRoute}
        for expected in QueryRoute
    }
    metrics = tuple(_route_metrics(route, results) for route in QueryRoute)
    return RoutingEvaluation(cases=results, route_metrics=metrics, confusion=confusion)


def _evaluate_case(case: RoutingCase, decision: QueryRouteDecision) -> RoutingCaseResult:
    return RoutingCaseResult(
        name=case.name,
        query=case.query,
        expected_route=case.expected_route,
        predicted_route=decision.route,
        expected_catalog_tool=case.expected_catalog_tool,
        predicted_catalog_tool=decision.catalog_tool,
        rule_id=decision.rule_id,
    )


def _route_metrics(route: QueryRoute, cases: tuple[RoutingCaseResult, ...]) -> RouteMetrics:
    support = sum(case.expected_route is route for case in cases)
    predicted = sum(case.predicted_route is route for case in cases)
    true_positives = sum(case.expected_route is route and case.predicted_route is route for case in cases)
    return RouteMetrics(
        route=route,
        support=support,
        predicted=predicted,
        true_positives=true_positives,
        precision=0.0 if predicted == 0 else true_positives / predicted,
        recall=0.0 if support == 0 else true_positives / support,
    )


def _exact_decision(query: str, rule_id: str, catalog_tool: CatalogTool, *signals: str) -> QueryRouteDecision:
    return QueryRouteDecision(
        query=query.strip(),
        route=QueryRoute.EXACT_CATALOG,
        rule_id=rule_id,
        rationale=(
            "The question asks for exact corpus metadata, so use the exhaustive SQLite catalog rather than top-k retrieval."
        ),
        matched_signals=tuple(signals),
        catalog_tool=catalog_tool,
        exhaustive=True,
    )


def _semantic_decision(
    query: str,
    route: QueryRoute,
    rule_id: str,
    rationale: str,
    *signals: str,
) -> QueryRouteDecision:
    return QueryRouteDecision(
        query=query.strip(),
        route=route,
        rule_id=rule_id,
        rationale=rationale,
        matched_signals=tuple(signals),
    )


def _matched(*matches: re.Match[str] | None) -> tuple[str, ...]:
    return tuple(match.group() for match in matches if match is not None)
