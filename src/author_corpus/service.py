"""One auditable execution service for routed corpus questions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field, model_validator

from author_corpus.answering import GroundedAnswer, GroundedAnswerer
from author_corpus.catalog import CorpusCatalog
from author_corpus.exact import ExactCatalogResult, execute_catalog_query
from author_corpus.hybrid import RetrievalProfiles
from author_corpus.local_llm import LocalModelSettings
from author_corpus.retrieval import SemanticCorpusSearch, SemanticSearchResult
from author_corpus.routing import QueryRoute, QueryRouteDecision, route_query
from author_corpus.tracing import (
    GenerationTraceSettings,
    QueryTrace,
    QueryTraceStore,
    RetrievalTraceSettings,
)


class QueryTiming(BaseModel):
    """Elapsed wall-clock time for one query phase."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(min_length=1)
    elapsed_seconds: float = Field(ge=0.0)


@dataclass(frozen=True, slots=True)
class QueryTraceContext:
    """Local persistence inputs required to freeze a generated answer."""

    store: QueryTraceStore
    corpus_fingerprint: str
    document_content_hashes: Mapping[str, str]
    generation_settings: LocalModelSettings


class CorpusQueryResult(BaseModel):
    """One routed result with exactly the artifacts its path produced."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=1)
    retrieval_query: str = Field(min_length=1)
    decision: QueryRouteDecision
    exact: ExactCatalogResult | None = None
    semantic: SemanticSearchResult | None = None
    grounded_answer: GroundedAnswer | None = None
    trace_id: str | None = None
    timings: tuple[QueryTiming, ...]

    @model_validator(mode="after")
    def validate_route_artifacts(self) -> CorpusQueryResult:
        """Prevent exact and semantic execution artifacts from being mixed."""
        if self.decision.route is QueryRoute.EXACT_CATALOG:
            if self.exact is None or self.semantic is not None or self.grounded_answer is not None:
                raise ValueError("Exact results require only an exact catalog artifact.")
            if self.retrieval_query != self.query:
                raise ValueError("Exact results cannot use a contextualized retrieval query.")
        elif self.exact is not None or self.semantic is None:
            raise ValueError("Semantic results require retrieval and cannot include an exact artifact.")
        if (
            self.grounded_answer is not None
            and self.semantic is not None
            and self.grounded_answer.evidence != self.semantic.passages
        ):
            raise ValueError("A grounded answer must use the exact passages already returned by retrieval.")
        if self.trace_id is not None and self.grounded_answer is None:
            raise ValueError("Only a grounded answer can have a durable query trace.")
        return self

    def to_markdown(self) -> str:
        """Render the answer, sources, route, and phase timings."""
        if self.exact is not None:
            body = self.exact.to_markdown()
        elif self.grounded_answer is not None:
            body = self.grounded_answer.to_markdown()
        elif self.semantic is not None:
            body = _semantic_preview(self.semantic)
        else:
            raise RuntimeError("Validated query result unexpectedly lacked an artifact.")

        timing_text = ", ".join(f"{timing.label} {timing.elapsed_seconds:.3f}s" for timing in self.timings)
        trace_text = "" if self.trace_id is None else f" · trace `{self.trace_id}`"
        return f"{body}\n\n---\nRoute: `{self.decision.route.value}` · {timing_text}{trace_text}"


class CorpusQueryService:
    """Route, execute, retrieve once, and optionally generate one answer."""

    def __init__(
        self,
        catalog: CorpusCatalog,
        retrieval_profiles: RetrievalProfiles,
        *,
        complete: Callable[[str], str] | None = None,
        model_id: str | None = None,
        default_author: str | None = None,
        minimum_document_author_fraction: float | None = 0.8,
        trace_context: QueryTraceContext | None = None,
    ) -> None:
        """Initialize a reusable query service with explicit generation settings."""
        if (complete is None) != (model_id is None):
            raise ValueError("complete and model_id must either both be provided or both be omitted.")
        if minimum_document_author_fraction is not None and not 0.0 <= minimum_document_author_fraction <= 1.0:
            raise ValueError("minimum_document_author_fraction must be between 0 and 1.")
        self.catalog = catalog
        self.retrieval_profiles = retrieval_profiles
        self.complete = complete
        self.model_id = model_id
        self.default_author = default_author
        self.minimum_document_author_fraction = minimum_document_author_fraction
        self.trace_context = trace_context

    def ask(
        self,
        query: str,
        *,
        retrieval_query: str | None = None,
        generate: bool = True,
    ) -> CorpusQueryResult:
        """Execute one question without allowing context to alter exact routing."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Query must not be empty.")

        route_started = perf_counter()
        decision = route_query(normalized_query)
        timings = [QueryTiming(label="routing", elapsed_seconds=perf_counter() - route_started)]
        if decision.route is QueryRoute.EXACT_CATALOG:
            exact_started = perf_counter()
            exact = execute_catalog_query(self.catalog, decision, default_author=self.default_author)
            timings.append(QueryTiming(label="exact catalog", elapsed_seconds=perf_counter() - exact_started))
            return CorpusQueryResult(
                query=normalized_query,
                retrieval_query=normalized_query,
                decision=decision,
                exact=exact,
                timings=tuple(timings),
            )

        effective_query = (retrieval_query or normalized_query).strip()
        if not effective_query:
            raise ValueError("Retrieval query must not be empty.")
        search = self._search_for(decision.route)
        retrieval_started = perf_counter()
        semantic = search.search(
            effective_query,
            minimum_document_author_fraction=self.minimum_document_author_fraction,
        )
        timings.append(QueryTiming(label="retrieval", elapsed_seconds=perf_counter() - retrieval_started))

        answer: GroundedAnswer | None = None
        trace_id: str | None = None
        if generate and self.complete is not None and self.model_id is not None:
            generation_started = perf_counter()
            answer = GroundedAnswerer(
                search,
                self.complete,
                model_id=self.model_id,
                evidence_limit=search.default_limit,
            ).answer_from_search_result(semantic)
            generation_timing = QueryTiming(label="generation", elapsed_seconds=perf_counter() - generation_started)
            timings.append(generation_timing)
            if self.trace_context is not None:
                trace_started = perf_counter()
                trace = QueryTrace.from_grounded_answer(
                    answer,
                    corpus_fingerprint=self.trace_context.corpus_fingerprint,
                    document_content_hashes=self.trace_context.document_content_hashes,
                    retrieval=RetrievalTraceSettings(
                        evidence_limit=search.default_limit,
                        max_passages_per_document=search.max_passages_per_document,
                        minimum_document_author_fraction=self.minimum_document_author_fraction,
                        strategy=search.strategy,
                        score_kind=search.score_kind,
                    ),
                    generation=GenerationTraceSettings.from_local_settings(
                        self.trace_context.generation_settings,
                        prompt_version=answer.prompt_version,
                    ),
                    elapsed_seconds=generation_timing.elapsed_seconds,
                    user_query=normalized_query,
                )
                self.trace_context.store.put(trace)
                trace_id = trace.trace_id
                timings.append(QueryTiming(label="trace persistence", elapsed_seconds=perf_counter() - trace_started))

        return CorpusQueryResult(
            query=normalized_query,
            retrieval_query=effective_query,
            decision=decision,
            semantic=semantic,
            grounded_answer=answer,
            trace_id=trace_id,
            timings=tuple(timings),
        )

    def _search_for(self, route: QueryRoute) -> SemanticCorpusSearch:
        if route is QueryRoute.BROAD_DISCOVERY:
            return self.retrieval_profiles.discovery
        if route is QueryRoute.FOCUSED_EVIDENCE:
            return self.retrieval_profiles.hybrid_evidence
        raise ValueError(f"No semantic search profile exists for route {route}.")


def _semantic_preview(result: SemanticSearchResult) -> str:
    if not result.passages:
        return "The retrieval step returned no passages."
    lines = ["Generation is disabled. Retrieved evidence:"]
    for passage in result.passages:
        title = passage.title.replace("[", r"\[").replace("]", r"\]")
        if passage.canonical_source_uri:
            lines.append(f"- [{title}]({passage.canonical_source_uri}): {passage.text[:240].replace(chr(10), ' ')}")
        else:
            lines.append(f"- {title}: {passage.text[:240].replace(chr(10), ' ')}")
    return "\n".join(lines)
