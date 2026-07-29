"""One auditable execution service for routed corpus questions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from author_corpus.answering import GroundedAnswer, GroundedAnswerer
from author_corpus.catalog import CorpusCatalog
from author_corpus.exact import ExactCatalogResult, execute_catalog_query
from author_corpus.hybrid import RetrievalProfiles
from author_corpus.identity import AuthorIdentity, AuthorQueryResolution, resolve_author_query
from author_corpus.local_llm import LocalModelSettings
from author_corpus.reasoning import BoundedReasoningEngine, StructuredSemanticClaimVerifier
from author_corpus.reasoning_models import ReasoningTrace
from author_corpus.retrieval import SemanticCorpusSearch, SemanticSearchResult
from author_corpus.routing import QueryRoute, QueryRouteDecision, route_query
from author_corpus.scope import AuthorScope
from author_corpus.tracing import (
    GenerationTraceSettings,
    QueryTrace,
    QueryTraceStore,
    RetrievalTraceSettings,
)

VerifierMode = Literal["conservative", "semantic"]


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
    author_resolution: AuthorQueryResolution | None = None
    exact: ExactCatalogResult | None = None
    semantic: SemanticSearchResult | None = None
    grounded_answer: GroundedAnswer | None = None
    reasoning: ReasoningTrace | None = None
    trace_id: str | None = None
    timings: tuple[QueryTiming, ...]

    @model_validator(mode="after")
    def validate_route_artifacts(self) -> CorpusQueryResult:
        """Prevent exact and semantic execution artifacts from being mixed."""
        if self.decision.route is QueryRoute.EXACT_CATALOG:
            if (
                self.exact is None
                or self.author_resolution is not None
                or self.semantic is not None
                or self.grounded_answer is not None
                or self.reasoning is not None
            ):
                raise ValueError("Exact results require only an exact catalog artifact.")
            if self.retrieval_query != self.query:
                raise ValueError("Exact results cannot use a contextualized retrieval query.")
        elif self.exact is not None or self.semantic is None or self.author_resolution is None:
            raise ValueError("Semantic results require retrieval and cannot include an exact artifact.")
        if (
            self.grounded_answer is not None
            and self.semantic is not None
            and self.grounded_answer.evidence != self.semantic.passages
        ):
            raise ValueError("A grounded answer must use the exact passages already returned by retrieval.")
        if self.trace_id is not None and self.grounded_answer is None:
            raise ValueError("Only a grounded answer can have a durable query trace.")
        if self.reasoning is not None and self.grounded_answer is None:
            raise ValueError("Reasoning metadata requires a grounded answer.")
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
        span_text = ""
        if self.grounded_answer is not None and self.grounded_answer.cited_evidence_numbers:
            covered, total = self.grounded_answer.cited_span_coverage
            span_text = f" · exact evidence spans {covered}/{total}"
        reasoning_text = ""
        if self.reasoning is not None:
            verifier_ids = tuple(
                dict.fromkeys(
                    verification.verifier_id for round_ in self.reasoning.rounds for verification in round_.verifications
                )
            )
            verifier_text = ", ".join(verifier_ids) if verifier_ids else "no extracted claims"
            reasoning_text = f" · bounded reasoning {len(self.reasoning.rounds)} round(s) · verifier `{verifier_text}`"
        identity_text = ""
        if self.author_resolution is not None and self.author_resolution.changed:
            identity_text = "\n\n_Retrieval resolved configured author references to the corpus-author identity._"
        return (
            f"{body}{identity_text}\n\n---\n"
            f"Route: `{self.decision.route.value}` · {timing_text}{span_text}{reasoning_text}{trace_text}"
        )


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
        author_aliases: tuple[str, ...] = (),
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
        self.default_scope = (
            AuthorScope.for_author(default_author) if default_author is not None and default_author.strip() else AuthorScope()
        )
        self.author_identity = (
            AuthorIdentity.from_catalog(
                default_author=default_author,
                aliases=author_aliases,
                catalog_authors=(name for name, _ in catalog.author_counts()),
            )
            if default_author is not None and default_author.strip()
            else None
        )
        self.minimum_document_author_fraction = minimum_document_author_fraction
        self.trace_context = trace_context

    def ask(
        self,
        query: str,
        *,
        retrieval_query: str | None = None,
        generate: bool = True,
        reason: bool = False,
        verifier: VerifierMode = "conservative",
    ) -> CorpusQueryResult:
        """Execute one question without allowing context to alter exact routing."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Query must not be empty.")
        if verifier not in {"conservative", "semantic"}:
            raise ValueError(f"Unknown verifier mode: {verifier!r}.")

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

        contextualized_query = (retrieval_query or normalized_query).strip()
        if not contextualized_query:
            raise ValueError("Retrieval query must not be empty.")
        identity_started = perf_counter()
        author_resolution = resolve_author_query(contextualized_query, self.author_identity)
        effective_query = author_resolution.retrieval_query
        timings.append(QueryTiming(label="author identity", elapsed_seconds=perf_counter() - identity_started))
        search = self._search_for(decision.route)
        answer: GroundedAnswer | None = None
        reasoning: ReasoningTrace | None = None
        trace_id: str | None = None
        answer_elapsed_seconds = 0.0
        if reason and generate and self.complete is not None and self.model_id is not None:
            reasoning_started = perf_counter()
            reasoned = BoundedReasoningEngine(
                search,
                self.complete,
                model_id=self.model_id,
                verifier=StructuredSemanticClaimVerifier(self.complete) if verifier == "semantic" else None,
            ).answer(
                author_resolution.grounding_question,
                retrieval_question=effective_query,
                author_scope=self.default_scope,
                minimum_document_author_fraction=self.minimum_document_author_fraction,
            )
            semantic = reasoned.search_result
            answer = reasoned.grounded_answer
            reasoning = reasoned.reasoning
            answer_elapsed_seconds = perf_counter() - reasoning_started
            timings.extend(
                (
                    QueryTiming(
                        label="reasoning retrieval",
                        elapsed_seconds=sum(round_.retrieval_seconds for round_ in reasoning.rounds),
                    ),
                    QueryTiming(
                        label="reasoning generation",
                        elapsed_seconds=sum(round_.generation_seconds for round_ in reasoning.rounds),
                    ),
                    QueryTiming(
                        label="claim verification",
                        elapsed_seconds=sum(round_.verification_seconds for round_ in reasoning.rounds),
                    ),
                )
            )
            timings.append(QueryTiming(label="bounded reasoning", elapsed_seconds=answer_elapsed_seconds))
        else:
            retrieval_started = perf_counter()
            semantic = search.search(
                effective_query,
                minimum_document_author_fraction=self.minimum_document_author_fraction,
            )
            timings.append(QueryTiming(label="retrieval", elapsed_seconds=perf_counter() - retrieval_started))

        if generate and answer is None and self.complete is not None and self.model_id is not None:
            generation_started = perf_counter()
            answer = GroundedAnswerer(
                search,
                self.complete,
                model_id=self.model_id,
                evidence_limit=search.default_limit,
            ).answer_from_search_result(semantic, question=author_resolution.grounding_question)
            generation_timing = QueryTiming(label="generation", elapsed_seconds=perf_counter() - generation_started)
            answer_elapsed_seconds = generation_timing.elapsed_seconds
            timings.append(generation_timing)
        if answer is not None and self.trace_context is not None:
            trace_started = perf_counter()
            trace = QueryTrace.from_grounded_answer(
                answer,
                corpus_fingerprint=self.trace_context.corpus_fingerprint,
                document_content_hashes=self.trace_context.document_content_hashes,
                retrieval=RetrievalTraceSettings(
                    evidence_limit=search.default_limit,
                    max_passages_per_document=search.max_passages_per_document,
                    minimum_document_author_fraction=self.minimum_document_author_fraction,
                    strategy=semantic.strategy,
                    score_kind=semantic.score_kind,
                ),
                generation=GenerationTraceSettings.from_local_settings(
                    self.trace_context.generation_settings,
                    prompt_version=answer.prompt_version,
                ),
                elapsed_seconds=answer_elapsed_seconds,
                user_query=normalized_query,
                author_scope=self.default_scope,
                reasoning=reasoning,
            )
            self.trace_context.store.put(trace)
            trace_id = trace.trace_id
            timings.append(QueryTiming(label="trace persistence", elapsed_seconds=perf_counter() - trace_started))

        return CorpusQueryResult(
            query=normalized_query,
            retrieval_query=effective_query,
            decision=decision,
            author_resolution=author_resolution,
            semantic=semantic,
            grounded_answer=answer,
            reasoning=reasoning,
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
