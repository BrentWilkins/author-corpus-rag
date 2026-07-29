"""Small, reproducible evaluation sets for passage retrieval."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from author_corpus.retrieval import RetrievedPassage, SemanticCorpusSearch


class RelevantPassage(BaseModel):
    """Inspectable evidence requirements for one acceptable passage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    document_id: str
    contains: tuple[str, ...] = Field(min_length=1)
    section_path: tuple[str, ...] = ()
    minimum_document_author_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    minimum_quoted_speech_fraction: float | None = Field(default=None, ge=0.0, le=1.0)


class RetrievalCase(BaseModel):
    """One query with known relevant documents and optional exact evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    query: str
    relevant_document_ids: tuple[str, ...] = Field(min_length=1)
    relevant_passages: tuple[RelevantPassage, ...] = ()
    minimum_document_author_fraction: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_passage_documents(self) -> RetrievalCase:
        """Require every passage expectation to name a relevant document."""
        relevant_ids = set(self.relevant_document_ids)
        unknown_ids = {passage.document_id for passage in self.relevant_passages if passage.document_id not in relevant_ids}
        if unknown_ids:
            raise ValueError(f"Passage expectations reference non-relevant documents: {sorted(unknown_ids)}")
        return self


class RetrievalCaseResult(BaseModel):
    """Measured ranking behavior for one retrieval case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    query: str
    retrieved_document_ids: tuple[str, ...]
    relevant_document_ids: tuple[str, ...]
    first_relevant_rank: int | None
    normalized_discounted_cumulative_gain: float = Field(ge=0.0, le=1.0)
    has_passage_expectations: bool = False
    first_relevant_passage_rank: int | None = None
    passage_normalized_discounted_cumulative_gain: float | None = Field(default=None, ge=0.0, le=1.0)

    @property
    def hit(self) -> bool:
        """Return whether at least one relevant document was retrieved."""
        return self.first_relevant_rank is not None

    @property
    def reciprocal_rank(self) -> float:
        """Return reciprocal rank for the first relevant document."""
        return 0.0 if self.first_relevant_rank is None else 1.0 / self.first_relevant_rank

    @property
    def passage_hit(self) -> bool:
        """Return whether an expected passage—not merely its document—was retrieved."""
        return self.first_relevant_passage_rank is not None

    @property
    def passage_reciprocal_rank(self) -> float:
        """Return reciprocal rank for the first passage-level match."""
        return 0.0 if self.first_relevant_passage_rank is None else 1.0 / self.first_relevant_passage_rank


class RetrievalEvaluation(BaseModel):
    """Aggregate and per-case semantic retrieval measurements."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    top_k: int = Field(ge=1)
    cases: tuple[RetrievalCaseResult, ...]

    @property
    def hit_rate(self) -> float:
        """Return the fraction of cases with a relevant top-k result."""
        return 0.0 if not self.cases else sum(case.hit for case in self.cases) / len(self.cases)

    @property
    def mean_reciprocal_rank(self) -> float:
        """Return mean reciprocal rank across all cases."""
        return 0.0 if not self.cases else sum(case.reciprocal_rank for case in self.cases) / len(self.cases)

    @property
    def mean_normalized_discounted_cumulative_gain(self) -> float:
        """Return mean document-level nDCG across all cases."""
        return 0.0 if not self.cases else sum(case.normalized_discounted_cumulative_gain for case in self.cases) / len(self.cases)

    @property
    def passage_case_count(self) -> int:
        """Return the number of cases with passage-level expectations."""
        return sum(case.has_passage_expectations for case in self.cases)

    @property
    def passage_hit_rate(self) -> float | None:
        """Return passage hit rate, or None when no passages were labeled."""
        passage_cases = tuple(case for case in self.cases if case.has_passage_expectations)
        return None if not passage_cases else sum(case.passage_hit for case in passage_cases) / len(passage_cases)

    @property
    def passage_mean_reciprocal_rank(self) -> float | None:
        """Return passage MRR, or None when no passages were labeled."""
        passage_cases = tuple(case for case in self.cases if case.has_passage_expectations)
        if not passage_cases:
            return None
        return sum(case.passage_reciprocal_rank for case in passage_cases) / len(passage_cases)

    @property
    def passage_mean_normalized_discounted_cumulative_gain(self) -> float | None:
        """Return mean passage-level nDCG, or None when no passages were labeled."""
        values = tuple(
            case.passage_normalized_discounted_cumulative_gain
            for case in self.cases
            if case.passage_normalized_discounted_cumulative_gain is not None
        )
        return None if not values else sum(values) / len(values)


class RetrievalStrategyEvaluation(BaseModel):
    """One named strategy's metrics and measured evaluation latency."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: str = Field(min_length=1)
    elapsed_seconds: float = Field(ge=0.0)
    evaluation: RetrievalEvaluation


class RetrievalBenchmark(BaseModel):
    """Comparable evaluations for multiple retrieval strategies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategies: tuple[RetrievalStrategyEvaluation, ...] = Field(min_length=1)


class _RetrievalCaseFile(BaseModel):
    """Validated on-disk representation of retrieval cases."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cases: tuple[RetrievalCase, ...] = Field(min_length=1)


def load_retrieval_cases(path: str | Path) -> tuple[RetrievalCase, ...]:
    """Load and validate a YAML retrieval evaluation set."""
    value: object = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    return _RetrievalCaseFile.model_validate(value).cases


def evaluate_retrieval(
    search: SemanticCorpusSearch,
    cases: tuple[RetrievalCase, ...],
    *,
    top_k: int = 5,
    query_transform: Callable[[str], str] | None = None,
) -> RetrievalEvaluation:
    """Evaluate whether known relevant documents occur in the top results."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    results: list[RetrievalCaseResult] = []
    for case in cases:
        retrieval_query = case.query if query_transform is None else query_transform(case.query)
        search_result = search.search(
            retrieval_query,
            limit=top_k,
            minimum_document_author_fraction=case.minimum_document_author_fraction,
        )
        retrieved_ids = tuple(passage.document_id for passage in search_result.passages)
        relevant_ids = set(case.relevant_document_ids)
        first_relevant_rank = next(
            (rank for rank, document_id in enumerate(retrieved_ids, start=1) if document_id in relevant_ids),
            None,
        )
        first_relevant_passage_rank = next(
            (
                passage.rank
                for passage in search_result.passages
                if any(_passage_matches(passage, expected) for expected in case.relevant_passages)
            ),
            None,
        )
        document_relevance = _unique_document_relevance(retrieved_ids, relevant_ids)
        passage_relevance = _unique_passage_relevance(search_result.passages, case.relevant_passages)
        results.append(
            RetrievalCaseResult(
                name=case.name,
                query=case.query,
                retrieved_document_ids=retrieved_ids,
                relevant_document_ids=case.relevant_document_ids,
                first_relevant_rank=first_relevant_rank,
                normalized_discounted_cumulative_gain=_binary_ndcg(
                    document_relevance,
                    relevant_count=len(relevant_ids),
                ),
                has_passage_expectations=bool(case.relevant_passages),
                first_relevant_passage_rank=first_relevant_passage_rank,
                passage_normalized_discounted_cumulative_gain=(
                    _binary_ndcg(passage_relevance, relevant_count=len(case.relevant_passages))
                    if case.relevant_passages
                    else None
                ),
            )
        )
    return RetrievalEvaluation(top_k=top_k, cases=tuple(results))


def evaluate_retrieval_strategies(
    searches: Mapping[str, SemanticCorpusSearch],
    cases: tuple[RetrievalCase, ...],
    *,
    top_k: int = 5,
    query_transform: Callable[[str], str] | None = None,
) -> RetrievalBenchmark:
    """Time and evaluate multiple retrieval strategies on identical cases."""
    if not searches:
        raise ValueError("At least one retrieval strategy is required.")
    strategy_results: list[RetrievalStrategyEvaluation] = []
    for strategy, search in searches.items():
        started = perf_counter()
        evaluation = evaluate_retrieval(search, cases, top_k=top_k, query_transform=query_transform)
        strategy_results.append(
            RetrievalStrategyEvaluation(
                strategy=strategy,
                elapsed_seconds=perf_counter() - started,
                evaluation=evaluation,
            )
        )
    return RetrievalBenchmark(strategies=tuple(strategy_results))


def _passage_matches(passage: RetrievedPassage, expected: RelevantPassage) -> bool:
    if passage.document_id != expected.document_id:
        return False
    normalized_text = passage.text.casefold()
    if not all(fragment.casefold() in normalized_text for fragment in expected.contains):
        return False
    if expected.section_path and passage.section_path != expected.section_path:
        return False
    if expected.minimum_document_author_fraction is not None and (
        passage.document_author_fraction is None or passage.document_author_fraction < expected.minimum_document_author_fraction
    ):
        return False
    return not (
        expected.minimum_quoted_speech_fraction is not None
        and (passage.quoted_speech_fraction is None or passage.quoted_speech_fraction < expected.minimum_quoted_speech_fraction)
    )


def _binary_ndcg(relevance: tuple[bool, ...], *, relevant_count: int) -> float:
    if relevant_count < 1:
        return 0.0
    discounted_gain = sum(1.0 / math.log2(rank + 1) for rank, is_relevant in enumerate(relevance, start=1) if is_relevant)
    ideal_count = min(relevant_count, len(relevance))
    ideal_gain = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return 0.0 if ideal_gain == 0.0 else discounted_gain / ideal_gain


def _unique_document_relevance(retrieved_ids: tuple[str, ...], relevant_ids: set[str]) -> tuple[bool, ...]:
    matched: set[str] = set()
    relevance: list[bool] = []
    for document_id in retrieved_ids:
        is_new_match = document_id in relevant_ids and document_id not in matched
        relevance.append(is_new_match)
        if is_new_match:
            matched.add(document_id)
    return tuple(relevance)


def _unique_passage_relevance(
    passages: tuple[RetrievedPassage, ...],
    expected_passages: tuple[RelevantPassage, ...],
) -> tuple[bool, ...]:
    unmatched = set(range(len(expected_passages)))
    relevance: list[bool] = []
    for passage in passages:
        matched_index = next(
            (index for index in sorted(unmatched) if _passage_matches(passage, expected_passages[index])),
            None,
        )
        relevance.append(matched_index is not None)
        if matched_index is not None:
            unmatched.remove(matched_index)
    return tuple(relevance)
