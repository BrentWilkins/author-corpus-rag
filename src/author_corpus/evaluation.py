"""Small, reproducible evaluation sets for semantic retrieval."""

from __future__ import annotations

from pathlib import Path

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
    has_passage_expectations: bool = False
    first_relevant_passage_rank: int | None = None

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
) -> RetrievalEvaluation:
    """Evaluate whether known relevant documents occur in the top results."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    results: list[RetrievalCaseResult] = []
    for case in cases:
        search_result = search.search(
            case.query,
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
        results.append(
            RetrievalCaseResult(
                name=case.name,
                query=case.query,
                retrieved_document_ids=retrieved_ids,
                relevant_document_ids=case.relevant_document_ids,
                first_relevant_rank=first_relevant_rank,
                has_passage_expectations=bool(case.relevant_passages),
                first_relevant_passage_rank=first_relevant_passage_rank,
            )
        )
    return RetrievalEvaluation(top_k=top_k, cases=tuple(results))


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
