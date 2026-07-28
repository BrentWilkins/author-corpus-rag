"""Small, reproducible evaluation sets for semantic retrieval."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from author_corpus.retrieval import SemanticCorpusSearch


class RetrievalCase(BaseModel):
    """One query with known relevant logical documents."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    query: str
    relevant_document_ids: tuple[str, ...] = Field(min_length=1)


class RetrievalCaseResult(BaseModel):
    """Measured ranking behavior for one retrieval case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    query: str
    retrieved_document_ids: tuple[str, ...]
    relevant_document_ids: tuple[str, ...]
    first_relevant_rank: int | None

    @property
    def hit(self) -> bool:
        """Return whether at least one relevant document was retrieved."""
        return self.first_relevant_rank is not None

    @property
    def reciprocal_rank(self) -> float:
        """Return reciprocal rank for the first relevant document."""
        return 0.0 if self.first_relevant_rank is None else 1.0 / self.first_relevant_rank


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
        search_result = search.search(case.query, limit=top_k)
        retrieved_ids = tuple(passage.document_id for passage in search_result.passages)
        relevant_ids = set(case.relevant_document_ids)
        first_relevant_rank = next(
            (rank for rank, document_id in enumerate(retrieved_ids, start=1) if document_id in relevant_ids),
            None,
        )
        results.append(
            RetrievalCaseResult(
                name=case.name,
                query=case.query,
                retrieved_document_ids=retrieved_ids,
                relevant_document_ids=case.relevant_document_ids,
                first_relevant_rank=first_relevant_rank,
            )
        )
    return RetrievalEvaluation(top_k=top_k, cases=tuple(results))
