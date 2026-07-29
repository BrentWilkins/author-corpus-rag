"""Inspectable passage retrieval with document-level provenance."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from typing import Literal

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import MetadataMode, NodeWithScore
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

ScoreKind = Literal["cosine_similarity", "bm25", "reciprocal_rank_fusion", "unknown"]

_RETRIEVAL_CONTRIBUTIONS_KEY = "_retrieval_contributions"


class RetrievalContribution(BaseModel):
    """One retrieval method's inspectable contribution to a candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: str = Field(min_length=1)
    rank: int = Field(ge=1)
    score: float | None = None
    score_kind: ScoreKind = "unknown"
    rrf_contribution: float | None = Field(default=None, ge=0.0)


class RetrievedPassage(BaseModel):
    """One semantically retrieved passage and its source metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int = Field(ge=1)
    score: float | None = None
    score_kind: ScoreKind = "unknown"
    retrieval_contributions: tuple[RetrievalContribution, ...] = ()
    document_id: str
    title: str
    authors: tuple[str, ...] = ()
    published_at: str | None = None
    document_type: str
    source_uris: tuple[str, ...] = ()
    canonical_source_uri: str | None = None
    section_path: tuple[str, ...] = ()
    passage_voice: Literal["document_author", "quoted_speech", "uncertain", "mixed", "unknown"] = "unknown"
    document_author_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    quoted_speech_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertain_voice_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    attributed_speakers: tuple[str, ...] = ()
    text: str


class SemanticSearchResult(BaseModel):
    """A non-exhaustive semantic search result with explicit coverage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    strategy: str = "semantic"
    score_kind: ScoreKind = "unknown"
    passages: tuple[RetrievedPassage, ...]
    inspected_candidates: int = Field(ge=0)
    discarded_by_voice_filter: int = Field(default=0, ge=0)
    exhaustive: bool = False

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Reject an empty semantic query."""
        query = value.strip()
        if not query:
            raise ValueError("Semantic query must not be empty.")
        return query


class SemanticCorpusSearch:
    """Search ranked candidates while preserving inspectable evidence."""

    def __init__(
        self,
        retriever: BaseRetriever,
        *,
        default_limit: int = 5,
        max_passages_per_document: int = 1,
        strategy: str = "dense",
        score_kind: ScoreKind = "cosine_similarity",
    ) -> None:
        """Initialize search over a configured LlamaIndex retriever."""
        if default_limit < 1:
            raise ValueError("default_limit must be at least 1.")
        if max_passages_per_document < 1:
            raise ValueError("max_passages_per_document must be at least 1.")
        if not strategy.strip():
            raise ValueError("strategy must not be empty.")
        self.retriever = retriever
        self.default_limit = default_limit
        self.max_passages_per_document = max_passages_per_document
        self.strategy = strategy.strip()
        self.score_kind = score_kind

    def search(
        self,
        query: str,
        *,
        limit: int | None = None,
        minimum_document_author_fraction: float | None = None,
    ) -> SemanticSearchResult:
        """Return the highest-ranked passages without claiming full coverage."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Semantic query must not be empty.")
        result_limit = self.default_limit if limit is None else limit
        if result_limit < 1:
            raise ValueError("limit must be at least 1.")
        if minimum_document_author_fraction is not None and not 0.0 <= minimum_document_author_fraction <= 1.0:
            raise ValueError("minimum_document_author_fraction must be between 0 and 1.")

        candidates = self.retriever.retrieve(normalized_query)
        document_counts: Counter[str] = Counter()
        passages: list[RetrievedPassage] = []
        discarded_by_voice_filter = 0
        for candidate_rank, candidate in enumerate(candidates, start=1):
            document_id = _document_id(candidate)
            if document_counts[document_id] >= self.max_passages_per_document:
                continue
            passage = _to_passage(
                candidate,
                rank=len(passages) + 1,
                document_id=document_id,
                candidate_rank=candidate_rank,
                default_method=self.strategy,
                default_score_kind=self.score_kind,
            )
            if minimum_document_author_fraction is not None and (
                passage.document_author_fraction is None or passage.document_author_fraction < minimum_document_author_fraction
            ):
                discarded_by_voice_filter += 1
                continue
            document_counts[document_id] += 1
            passages.append(passage)
            if len(passages) == result_limit:
                break

        return SemanticSearchResult(
            query=normalized_query,
            strategy=self.strategy,
            score_kind=self.score_kind,
            passages=tuple(passages),
            inspected_candidates=len(candidates),
            discarded_by_voice_filter=discarded_by_voice_filter,
        )


def _to_passage(
    candidate: NodeWithScore,
    *,
    rank: int,
    document_id: str,
    candidate_rank: int,
    default_method: str,
    default_score_kind: ScoreKind,
) -> RetrievedPassage:
    metadata: Mapping[str, object] = candidate.node.metadata
    canonical_source_uri = _optional_text(metadata.get("canonical_source_uri"))
    return RetrievedPassage(
        rank=rank,
        score=candidate.score,
        score_kind=default_score_kind,
        retrieval_contributions=_retrieval_contributions(
            metadata,
            default_method=default_method,
            default_rank=candidate_rank,
            default_score=candidate.score,
            default_score_kind=default_score_kind,
        ),
        document_id=document_id,
        title=_optional_text(metadata.get("title")) or "Untitled document",
        authors=_string_sequence(metadata.get("authors")),
        published_at=_optional_text(metadata.get("published_at")),
        document_type=_optional_text(metadata.get("document_type")) or "document",
        source_uris=_string_sequence(metadata.get("source_uris")),
        canonical_source_uri=canonical_source_uri,
        section_path=_string_sequence(metadata.get("section_path")),
        passage_voice=_passage_voice(metadata.get("passage_voice")),
        document_author_fraction=_optional_float(metadata.get("document_author_fraction")),
        quoted_speech_fraction=_optional_float(metadata.get("quoted_speech_fraction")),
        uncertain_voice_fraction=_optional_float(metadata.get("uncertain_voice_fraction")),
        attributed_speakers=_string_sequence(metadata.get("attributed_speakers")),
        text=candidate.node.get_content(metadata_mode=MetadataMode.NONE).strip(),
    )


def _document_id(candidate: NodeWithScore) -> str:
    metadata: Mapping[str, object] = candidate.node.metadata
    return _optional_text(metadata.get("document_id")) or candidate.node.ref_doc_id or candidate.node.node_id


def _string_sequence(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            parsed: object = json.loads(value)
        except json.JSONDecodeError:
            return (value,) if value.strip() else ()
        value = parsed
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _passage_voice(
    value: object,
) -> Literal["document_author", "quoted_speech", "uncertain", "mixed", "unknown"]:
    if value in {"document_author", "quoted_speech", "uncertain", "mixed"}:
        return value
    return "unknown"


def _retrieval_contributions(
    metadata: Mapping[str, object],
    *,
    default_method: str,
    default_rank: int,
    default_score: float | None,
    default_score_kind: ScoreKind,
) -> tuple[RetrievalContribution, ...]:
    value = metadata.get(_RETRIEVAL_CONTRIBUTIONS_KEY)
    if isinstance(value, str):
        try:
            contributions = TypeAdapter(tuple[RetrievalContribution, ...]).validate_json(value)
        except ValueError as error:
            raise ValueError("Invalid internal retrieval-contribution metadata.") from error
        if contributions:
            return contributions
    elif value is not None:
        raise ValueError("Invalid internal retrieval-contribution metadata type.")
    return (
        RetrievalContribution(
            method=default_method,
            rank=default_rank,
            score=default_score,
            score_kind=default_score_kind,
        ),
    )
