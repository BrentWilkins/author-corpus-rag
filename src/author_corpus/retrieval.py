"""Inspectable semantic retrieval with document-level provenance."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import MetadataMode, NodeWithScore
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RetrievedPassage(BaseModel):
    """One semantically retrieved passage and its source metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rank: int = Field(ge=1)
    score: float | None = None
    document_id: str
    title: str
    authors: tuple[str, ...] = ()
    published_at: str | None = None
    document_type: str
    source_uris: tuple[str, ...] = ()
    canonical_source_uri: str | None = None
    text: str


class SemanticSearchResult(BaseModel):
    """A non-exhaustive semantic search result with explicit coverage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    passages: tuple[RetrievedPassage, ...]
    inspected_candidates: int = Field(ge=0)
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
    """Search vector-index candidates while preserving inspectable evidence."""

    def __init__(
        self,
        retriever: BaseRetriever,
        *,
        default_limit: int = 5,
        max_passages_per_document: int = 1,
    ) -> None:
        """Initialize search over a configured LlamaIndex retriever."""
        if default_limit < 1:
            raise ValueError("default_limit must be at least 1.")
        if max_passages_per_document < 1:
            raise ValueError("max_passages_per_document must be at least 1.")
        self.retriever = retriever
        self.default_limit = default_limit
        self.max_passages_per_document = max_passages_per_document

    def search(
        self,
        query: str,
        *,
        limit: int | None = None,
    ) -> SemanticSearchResult:
        """Return the highest-ranked passages without claiming full coverage."""
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Semantic query must not be empty.")
        result_limit = self.default_limit if limit is None else limit
        if result_limit < 1:
            raise ValueError("limit must be at least 1.")

        candidates = self.retriever.retrieve(normalized_query)
        document_counts: Counter[str] = Counter()
        passages: list[RetrievedPassage] = []
        for candidate in candidates:
            document_id = _document_id(candidate)
            if document_counts[document_id] >= self.max_passages_per_document:
                continue
            document_counts[document_id] += 1
            passages.append(
                _to_passage(
                    candidate,
                    rank=len(passages) + 1,
                    document_id=document_id,
                )
            )
            if len(passages) == result_limit:
                break

        return SemanticSearchResult(
            query=normalized_query,
            passages=tuple(passages),
            inspected_candidates=len(candidates),
        )


def _to_passage(
    candidate: NodeWithScore,
    *,
    rank: int,
    document_id: str,
) -> RetrievedPassage:
    metadata: Mapping[str, object] = candidate.node.metadata
    canonical_source_uri = _optional_text(metadata.get("canonical_source_uri"))
    return RetrievedPassage(
        rank=rank,
        score=candidate.score,
        document_id=document_id,
        title=_optional_text(metadata.get("title")) or "Untitled document",
        authors=_string_sequence(metadata.get("authors")),
        published_at=_optional_text(metadata.get("published_at")),
        document_type=_optional_text(metadata.get("document_type")) or "document",
        source_uris=_string_sequence(metadata.get("source_uris")),
        canonical_source_uri=canonical_source_uri,
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
