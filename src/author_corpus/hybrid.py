"""Inspectable lexical retrieval and reciprocal-rank fusion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from author_corpus.persistence import CacheLayout
from author_corpus.retrieval import RetrievalContribution, ScoreKind, SemanticCorpusSearch

_CONTRIBUTIONS_KEY = "_retrieval_contributions"
_BM25_INDEX_VERSION = "bm25s-en-stemmed-v1"
_BM25_MANIFEST_NAME = "author-corpus-bm25.json"
_REQUIRED_BM25_FILES = frozenset(
    {
        "corpus.jsonl",
        "data.csc.index.npy",
        "indices.csc.index.npy",
        "indptr.csc.index.npy",
        "params.index.json",
        "retriever.json",
        "vocab.index.json",
        _BM25_MANIFEST_NAME,
    }
)


class _Bm25Manifest(BaseModel):
    """Validated completion marker for a persisted lexical index."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    node_count: int = Field(ge=1)


@dataclass(frozen=True, slots=True)
class RetrieverArm:
    """One named ranking source participating in fusion."""

    method: str
    retriever: BaseRetriever
    score_kind: ScoreKind
    weight: float = 1.0

    def __post_init__(self) -> None:
        """Reject empty names and non-positive fusion weights."""
        if not self.method.strip():
            raise ValueError("Retriever method must not be empty.")
        if self.weight <= 0.0:
            raise ValueError("Retriever weight must be positive.")


@dataclass(frozen=True, slots=True)
class RetrievalProfiles:
    """Search profiles for broad discovery and focused evidence gathering."""

    discovery: SemanticCorpusSearch
    dense_evidence: SemanticCorpusSearch
    lexical_evidence: SemanticCorpusSearch
    hybrid_evidence: SemanticCorpusSearch

    def evaluation_searches(self) -> dict[str, SemanticCorpusSearch]:
        """Return comparable focused-evidence strategies by stable name."""
        return {
            "dense": self.dense_evidence,
            "lexical": self.lexical_evidence,
            "hybrid_rrf": self.hybrid_evidence,
        }


class ReciprocalRankFusionRetriever(BaseRetriever):
    """Fuse independent rankings without mixing their incomparable raw scores."""

    def __init__(
        self,
        arms: tuple[RetrieverArm, ...],
        *,
        similarity_top_k: int = 30,
        rank_constant: float = 60.0,
    ) -> None:
        """Initialize deterministic weighted reciprocal-rank fusion."""
        if len(arms) < 2:
            raise ValueError("Reciprocal-rank fusion requires at least two retrievers.")
        if len({arm.method for arm in arms}) != len(arms):
            raise ValueError("Retriever method names must be unique.")
        if similarity_top_k < 1:
            raise ValueError("similarity_top_k must be at least 1.")
        if rank_constant <= 0.0:
            raise ValueError("rank_constant must be positive.")
        self.arms = arms
        self.similarity_top_k = similarity_top_k
        self.rank_constant = rank_constant
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve from each arm and combine ranks with auditable contributions."""
        representatives: dict[str, NodeWithScore] = {}
        contributions: dict[str, list[RetrievalContribution]] = {}
        fused_scores: dict[str, float] = {}
        best_ranks: dict[str, int] = {}

        for arm in self.arms:
            candidates = arm.retriever.retrieve(query_bundle)
            arm_node_ids: set[str] = set()
            for rank, candidate in enumerate(candidates, start=1):
                key = candidate.node.node_id
                if key in arm_node_ids:
                    continue
                arm_node_ids.add(key)
                contribution = arm.weight / (self.rank_constant + rank)
                representatives.setdefault(key, candidate)
                contributions.setdefault(key, []).append(
                    RetrievalContribution(
                        method=arm.method,
                        rank=rank,
                        score=candidate.score,
                        score_kind=arm.score_kind,
                        rrf_contribution=contribution,
                    )
                )
                fused_scores[key] = fused_scores.get(key, 0.0) + contribution
                best_ranks[key] = min(best_ranks.get(key, rank), rank)

        ordered_keys = sorted(
            fused_scores,
            key=lambda key: (-fused_scores[key], best_ranks[key], key),
        )[: self.similarity_top_k]
        fused: list[NodeWithScore] = []
        for key in ordered_keys:
            node = representatives[key].node.model_copy(deep=True)
            node.metadata[_CONTRIBUTIONS_KEY] = json.dumps(
                [item.model_dump(mode="json") for item in contributions[key]],
                ensure_ascii=False,
                sort_keys=True,
            )
            fused.append(NodeWithScore(node=node, score=fused_scores[key]))
        return fused


def load_or_build_bm25_retriever(
    index: VectorStoreIndex,
    layout: CacheLayout,
    *,
    similarity_top_k: int = 30,
    show_progress: bool = False,
) -> tuple[BaseRetriever, bool]:
    """Load a fingerprinted BM25 index or build it over vector-index nodes."""
    if similarity_top_k < 1:
        raise ValueError("similarity_top_k must be at least 1.")
    node_count = len(index.docstore.docs)
    if bm25_index_exists(layout, expected_node_count=node_count):
        retriever = BM25Retriever.from_persist_dir(str(layout.bm25_index_dir))
        retriever.similarity_top_k = similarity_top_k
        return retriever, True

    retriever = BM25Retriever.from_defaults(
        index=index,
        similarity_top_k=similarity_top_k,
        verbose=show_progress,
    )
    layout.bm25_index_dir.mkdir(parents=True, exist_ok=True)
    retriever.persist(str(layout.bm25_index_dir), show_progress=show_progress)
    _write_bm25_manifest(layout.bm25_index_dir, node_count=node_count)
    return retriever, False


def build_retrieval_profiles(
    index: VectorStoreIndex,
    layout: CacheLayout,
    *,
    candidate_pool_size: int = 30,
    evidence_passages_per_document: int = 3,
    show_progress: bool = False,
) -> tuple[RetrievalProfiles, bool]:
    """Build cached dense, lexical, and fused search profiles over one passage set."""
    if candidate_pool_size < 1:
        raise ValueError("candidate_pool_size must be at least 1.")
    if evidence_passages_per_document < 1:
        raise ValueError("evidence_passages_per_document must be at least 1.")
    dense_retriever = index.as_retriever(similarity_top_k=candidate_pool_size)
    lexical_retriever, lexical_loaded_from_cache = load_or_build_bm25_retriever(
        index,
        layout,
        similarity_top_k=candidate_pool_size,
        show_progress=show_progress,
    )
    hybrid_retriever = ReciprocalRankFusionRetriever(
        (
            RetrieverArm("dense", dense_retriever, "cosine_similarity"),
            RetrieverArm("lexical", lexical_retriever, "bm25"),
        ),
        similarity_top_k=candidate_pool_size,
    )
    return (
        RetrievalProfiles(
            discovery=SemanticCorpusSearch(
                dense_retriever,
                default_limit=5,
                max_passages_per_document=1,
                strategy="dense_discovery",
                score_kind="cosine_similarity",
            ),
            dense_evidence=SemanticCorpusSearch(
                dense_retriever,
                default_limit=6,
                max_passages_per_document=evidence_passages_per_document,
                strategy="dense_evidence",
                score_kind="cosine_similarity",
            ),
            lexical_evidence=SemanticCorpusSearch(
                lexical_retriever,
                default_limit=6,
                max_passages_per_document=evidence_passages_per_document,
                strategy="lexical_evidence",
                score_kind="bm25",
            ),
            hybrid_evidence=SemanticCorpusSearch(
                hybrid_retriever,
                default_limit=6,
                max_passages_per_document=evidence_passages_per_document,
                strategy="hybrid_evidence",
                score_kind="reciprocal_rank_fusion",
            ),
        ),
        lexical_loaded_from_cache,
    )


def bm25_index_exists(layout: CacheLayout, *, expected_node_count: int | None = None) -> bool:
    """Return whether all files required to reload the lexical index exist."""
    if not all((layout.bm25_index_dir / filename).is_file() for filename in _REQUIRED_BM25_FILES):
        return False
    try:
        manifest = _Bm25Manifest.model_validate_json((layout.bm25_index_dir / _BM25_MANIFEST_NAME).read_text(encoding="utf-8"))
    except OSError, ValidationError:
        return False
    return manifest.version == _BM25_INDEX_VERSION and (expected_node_count is None or manifest.node_count == expected_node_count)


def _write_bm25_manifest(directory: Path, *, node_count: int) -> None:
    manifest = _Bm25Manifest(version=_BM25_INDEX_VERSION, node_count=node_count)
    target = directory / _BM25_MANIFEST_NAME
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
