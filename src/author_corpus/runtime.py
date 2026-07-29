"""Load a reusable local query runtime from private environment settings."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field

from author_corpus.catalog import CorpusCatalog
from author_corpus.claim_classification import (
    evaluate_claim_classifier,
    load_claim_classification_cases,
    validate_claim_classification_sources,
)
from author_corpus.hybrid import build_retrieval_profiles
from author_corpus.identity import parse_author_aliases
from author_corpus.indexing import INDEX_PIPELINE_VERSION, load_vector_index, vector_index_exists
from author_corpus.ingestion import load_corpus_config
from author_corpus.local_llm import LocalModelSettings, OpenAICompatibleCompleter
from author_corpus.models import CorpusLoadResult
from author_corpus.persistence import CacheLayout, corpus_fingerprint, write_manifest
from author_corpus.review import ClaimReviewProposal, ClaimReviewStore, ClaimReviewWorkspace
from author_corpus.service import CorpusQueryService, QueryTiming, QueryTraceContext
from author_corpus.tracing import QueryTraceStore


class RuntimeSettings(BaseModel):
    """Validated private settings required to load an interactive runtime."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config_path: Path
    embedding_model: str = Field(min_length=1)
    default_author: str | None = None
    author_aliases: tuple[str, ...] = ()
    claim_evaluation_path: Path | None = None
    model: LocalModelSettings
    chunk_size: int = Field(default=1024, ge=1)
    chunk_overlap: int = Field(default=200, ge=0)

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str],
        *,
        project_root: Path,
    ) -> RuntimeSettings:
        """Resolve runtime settings without exposing private values in code."""
        config_value = environment.get("AUTHOR_CORPUS_CONFIG", "").strip()
        if not config_value:
            raise ValueError("Set AUTHOR_CORPUS_CONFIG in the environment or project .env file.")
        config_path = Path(config_value).expanduser()
        if not config_path.is_absolute():
            config_path = project_root / config_path

        claim_evaluation_value = environment.get("AUTHOR_CORPUS_CLAIM_EVAL", "").strip()
        claim_evaluation_path = Path(claim_evaluation_value).expanduser() if claim_evaluation_value else None
        if claim_evaluation_path is not None and not claim_evaluation_path.is_absolute():
            claim_evaluation_path = project_root / claim_evaluation_path

        model_id = environment.get("OLLAMA_MODEL", "").strip()
        if not model_id or model_id == "your-local-chat-model":
            raise ValueError("Set OLLAMA_MODEL to an installed local chat model.")
        return cls(
            config_path=config_path.resolve(),
            embedding_model=environment.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip(),
            default_author=environment.get("AUTHOR_CORPUS_DEFAULT_AUTHOR") or None,
            author_aliases=parse_author_aliases(environment.get("AUTHOR_CORPUS_DEFAULT_AUTHOR_ALIASES")),
            claim_evaluation_path=claim_evaluation_path.resolve() if claim_evaluation_path is not None else None,
            model=LocalModelSettings.model_validate(
                {
                    "model_id": model_id,
                    "base_url": environment.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                    "reasoning_effort": environment.get("OLLAMA_REASONING_EFFORT", "none"),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class QueryRuntime:
    """Loaded corpus service plus startup identity and timings."""

    service: CorpusQueryService
    load_result: CorpusLoadResult
    layout: CacheLayout
    fingerprint: str
    review_workspace: ClaimReviewWorkspace | None
    timings: tuple[QueryTiming, ...]

    @property
    def corpus_name(self) -> str:
        """Return the private configured display name or a generic fallback."""
        return self.load_result.name or "Author Corpus"


def load_query_runtime(
    *,
    environment: Mapping[str, str] | None = None,
    project_root: Path | None = None,
) -> QueryRuntime:
    """Load exact, dense, lexical, and generation services from durable artifacts.

    The interactive runtime never performs a missing vector-index build. Run
    ``author-corpus build-index`` explicitly when the fingerprint changes.
    """
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore[import-untyped]

    values = os.environ if environment is None else environment
    root = Path(__file__).resolve().parents[2] if project_root is None else project_root.resolve()
    settings = RuntimeSettings.from_environment(values, project_root=root)
    timings: list[QueryTiming] = []
    total_started = perf_counter()

    load_started = perf_counter()
    load_result = load_corpus_config(settings.config_path)
    load_result.raise_for_errors()
    documents = load_result.documents
    timings.append(_timing("load corpus", load_started))

    index_options: dict[str, object] = {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "index_pipeline_version": INDEX_PIPELINE_VERSION,
        "embedding_model": settings.embedding_model,
    }
    fingerprint = corpus_fingerprint(documents, options=index_options)
    layout = CacheLayout(root / ".cache", fingerprint)
    layout.ensure()

    catalog_started = perf_counter()
    catalog = CorpusCatalog(layout.catalog_path)
    catalog.rebuild(documents)
    write_manifest(layout, documents, options=index_options)
    timings.append(_timing("exact catalog", catalog_started))

    embedding_started = perf_counter()
    embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model, device="cpu")
    timings.append(_timing("embedding model", embedding_started))
    if not vector_index_exists(layout):
        raise RuntimeError(
            f"The vector index is missing for fingerprint {fingerprint}. "
            "Run `uv run --extra local author-corpus build-index` first."
        )

    vector_started = perf_counter()
    vector_index = load_vector_index(layout, embed_model=embed_model)
    timings.append(_timing("vector index", vector_started))

    profiles_started = perf_counter()
    retrieval_profiles, _ = build_retrieval_profiles(
        vector_index,
        layout,
        candidate_pool_size=30,
        evidence_passages_per_document=3,
    )
    timings.append(_timing("retrieval profiles", profiles_started))

    complete = OpenAICompatibleCompleter(settings.model)
    service = CorpusQueryService(
        catalog,
        retrieval_profiles,
        complete=complete,
        model_id=settings.model.model_id,
        default_author=settings.default_author,
        author_aliases=settings.author_aliases,
        minimum_document_author_fraction=0.8,
        trace_context=QueryTraceContext(
            store=QueryTraceStore(layout.query_trace_path),
            corpus_fingerprint=fingerprint,
            document_content_hashes={document.document_id: document.content_hash for document in documents},
            generation_settings=settings.model,
        ),
    )
    review_started = perf_counter()
    review_workspace = _load_review_workspace(
        settings.claim_evaluation_path,
        load_result=load_result,
        fingerprint=fingerprint,
        store=ClaimReviewStore(layout.claim_review_path),
    )
    timings.append(_timing("claim review proposals", review_started))
    timings.append(_timing("total runtime load", total_started))
    return QueryRuntime(
        service=service,
        load_result=load_result,
        layout=layout,
        fingerprint=fingerprint,
        review_workspace=review_workspace,
        timings=tuple(timings),
    )


def _timing(label: str, started_at: float) -> QueryTiming:
    return QueryTiming(label=label, elapsed_seconds=perf_counter() - started_at)


def _load_review_workspace(
    evaluation_path: Path | None,
    *,
    load_result: CorpusLoadResult,
    fingerprint: str,
    store: ClaimReviewStore,
) -> ClaimReviewWorkspace | None:
    """Load configured source-bound classifier cases as stable review proposals."""
    if evaluation_path is None:
        return None
    cases = load_claim_classification_cases(evaluation_path)
    documents = {document.document_id: document for document in load_result.documents}
    issues = validate_claim_classification_sources(cases, documents)
    if issues:
        details = "; ".join(f"{issue.span_id}: {issue.code}" for issue in issues)
        raise ValueError(f"Claim-review proposals contain stale source provenance: {details}")
    evaluation = evaluate_claim_classifier(cases)
    proposals = tuple(
        ClaimReviewProposal.from_decision(
            result.decision,
            corpus_fingerprint=fingerprint,
            claim_id=f"review-{case.name}",
            evidence_spans=(case.evidence_span,),
        )
        for case, result in zip(cases, evaluation.cases, strict=True)
        if case.evidence_span is not None
    )
    return ClaimReviewWorkspace(proposals=proposals, store=store)
