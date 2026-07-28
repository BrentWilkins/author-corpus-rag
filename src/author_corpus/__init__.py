"""Build and query a source-agnostic author corpus."""

from author_corpus.answering import GroundedAnswer, GroundedAnswerer
from author_corpus.audit import (
    AuditedClaim,
    ClaimRelation,
    EvidenceLedger,
    EvidenceSpan,
    EvidenceValidationIssue,
)
from author_corpus.catalog import AuthorDocumentStats, CorpusCatalog
from author_corpus.evaluation import (
    RetrievalCase,
    RetrievalCaseResult,
    RetrievalEvaluation,
    evaluate_retrieval,
    load_retrieval_cases,
)
from author_corpus.ingestion import load_corpus, load_corpus_config
from author_corpus.models import (
    CorpusDocument,
    CorpusLoadResult,
    LoadIssue,
    SourceReference,
)
from author_corpus.persistence import CacheLayout, corpus_fingerprint
from author_corpus.querying import build_catalog_tools
from author_corpus.retrieval import (
    RetrievedPassage,
    SemanticCorpusSearch,
    SemanticSearchResult,
)
from author_corpus.summaries import (
    CachedChunkSummary,
    CachedCorpusSynthesis,
    CachedSummary,
    DocumentSummaryBuildResult,
    KnowledgeBuildResult,
    SummaryProgress,
    SummaryStore,
    build_cached_document_summaries,
    build_cached_knowledge,
)
from author_corpus.timing import TimingLog, TimingRecord
from author_corpus.tracing import (
    GenerationTraceSettings,
    QueryTrace,
    QueryTraceStore,
    RetrievalTraceSettings,
    TracedEvidence,
    TraceFreshness,
)
from author_corpus.workflow import NotebookRunOptions

__all__ = [
    "AuditedClaim",
    "AuthorDocumentStats",
    "CacheLayout",
    "CachedChunkSummary",
    "CachedCorpusSynthesis",
    "CachedSummary",
    "ClaimRelation",
    "CorpusCatalog",
    "CorpusDocument",
    "CorpusLoadResult",
    "DocumentSummaryBuildResult",
    "EvidenceLedger",
    "EvidenceSpan",
    "EvidenceValidationIssue",
    "GenerationTraceSettings",
    "GroundedAnswer",
    "GroundedAnswerer",
    "KnowledgeBuildResult",
    "LoadIssue",
    "NotebookRunOptions",
    "QueryTrace",
    "QueryTraceStore",
    "RetrievedPassage",
    "RetrievalCase",
    "RetrievalCaseResult",
    "RetrievalEvaluation",
    "RetrievalTraceSettings",
    "SemanticCorpusSearch",
    "SemanticSearchResult",
    "SourceReference",
    "SummaryProgress",
    "SummaryStore",
    "TimingLog",
    "TimingRecord",
    "TraceFreshness",
    "TracedEvidence",
    "build_cached_document_summaries",
    "build_cached_knowledge",
    "build_catalog_tools",
    "corpus_fingerprint",
    "evaluate_retrieval",
    "load_corpus",
    "load_corpus_config",
    "load_retrieval_cases",
]
