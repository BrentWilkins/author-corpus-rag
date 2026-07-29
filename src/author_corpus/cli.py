"""Headless maintenance commands for local corpus artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

import author_corpus
from author_corpus.hybrid import load_or_build_bm25_retriever
from author_corpus.indexing import (
    INDEX_PIPELINE_VERSION,
    build_vector_index_from_nodes,
    load_vector_index,
    persist_vector_index,
    split_documents,
    vector_index_exists,
)
from author_corpus.ingestion import load_corpus_config
from author_corpus.local_llm import LocalModelSettings, OpenAICompatibleCompleter
from author_corpus.persistence import CacheLayout, corpus_fingerprint, write_manifest
from author_corpus.summaries import (
    SummaryProgress,
    SummaryStore,
    build_cached_document_summaries,
    build_cached_knowledge,
)
from author_corpus.timing import TimingLog


def main() -> None:
    """Run the selected headless corpus maintenance command."""
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.command == "build-knowledge":
        _build_knowledge(
            max_input_characters=arguments.max_input_characters,
            max_tokens=arguments.max_tokens,
            include_experimental_synthesis=arguments.include_experimental_synthesis,
        )
    elif arguments.command == "build-index":
        _build_index(
            embedding_model=arguments.embedding_model,
            chunk_size=arguments.chunk_size,
            chunk_overlap=arguments.chunk_overlap,
        )
    elif arguments.command == "chat":
        _chat(
            server_name=arguments.server_name,
            server_port=arguments.server_port,
            share=arguments.share,
            inbrowser=arguments.inbrowser,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and inspect local author-corpus artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    index = subparsers.add_parser(
        "build-index",
        help="Build a versioned structure- and voice-aware vector index.",
    )
    index.add_argument(
        "--embedding-model",
        default=None,
        help="Hugging Face embedding model; defaults to EMBEDDING_MODEL or BAAI/bge-small-en-v1.5.",
    )
    index.add_argument("--chunk-size", type=int, default=1024, help="Maximum tokens per chunk including embedding metadata.")
    index.add_argument("--chunk-overlap", type=int, default=200, help="Overlapping tokens retained between oversized chunks.")
    knowledge = subparsers.add_parser(
        "build-knowledge",
        help="Build or resume per-document navigation summaries.",
    )
    knowledge.add_argument(
        "--max-input-characters",
        type=int,
        default=16_000,
        help="Maximum source characters supplied to one model request.",
    )
    knowledge.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum generated tokens per summary request.",
    )
    knowledge.add_argument(
        "--include-experimental-synthesis",
        action="store_true",
        help="Also build the unaudited broad corpus synthesis.",
    )
    chat = subparsers.add_parser(
        "chat",
        help="Launch the local routed and source-grounded Gradio chat interface.",
    )
    chat.add_argument("--server-name", default="127.0.0.1", help="Bind address; defaults to local access only.")
    chat.add_argument("--server-port", type=int, default=7860, help="Local server port.")
    chat.add_argument("--share", action="store_true", help="Request a temporary public Gradio share link.")
    chat.add_argument("--inbrowser", action="store_true", help="Open the interface in the default browser.")
    return parser


def _chat(
    *,
    server_name: str,
    server_port: int,
    share: bool,
    inbrowser: bool,
) -> None:
    project_root = Path(author_corpus.__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    from author_corpus.runtime import load_query_runtime
    from author_corpus.ui import launch_chat_interface

    runtime = load_query_runtime(project_root=project_root)
    for timing in runtime.timings:
        print(f"{timing.label}: {timing.elapsed_seconds:.3f}s", flush=True)
    launch_chat_interface(
        runtime.service,
        corpus_name=runtime.corpus_name,
        review_workspace=runtime.review_workspace,
        trace_store=runtime.trace_store,
        server_name=server_name,
        server_port=server_port,
        share=share,
        inbrowser=inbrowser,
    )


def _build_index(
    *,
    embedding_model: str | None,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # type: ignore[import-untyped]

    timings = TimingLog()
    total_started = timings.start()
    project_root = Path(author_corpus.__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    load_started = timings.start()
    config_path = _required_path("AUTHOR_CORPUS_CONFIG", relative_to=project_root)
    load_result = load_corpus_config(config_path)
    load_result.raise_for_errors()
    documents = load_result.documents
    print(timings.finish("Load and validate corpus", load_started), flush=True)

    resolved_embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    index_options = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "index_pipeline_version": INDEX_PIPELINE_VERSION,
        "embedding_model": resolved_embedding_model,
    }
    fingerprint = corpus_fingerprint(documents, options=index_options)
    layout = CacheLayout(project_root / ".cache", fingerprint)
    model_started = timings.start()
    embed_model = HuggingFaceEmbedding(model_name=resolved_embedding_model, device="cpu")
    print(timings.finish("Load embedding model", model_started), flush=True)

    if vector_index_exists(layout):
        vector_started = timings.start()
        index = load_vector_index(layout, embed_model=embed_model)
        print(timings.finish("Load vector index", vector_started), flush=True)
    else:
        chunk_started = timings.start()
        nodes = split_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        print(f"{timings.finish('Structure and voice-aware chunking', chunk_started)} ({len(nodes)} chunks)", flush=True)

        index_started = timings.start()
        index = build_vector_index_from_nodes(nodes, embed_model=embed_model)
        print(timings.finish("Embed chunks and build index", index_started), flush=True)

        persist_started = timings.start()
        persist_vector_index(index, layout)
        write_manifest(layout, documents, options=index_options)
        print(timings.finish("Persist vector index", persist_started), flush=True)
        print(f"Vector index ready: {layout.vector_index_dir}", flush=True)

    lexical_started = timings.start()
    _, lexical_loaded_from_cache = load_or_build_bm25_retriever(index, layout, similarity_top_k=30)
    lexical_action = "loaded" if lexical_loaded_from_cache else "built and persisted"
    print(f"{timings.finish('Load or build BM25 index', lexical_started)} ({lexical_action})", flush=True)
    print(timings.finish("Total index command", total_started), flush=True)


def _build_knowledge(
    *,
    max_input_characters: int,
    max_tokens: int,
    include_experimental_synthesis: bool,
) -> None:
    project_root = Path(author_corpus.__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    config_path = _required_path("AUTHOR_CORPUS_CONFIG", relative_to=project_root)
    load_result = load_corpus_config(config_path)
    load_result.raise_for_errors()
    documents = load_result.documents

    model_id = _required_setting("OLLAMA_MODEL")
    if model_id == "your-local-chat-model":
        raise ValueError("Set OLLAMA_MODEL to an installed local chat model.")
    settings = LocalModelSettings.model_validate(
        {
            "model_id": model_id,
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            "reasoning_effort": os.getenv("OLLAMA_REASONING_EFFORT", "none"),
            "max_tokens": max_tokens,
        }
    )
    complete = OpenAICompatibleCompleter(settings)
    knowledge_fingerprint = corpus_fingerprint(documents)
    layout = CacheLayout(project_root / ".cache", knowledge_fingerprint)
    store = SummaryStore(layout.summary_path)

    progress_printer = _ProgressPrinter()
    if include_experimental_synthesis:
        knowledge_result = build_cached_knowledge(
            documents,
            store,
            complete,
            corpus_fingerprint=knowledge_fingerprint,
            model_id=settings.model_id,
            max_input_characters=max_input_characters,
            progress=progress_printer,
        )
        generated_count = knowledge_result.generated_document_summaries
        reused_count = knowledge_result.reused_document_summaries
        detail = f"synthesis cache hit={knowledge_result.synthesis_loaded_from_cache}"
    else:
        summary_result = build_cached_document_summaries(
            documents,
            store,
            complete,
            model_id=settings.model_id,
            max_input_characters=max_input_characters,
            progress=progress_printer,
        )
        generated_count = summary_result.generated_document_summaries
        reused_count = summary_result.reused_document_summaries
        detail = "broad synthesis skipped"
    print(
        "Navigation summaries ready:",
        f"{generated_count} generated,",
        f"{reused_count} reused,",
        f"{detail}.",
        f"total elapsed={progress_printer.total_elapsed_seconds():.1f}s.",
        flush=True,
    )


class _ProgressPrinter:
    """Print per-document and total elapsed time for a knowledge build."""

    def __init__(self) -> None:
        self.started_at = perf_counter()
        self.previous_at = self.started_at

    def __call__(self, progress: SummaryProgress) -> None:
        """Print one completed document with step and total elapsed time."""
        current = perf_counter()
        print(
            f"[{progress.completed_documents}/{progress.total_documents}] "
            f"{progress.status}: {progress.document_id} "
            f"(step={current - self.previous_at:.1f}s, total={current - self.started_at:.1f}s)",
            flush=True,
        )
        self.previous_at = current

    def total_elapsed_seconds(self) -> float:
        """Return total wall-clock time since this printer was created."""
        return perf_counter() - self.started_at


def _required_path(name: str, *, relative_to: Path) -> Path:
    value = _required_setting(name)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve()


def _required_setting(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Set {name} in the environment or project .env file.")
    return value
