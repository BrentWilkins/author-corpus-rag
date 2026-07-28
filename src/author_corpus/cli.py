"""Headless maintenance commands for local corpus artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

import author_corpus
from author_corpus.ingestion import load_corpus_config
from author_corpus.local_llm import LocalModelSettings, OpenAICompatibleCompleter
from author_corpus.persistence import CacheLayout, corpus_fingerprint
from author_corpus.summaries import (
    SummaryProgress,
    SummaryStore,
    build_cached_document_summaries,
    build_cached_knowledge,
)


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and inspect local author-corpus artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)
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
    return parser


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
