"""Tests for resumable hierarchical corpus knowledge generation."""

from pathlib import Path

import pytest

from author_corpus.ingestion import load_corpus
from author_corpus.persistence import corpus_fingerprint
from author_corpus.summaries import (
    CORPUS_SYNTHESIS_PROMPT_VERSION,
    SummaryProgress,
    SummaryStore,
    build_cached_document_summaries,
    build_cached_knowledge,
)


def test_document_summary_build_skips_broad_synthesis(tmp_path: Path) -> None:
    """Default maintenance work must not start unaudited corpus synthesis."""
    document_path = tmp_path / "work.md"
    document_path.write_text("# Synthetic Work\n\nA short synthetic document.", encoding="utf-8")
    documents = load_corpus([document_path]).documents
    store = SummaryStore(tmp_path / "knowledge.sqlite3")
    calls: list[str] = []

    def complete(prompt: str) -> str:
        calls.append(prompt)
        return "A navigation summary."

    result = build_cached_document_summaries(
        documents,
        store,
        complete,
        model_id="synthetic-model",
    )

    assert result.generated_document_summaries == 1
    assert len(calls) == 1
    assert (
        store.get_corpus_synthesis(
            corpus_fingerprint(documents),
            model_id="synthetic-model",
            prompt_version=CORPUS_SYNTHESIS_PROMPT_VERSION,
        )
        is None
    )


def test_cached_knowledge_reuses_document_and_corpus_work(tmp_path: Path) -> None:
    """Avoid repeated model calls when content and generation settings match."""
    document_path = tmp_path / "work.md"
    document_path.write_text("# Synthetic Work\n\nA short synthetic document.", encoding="utf-8")
    documents = load_corpus([document_path]).documents
    fingerprint = corpus_fingerprint(documents)
    calls: list[str] = []

    def complete(prompt: str) -> str:
        calls.append(prompt)
        return f"Synthetic generated text {len(calls)}."

    store = SummaryStore(tmp_path / "knowledge.sqlite3")
    first = build_cached_knowledge(
        documents,
        store,
        complete,
        corpus_fingerprint=fingerprint,
        model_id="synthetic-model",
    )
    first_call_count = len(calls)
    second = build_cached_knowledge(
        documents,
        store,
        complete,
        corpus_fingerprint=fingerprint,
        model_id="synthetic-model",
    )

    assert first.generated_document_summaries == 1
    assert first.reused_document_summaries == 0
    assert first.synthesis_loaded_from_cache is False
    assert second.generated_document_summaries == 0
    assert second.reused_document_summaries == 1
    assert second.synthesis_loaded_from_cache is True
    assert len(calls) == first_call_count


def test_cached_knowledge_resumes_after_a_failure(tmp_path: Path) -> None:
    """Persist each document summary before a later generation failure."""
    first_path = tmp_path / "first.md"
    first_path.write_text("# First\n\nFirst synthetic document.", encoding="utf-8")
    second_path = tmp_path / "second.md"
    second_path.write_text("# Second\n\nSecond synthetic document.", encoding="utf-8")
    documents = load_corpus([tmp_path]).documents
    store = SummaryStore(tmp_path / "generated" / "knowledge.sqlite3")
    calls = 0

    def interrupted_complete(prompt: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Synthetic interruption")
        return "A cached synthetic summary."

    with pytest.raises(RuntimeError, match="Synthetic interruption"):
        build_cached_knowledge(
            documents,
            store,
            interrupted_complete,
            corpus_fingerprint=corpus_fingerprint(documents),
            model_id="synthetic-model",
        )

    progress: list[SummaryProgress] = []
    resumed = build_cached_knowledge(
        documents,
        store,
        lambda prompt: "A resumed synthetic result.",
        corpus_fingerprint=corpus_fingerprint(documents),
        model_id="synthetic-model",
        progress=progress.append,
    )

    assert resumed.generated_document_summaries == 1
    assert resumed.reused_document_summaries == 1
    assert [item.status for item in progress] == ["reused", "generated"]


def test_long_document_is_summarized_in_bounded_parts(tmp_path: Path) -> None:
    """Map and reduce a document too large for one model request."""
    document_path = tmp_path / "long.md"
    document_path.write_text("Synthetic paragraph. " * 300, encoding="utf-8")
    documents = load_corpus([document_path]).documents
    prompts: list[str] = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return "A concise synthetic partial analysis."

    result = build_cached_knowledge(
        documents,
        SummaryStore(tmp_path / "knowledge.sqlite3"),
        complete,
        corpus_fingerprint=corpus_fingerprint(documents),
        model_id="synthetic-model",
        max_input_characters=1_000,
    )

    document_prompts = [prompt for prompt in prompts if "Analyze this source document faithfully." in prompt]
    assert len(document_prompts) > 1
    assert all(len(prompt) < 2_000 for prompt in document_prompts)
    assert result.generated_document_summaries == 1


def test_long_document_resumes_from_cached_chunks(tmp_path: Path) -> None:
    """Reuse completed chunks after interruption within one large document."""
    document_path = tmp_path / "long.md"
    document_path.write_text("Synthetic paragraph. " * 300, encoding="utf-8")
    documents = load_corpus([document_path]).documents
    store = SummaryStore(tmp_path / "knowledge.sqlite3")
    interrupted_calls = 0

    def interrupted_complete(prompt: str) -> str:
        nonlocal interrupted_calls
        interrupted_calls += 1
        if interrupted_calls == 2:
            raise RuntimeError("Synthetic chunk interruption")
        return "A cached synthetic partial analysis."

    with pytest.raises(RuntimeError, match="Synthetic chunk interruption"):
        build_cached_knowledge(
            documents,
            store,
            interrupted_complete,
            corpus_fingerprint=corpus_fingerprint(documents),
            model_id="synthetic-model",
            max_input_characters=1_000,
        )

    resumed_prompts: list[str] = []

    def resumed_complete(prompt: str) -> str:
        resumed_prompts.append(prompt)
        return "A resumed synthetic result."

    build_cached_knowledge(
        documents,
        store,
        resumed_complete,
        corpus_fingerprint=corpus_fingerprint(documents),
        model_id="synthetic-model",
        max_input_characters=1_000,
    )

    document_prompts = [prompt for prompt in resumed_prompts if "Analyze this source document faithfully." in prompt]
    assert len(document_prompts) == 6
