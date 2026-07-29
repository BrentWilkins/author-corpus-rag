"""Tests for structure-aware, voice-aware index nodes."""

from pathlib import Path

from llama_index.core.schema import MetadataMode

from author_corpus.indexing import INDEX_PIPELINE_VERSION, split_documents, vector_index_exists
from author_corpus.models import CorpusDocument, SourceReference
from author_corpus.persistence import CacheLayout


def test_structure_context_is_embedded_but_evidence_remains_original() -> None:
    """Contextualize embeddings without prepending synthetic evidence text."""
    document = _document("# Guide\n\nOpening context.\n\n## Safety Tip\n\nTurn around when conditions deteriorate.")

    nodes = split_documents([document], chunk_size=256, chunk_overlap=20)
    safety = next(node for node in nodes if "Turn around" in node.get_content(metadata_mode=MetadataMode.NONE))

    evidence = safety.get_content(metadata_mode=MetadataMode.NONE)
    embedded = safety.get_content(metadata_mode=MetadataMode.EMBED)
    assert evidence.startswith("## Safety Tip")
    assert "Synthetic Guide" not in evidence
    assert "Synthetic Guide" in embedded
    assert "Guide > Safety Tip" in embedded
    assert safety.metadata["index_pipeline_version"] == INDEX_PIPELINE_VERSION


def test_giant_sections_are_capped_without_losing_quote_provenance() -> None:
    """Split oversized sections and inherit full-section quote annotations."""
    document = _document(
        "# Interview\n\nThe subject says, “" + ("This remains quoted material. " * 160) + "”\n\nNarration resumes."
    )

    nodes = split_documents([document], chunk_size=256, chunk_overlap=20)
    quote_interior_nodes = [
        node
        for node in nodes
        if "quoted material" in node.get_content(metadata_mode=MetadataMode.NONE)
        and "“" not in node.get_content(metadata_mode=MetadataMode.NONE)
        and "”" not in node.get_content(metadata_mode=MetadataMode.NONE)
    ]

    assert len(nodes) > 2
    assert quote_interior_nodes
    assert all(node.metadata["passage_voice"] == "quoted_speech" for node in quote_interior_nodes)
    assert all(node.metadata["voice_offset_basis"] == "section_offsets" for node in quote_interior_nodes)


def test_index_cache_requires_all_persisted_components(tmp_path: Path) -> None:
    """Reject an interrupted cache directory that only partially exists."""
    layout = CacheLayout(tmp_path, "fingerprint")
    layout.vector_index_dir.mkdir(parents=True)
    (layout.vector_index_dir / "index_store.json").write_text("{}", encoding="utf-8")

    assert vector_index_exists(layout) is False

    (layout.vector_index_dir / "docstore.json").write_text("{}", encoding="utf-8")
    (layout.vector_index_dir / "default__vector_store.json").write_text("{}", encoding="utf-8")

    assert vector_index_exists(layout) is True


def test_does_not_index_heading_only_parent_sections() -> None:
    """Keep structural containers as context without returning empty evidence."""
    document = _document("## Container\n\n### Evidence\n\nActual prose.")

    nodes = split_documents([document], chunk_size=256, chunk_overlap=20)

    assert len(nodes) == 1
    assert nodes[0].metadata["section_path"] == '["Container", "Evidence"]'
    assert "Actual prose." in nodes[0].get_content(metadata_mode=MetadataMode.NONE)


def test_indexes_uncertain_voice_without_calling_it_authorial() -> None:
    """Expose malformed quote uncertainty as retrievable metadata."""
    document = _document("## Interview\n\nNarration. “A broken quotation")

    node = split_documents([document], chunk_size=256, chunk_overlap=20)[0]

    assert node.metadata["passage_voice"] == "mixed"
    assert float(node.metadata["uncertain_voice_fraction"]) > 0.0
    assert float(node.metadata["document_author_fraction"]) < 1.0


def _document(content: str) -> CorpusDocument:
    return CorpusDocument(
        document_id="synthetic-guide",
        title="Synthetic Guide",
        authors=("Avery Stone",),
        content=content,
        content_hash="synthetic-hash",
        sources=(
            SourceReference(
                uri="https://example.test/guide",
                source_type="webpage",
                is_canonical=True,
            ),
        ),
        document_type="article",
    )
