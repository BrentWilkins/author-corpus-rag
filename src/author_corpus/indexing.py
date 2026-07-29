"""Convert normalized documents into persistent LlamaIndex vector indexes."""

from __future__ import annotations

import json
from collections.abc import Sequence

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, MetadataMode, TextNode

from author_corpus.models import CorpusDocument
from author_corpus.persistence import CacheLayout
from author_corpus.structure import MARKDOWN_STRUCTURE_VERSION, MarkdownSection, split_markdown_sections
from author_corpus.voice import VOICE_ANALYSIS_VERSION, VoiceAnalysis, analyze_voice, slice_voice_analysis

VECTOR_INDEX_ID = "author-corpus-vector"
INDEX_PIPELINE_VERSION = f"{MARKDOWN_STRUCTURE_VERSION}+{VOICE_ANALYSIS_VERSION}"

_EMBED_CONTEXT_KEYS = frozenset({"title", "section_context", "section_lead"})
_REQUIRED_INDEX_FILES = frozenset({"default__vector_store.json", "docstore.json", "index_store.json"})


def as_llama_documents(documents: Sequence[CorpusDocument]) -> list[Document]:
    """Convert logical corpus documents while preserving source-level metadata."""
    return [
        Document(
            id_=document.document_id,
            text=document.content,
            metadata=_document_metadata(document),
        )
        for document in documents
    ]


def split_documents(
    documents: Sequence[CorpusDocument],
    *,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
) -> list[BaseNode]:
    """Split documents by Markdown structure and then by sentence boundaries."""
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    nodes: list[BaseNode] = []
    for document in documents:
        for section in split_markdown_sections(document.content):
            if not section.has_body:
                continue
            section_nodes = splitter.get_nodes_from_documents([_section_document(document, section)])
            section_voice = analyze_voice(section.text)
            for node in section_nodes:
                voice, offset_basis = _node_voice(node, section, section_voice)
                node.metadata.update(
                    {
                        "voice_analysis_version": voice.version,
                        "voice_offset_basis": offset_basis,
                        "passage_voice": voice.passage_voice,
                        "document_author_fraction": voice.document_author_fraction,
                        "quoted_speech_fraction": voice.quoted_speech_fraction,
                        "uncertain_voice_fraction": voice.uncertain_fraction,
                        "attributed_speakers": json.dumps(list(voice.attributed_speakers), ensure_ascii=False),
                    }
                )
                node.excluded_embed_metadata_keys = [key for key in node.metadata if key not in _EMBED_CONTEXT_KEYS]
                node.excluded_llm_metadata_keys = list(node.metadata)
            nodes.extend(section_nodes)
    return nodes


def _node_voice(
    node: BaseNode,
    section: MarkdownSection,
    section_voice: VoiceAnalysis,
) -> tuple[VoiceAnalysis, str]:
    node_text = node.get_content(metadata_mode=MetadataMode.NONE)
    if not isinstance(node, TextNode):
        return analyze_voice(node_text), "chunk_fallback"
    start = node.start_char_idx
    end = node.end_char_idx
    if start is not None and end is not None and section.text[start:end] == node_text:
        return (
            slice_voice_analysis(
                section_voice,
                section.text,
                start=start,
                end=end,
            ),
            "section_offsets",
        )
    return analyze_voice(node_text), "chunk_fallback"


def _document_metadata(document: CorpusDocument) -> dict[str, object]:
    canonical_source = document.canonical_source
    return {
        "document_id": document.document_id,
        "title": document.title,
        "authors": json.dumps(list(document.authors), ensure_ascii=False),
        "author_names": " | ".join(document.authors),
        "published_at": document.published_at or "",
        "document_type": document.document_type,
        "source_uris": json.dumps([source.uri for source in document.sources], ensure_ascii=False),
        "canonical_source_uri": canonical_source.uri if canonical_source else "",
    }


def _section_document(document: CorpusDocument, section: MarkdownSection) -> Document:
    metadata = {
        **_document_metadata(document),
        "section_ordinal": section.ordinal,
        "section_path": json.dumps(list(section.heading_path), ensure_ascii=False),
        "section_context": section.heading_context,
        "section_lead": section.lead,
        "index_pipeline_version": INDEX_PIPELINE_VERSION,
    }
    return Document(
        id_=f"{document.document_id}:section:{section.ordinal}",
        text=section.text,
        metadata=metadata,
        excluded_embed_metadata_keys=[key for key in metadata if key not in _EMBED_CONTEXT_KEYS],
        excluded_llm_metadata_keys=list(metadata),
    )


def build_vector_index(
    documents: Sequence[CorpusDocument],
    *,
    embed_model: BaseEmbedding,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
    show_progress: bool = True,
) -> VectorStoreIndex:
    """Build an in-memory vector index from normalized documents."""
    nodes = split_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return build_vector_index_from_nodes(nodes, embed_model=embed_model, show_progress=show_progress)


def build_vector_index_from_nodes(
    nodes: Sequence[BaseNode],
    *,
    embed_model: BaseEmbedding,
    show_progress: bool = True,
) -> VectorStoreIndex:
    """Build an in-memory vector index from prepared evidence nodes."""
    if not nodes:
        raise ValueError("Cannot build a vector index without nodes.")
    index = VectorStoreIndex(
        list(nodes),
        embed_model=embed_model,
        show_progress=show_progress,
    )
    index.set_index_id(VECTOR_INDEX_ID)
    return index


def persist_vector_index(
    index: VectorStoreIndex,
    layout: CacheLayout,
) -> None:
    """Persist a generated vector index under its corpus fingerprint."""
    layout.ensure()
    index.storage_context.persist(persist_dir=layout.vector_index_dir)


def load_vector_index(
    layout: CacheLayout,
    *,
    embed_model: BaseEmbedding,
) -> VectorStoreIndex:
    """Load a previously persisted vector index."""
    if not vector_index_exists(layout):
        raise FileNotFoundError(layout.vector_index_dir)
    storage_context = StorageContext.from_defaults(persist_dir=str(layout.vector_index_dir))
    index = load_index_from_storage(
        storage_context,
        index_id=VECTOR_INDEX_ID,
        embed_model=embed_model,
    )
    if not isinstance(index, VectorStoreIndex):
        raise TypeError(f"Expected VectorStoreIndex, got {type(index).__name__}.")
    return index


def load_or_build_vector_index(
    documents: Sequence[CorpusDocument],
    layout: CacheLayout,
    *,
    embed_model: BaseEmbedding,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
    show_progress: bool = True,
) -> tuple[VectorStoreIndex, bool]:
    """Load a cached index or build and persist it.

    Returns:
        A pair containing the index and whether it was loaded from cache.
    """
    if vector_index_exists(layout):
        return load_vector_index(layout, embed_model=embed_model), True
    index = build_vector_index(
        documents,
        embed_model=embed_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        show_progress=show_progress,
    )
    persist_vector_index(index, layout)
    return index, False


def vector_index_exists(layout: CacheLayout) -> bool:
    """Return whether the fingerprint directory contains a complete index."""
    return all((layout.vector_index_dir / filename).is_file() for filename in _REQUIRED_INDEX_FILES)
