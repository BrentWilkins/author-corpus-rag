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
from llama_index.core.schema import BaseNode

from author_corpus.models import CorpusDocument
from author_corpus.persistence import CacheLayout

VECTOR_INDEX_ID = "author-corpus-vector"


def as_llama_documents(documents: Sequence[CorpusDocument]) -> list[Document]:
    """Convert logical corpus documents while preserving source-level metadata."""
    llama_documents: list[Document] = []
    for document in documents:
        canonical_source = document.canonical_source
        metadata = {
            "document_id": document.document_id,
            "title": document.title,
            "authors": json.dumps(list(document.authors), ensure_ascii=False),
            "author_names": " | ".join(document.authors),
            "published_at": document.published_at or "",
            "document_type": document.document_type,
            "source_uris": json.dumps([source.uri for source in document.sources], ensure_ascii=False),
            "canonical_source_uri": canonical_source.uri if canonical_source else "",
        }
        llama_documents.append(
            Document(
                id_=document.document_id,
                text=document.content,
                metadata=metadata,
            )
        )
    return llama_documents


def split_documents(
    documents: Sequence[CorpusDocument],
    *,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
) -> list[BaseNode]:
    """Split documents into nodes that retain logical-document metadata."""
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.get_nodes_from_documents(as_llama_documents(documents))


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
    if not nodes:
        raise ValueError("Cannot build a vector index without nodes.")
    index = VectorStoreIndex(
        nodes,
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
    if not layout.vector_index_dir.exists():
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
    if layout.vector_index_dir.exists():
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
