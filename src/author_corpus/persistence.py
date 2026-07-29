"""Content-addressed cache paths and manifests for generated artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from author_corpus.models import CorpusDocument

CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CacheLayout:
    """Paths for all artifacts derived from one corpus fingerprint."""

    root: Path
    fingerprint: str

    @property
    def directory(self) -> Path:
        """Return the fingerprint-specific artifact directory."""
        return self.root / self.fingerprint

    @property
    def catalog_path(self) -> Path:
        """Return the generated SQLite catalog path."""
        return self.directory / "catalog.sqlite3"

    @property
    def vector_index_dir(self) -> Path:
        """Return the persisted LlamaIndex vector-index directory."""
        return self.directory / "vector_index"

    @property
    def bm25_index_dir(self) -> Path:
        """Return the persisted BM25 passage-index directory."""
        return self.directory / "bm25_index"

    @property
    def summary_path(self) -> Path:
        """Return the cross-fingerprint generated knowledge-store path."""
        return self.root / "knowledge.sqlite3"

    @property
    def query_trace_path(self) -> Path:
        """Return the cross-fingerprint query-trace database path."""
        return self.root / "query_traces.sqlite3"

    @property
    def manifest_path(self) -> Path:
        """Return the generated artifact manifest path."""
        return self.directory / "manifest.json"

    def ensure(self) -> None:
        """Create the fingerprint-specific artifact directory."""
        self.directory.mkdir(parents=True, exist_ok=True)


def corpus_fingerprint(
    documents: Sequence[CorpusDocument],
    *,
    options: Mapping[str, object] | None = None,
) -> str:
    """Return a stable hash of corpus contents and index-affecting options."""
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "documents": sorted(
            (
                {
                    "document_id": document.document_id,
                    "content_hash": document.content_hash,
                    "authors": list(document.authors),
                    "sources": [source.uri for source in document.sources],
                }
                for document in documents
            ),
            key=lambda item: str(item["document_id"]),
        ),
        "options": dict(options or {}),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]


def write_manifest(
    layout: CacheLayout,
    documents: Sequence[CorpusDocument],
    *,
    options: Mapping[str, object] | None = None,
) -> None:
    """Write a privacy-conscious manifest without document text or author names."""
    layout.ensure()
    manifest = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "fingerprint": layout.fingerprint,
        "document_count": len(documents),
        "documents": [
            {
                "document_id": document.document_id,
                "content_hash": document.content_hash,
            }
            for document in sorted(documents, key=lambda item: item.document_id)
        ],
        "options": dict(options or {}),
    }
    temporary_path = layout.manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(layout.manifest_path)


def read_manifest(layout: CacheLayout) -> dict[str, object] | None:
    """Return the saved manifest, or ``None`` when no manifest exists."""
    if not layout.manifest_path.exists():
        return None
    value: object = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid cache manifest: {layout.manifest_path}")
    return cast(dict[str, object], value)
