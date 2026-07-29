"""Exact, exhaustive metadata queries over a normalized corpus."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from author_corpus.models import CorpusDocument


@dataclass(frozen=True, slots=True)
class CatalogStats:
    """Corpus-wide counts returned by the exact catalog."""

    total_documents: int
    documents_with_authors: int
    documents_without_authors: int
    distinct_authors: int
    source_records: int


@dataclass(frozen=True, slots=True)
class AuthorDocumentStats:
    """Exact authorship counts for one author."""

    author: str
    credited_documents: int
    sole_authored_documents: int
    coauthored_documents: int


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One logical document returned by an exact catalog query."""

    document_id: str
    title: str
    authors: tuple[str, ...]
    published_at: str | None
    document_type: str
    source_uris: tuple[str, ...]


class CorpusCatalog:
    """A small SQLite catalog for queries that vector search cannot answer."""

    def __init__(self, path: str | Path) -> None:
        """Initialize a catalog backed by the given SQLite path."""
        self.path = Path(path)

    def rebuild(self, documents: Iterable[CorpusDocument]) -> None:
        """Replace generated catalog contents with the supplied corpus."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            _create_schema(connection)
            connection.execute("DELETE FROM document_authors")
            connection.execute("DELETE FROM sources")
            connection.execute("DELETE FROM documents")

            for document in documents:
                connection.execute(
                    """
                    INSERT INTO documents (
                        document_id,
                        title,
                        published_at,
                        document_type,
                        content_hash,
                        raw_metadata_json,
                        metadata_provenance_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document.document_id,
                        document.title,
                        document.published_at,
                        document.document_type,
                        document.content_hash,
                        json.dumps(document.raw_metadata, ensure_ascii=False, default=str),
                        json.dumps(document.metadata_provenance, ensure_ascii=False),
                    ),
                )
                for ordinal, author in enumerate(document.authors):
                    connection.execute(
                        """
                        INSERT INTO document_authors (
                            document_id, author_key, author_name, ordinal
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            document.document_id,
                            _author_key(author),
                            author,
                            ordinal,
                        ),
                    )
                for ordinal, source in enumerate(document.sources):
                    connection.execute(
                        """
                        INSERT INTO sources (
                            document_id,
                            source_order,
                            uri,
                            source_type,
                            is_canonical,
                            publisher,
                            retrieved_at,
                            content_hash,
                            metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            document.document_id,
                            ordinal,
                            source.uri,
                            source.source_type,
                            source.is_canonical,
                            source.publisher,
                            source.retrieved_at,
                            source.content_hash,
                            json.dumps(source.metadata, ensure_ascii=False, default=str),
                        ),
                    )

    def stats(self) -> CatalogStats:
        """Return corpus-wide document, author, and source counts."""
        with self._connect() as connection:
            total = _scalar(connection, "SELECT COUNT(*) FROM documents")
            with_authors = _scalar(
                connection,
                "SELECT COUNT(DISTINCT document_id) FROM document_authors",
            )
            distinct_authors = _scalar(
                connection,
                "SELECT COUNT(DISTINCT author_key) FROM document_authors",
            )
            source_records = _scalar(connection, "SELECT COUNT(*) FROM sources")
        return CatalogStats(
            total_documents=total,
            documents_with_authors=with_authors,
            documents_without_authors=total - with_authors,
            distinct_authors=distinct_authors,
            source_records=source_records,
        )

    def count_documents(
        self,
        *,
        author: str | None = None,
        document_type: str | None = None,
    ) -> int:
        """Count logical documents matching optional exact filters."""
        author_key = _author_key(author) if author is not None else None
        with self._connect() as connection:
            return _scalar(
                connection,
                """
                SELECT COUNT(*)
                FROM documents d
                WHERE (? IS NULL OR d.document_type = ?)
                  AND (
                      ? IS NULL
                      OR EXISTS (
                          SELECT 1
                          FROM document_authors filter_author
                          WHERE filter_author.document_id = d.document_id
                            AND filter_author.author_key = ?
                      )
                  )
                """,
                (document_type, document_type, author_key, author_key),
            )

    def author_counts(self) -> list[tuple[str, int]]:
        """Return each author and their number of distinct documents."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT MIN(author_name) AS display_name,
                       COUNT(DISTINCT document_id) AS document_count
                FROM document_authors
                GROUP BY author_key
                ORDER BY document_count DESC, display_name
                """
            ).fetchall()
        return [(str(row["display_name"]), int(row["document_count"])) for row in rows]

    def document_type_counts(self) -> list[tuple[str, int]]:
        """Return every exact document type and its logical-document count."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT document_type, COUNT(*) AS document_count
                FROM documents
                GROUP BY document_type
                ORDER BY document_count DESC, document_type
                """
            ).fetchall()
        return [(str(row["document_type"]), int(row["document_count"])) for row in rows]

    def coauthor_counts(self, author: str) -> list[tuple[str, int]]:
        """Return coauthors and shared-document counts for one author."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT MIN(other.author_name) AS display_name,
                       COUNT(DISTINCT other.document_id) AS document_count
                FROM document_authors target
                JOIN document_authors other
                  ON other.document_id = target.document_id
                 AND other.author_key != target.author_key
                WHERE target.author_key = ?
                GROUP BY other.author_key
                ORDER BY document_count DESC, display_name
                """,
                (_author_key(author),),
            ).fetchall()
        return [(str(row["display_name"]), int(row["document_count"])) for row in rows]

    def author_document_stats(self, author: str) -> AuthorDocumentStats:
        """Return credited, sole-authored, and coauthored work counts."""
        with self._connect() as connection:
            credited = _scalar(
                connection,
                """
                SELECT COUNT(DISTINCT target.document_id)
                FROM document_authors target
                WHERE target.author_key = ?
                """,
                (_author_key(author),),
            )
            sole_authored = _scalar(
                connection,
                """
                SELECT COUNT(*)
                FROM document_authors target
                WHERE target.author_key = ?
                  AND (
                      SELECT COUNT(*)
                      FROM document_authors all_authors
                      WHERE all_authors.document_id = target.document_id
                  ) = 1
                """,
                (_author_key(author),),
            )
        return AuthorDocumentStats(
            author=author,
            credited_documents=credited,
            sole_authored_documents=sole_authored,
            coauthored_documents=credited - sole_authored,
        )

    def list_documents(
        self,
        *,
        author: str | None = None,
        document_type: str | None = None,
    ) -> list[CatalogEntry]:
        """List logical documents matching optional exact filters."""
        author_key = _author_key(author) if author is not None else None
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT d.document_id,
                       d.title,
                       d.published_at,
                       d.document_type
                FROM documents d
                WHERE (? IS NULL OR d.document_type = ?)
                  AND (
                      ? IS NULL
                      OR EXISTS (
                          SELECT 1
                          FROM document_authors filter_author
                          WHERE filter_author.document_id = d.document_id
                            AND filter_author.author_key = ?
                      )
                  )
                ORDER BY d.published_at, d.title
                """,
                (document_type, document_type, author_key, author_key),
            ).fetchall()
            return [
                CatalogEntry(
                    document_id=str(row["document_id"]),
                    title=str(row["title"]),
                    authors=self._authors_for(connection, str(row["document_id"])),
                    published_at=(str(row["published_at"]) if row["published_at"] is not None else None),
                    document_type=str(row["document_type"]),
                    source_uris=self._sources_for(connection, str(row["document_id"])),
                )
                for row in rows
            ]

    def _authors_for(
        self,
        connection: sqlite3.Connection,
        document_id: str,
    ) -> tuple[str, ...]:
        rows = connection.execute(
            """
            SELECT author_name
            FROM document_authors
            WHERE document_id = ?
            ORDER BY ordinal
            """,
            (document_id,),
        ).fetchall()
        return tuple(str(row["author_name"]) for row in rows)

    def _sources_for(
        self,
        connection: sqlite3.Connection,
        document_id: str,
    ) -> tuple[str, ...]:
        rows = connection.execute(
            """
            SELECT uri
            FROM sources
            WHERE document_id = ?
            ORDER BY is_canonical DESC, source_order
            """,
            (document_id,),
        ).fetchall()
        return tuple(str(row["uri"]) for row in rows)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            published_at TEXT,
            document_type TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            raw_metadata_json TEXT NOT NULL,
            metadata_provenance_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS document_authors (
            document_id TEXT NOT NULL REFERENCES documents(document_id)
                ON DELETE CASCADE,
            author_key TEXT NOT NULL,
            author_name TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            PRIMARY KEY (document_id, author_key)
        );

        CREATE INDEX IF NOT EXISTS document_authors_author_key
            ON document_authors(author_key);

        CREATE TABLE IF NOT EXISTS sources (
            document_id TEXT NOT NULL REFERENCES documents(document_id)
                ON DELETE CASCADE,
            source_order INTEGER NOT NULL,
            uri TEXT NOT NULL,
            source_type TEXT NOT NULL,
            is_canonical INTEGER NOT NULL,
            publisher TEXT,
            retrieved_at TEXT,
            content_hash TEXT,
            metadata_json TEXT NOT NULL,
            PRIMARY KEY (document_id, uri)
        );

        CREATE INDEX IF NOT EXISTS sources_uri ON sources(uri);
        """
    )


def _author_key(author: str) -> str:
    return " ".join(author.casefold().split())


def _scalar(
    connection: sqlite3.Connection,
    query: str,
    parameters: tuple[str | None, ...] = (),
) -> int:
    row = connection.execute(query, parameters).fetchone()
    if row is None:
        raise RuntimeError("Expected an aggregate query to return one row.")
    return int(row[0])
