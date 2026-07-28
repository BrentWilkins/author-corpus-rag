"""Grounded query tools over exact and semantic corpus indexes."""

from __future__ import annotations

import json

from llama_index.core.tools import FunctionTool

from author_corpus.catalog import CorpusCatalog


def build_catalog_tools(catalog: CorpusCatalog) -> list[FunctionTool]:
    """Build generic tools for exhaustive metadata questions.

    The returned tools calculate results with SQLite. The language model may
    choose and explain a tool result, but it does not calculate corpus totals.

    Returns:
        Function tools for counts, authors, coauthors, and document listings.
    """

    def count_corpus_documents(
        author: str | None = None,
        document_type: str | None = None,
    ) -> str:
        """Count all logical documents matching optional exact filters.

        Args:
            author: An exact author name, matched case-insensitively.
            document_type: An exact document type such as article or paper.

        Returns:
            JSON containing the count and complete corpus coverage.
        """
        stats = catalog.stats()
        return _json(
            {
                "count": catalog.count_documents(
                    author=author,
                    document_type=document_type,
                ),
                "filters": {
                    "author": author,
                    "document_type": document_type,
                },
                "coverage": {
                    "inspected_documents": stats.total_documents,
                    "corpus_documents": stats.total_documents,
                    "exhaustive": True,
                },
            }
        )

    def list_corpus_authors() -> str:
        """List every known author with their distinct-document count.

        Returns:
            JSON containing author counts, unknown-author count, and coverage.
        """
        stats = catalog.stats()
        return _json(
            {
                "authors": [{"name": name, "document_count": count} for name, count in catalog.author_counts()],
                "documents_without_authors": stats.documents_without_authors,
                "coverage": {
                    "inspected_documents": stats.total_documents,
                    "corpus_documents": stats.total_documents,
                    "exhaustive": True,
                },
            }
        )

    def list_author_coauthors(author: str) -> str:
        """List every coauthor and shared-document count for one author.

        Args:
            author: The author whose coauthors should be listed.

        Returns:
            JSON containing exhaustive coauthor counts and corpus coverage.
        """
        stats = catalog.stats()
        return _json(
            {
                "author": author,
                "coauthors": [{"name": name, "shared_document_count": count} for name, count in catalog.coauthor_counts(author)],
                "coverage": {
                    "inspected_documents": stats.total_documents,
                    "corpus_documents": stats.total_documents,
                    "exhaustive": True,
                },
            }
        )

    def get_author_document_stats(author: str) -> str:
        """Return exact sole-author and coauthored document counts.

        Args:
            author: The author whose document-credit breakdown is requested.

        Returns:
            JSON containing exhaustive authorship counts and corpus coverage.
        """
        author_stats = catalog.author_document_stats(author)
        stats = catalog.stats()
        return _json(
            {
                "author": author_stats.author,
                "credited_documents": author_stats.credited_documents,
                "sole_authored_documents": author_stats.sole_authored_documents,
                "coauthored_documents": author_stats.coauthored_documents,
                "coverage": {
                    "inspected_documents": stats.total_documents,
                    "corpus_documents": stats.total_documents,
                    "exhaustive": True,
                },
            }
        )

    def list_corpus_documents(
        author: str | None = None,
        document_type: str | None = None,
    ) -> str:
        """List every logical document matching optional exact filters.

        Args:
            author: An exact author name, matched case-insensitively.
            document_type: An exact document type such as article or paper.

        Returns:
            JSON containing complete matching records and corpus coverage.
        """
        entries = catalog.list_documents(
            author=author,
            document_type=document_type,
        )
        stats = catalog.stats()
        return _json(
            {
                "documents": [
                    {
                        "document_id": entry.document_id,
                        "title": entry.title,
                        "authors": list(entry.authors),
                        "published_at": entry.published_at,
                        "document_type": entry.document_type,
                        "source_uris": list(entry.source_uris),
                    }
                    for entry in entries
                ],
                "matching_document_count": len(entries),
                "coverage": {
                    "inspected_documents": stats.total_documents,
                    "corpus_documents": stats.total_documents,
                    "exhaustive": True,
                },
            }
        )

    return [
        FunctionTool.from_defaults(
            fn=count_corpus_documents,
            name="count_corpus_documents",
            description=(
                "Use for exact or exhaustive corpus counts, including how many works an author wrote. "
                "This tool inspects the complete metadata catalog; never infer these totals from vector-search results."
            ),
        ),
        FunctionTool.from_defaults(
            fn=list_corpus_authors,
            name="list_corpus_authors",
            description=("Use to identify every author represented in the corpus and count their distinct logical documents."),
        ),
        FunctionTool.from_defaults(
            fn=list_author_coauthors,
            name="list_author_coauthors",
            description="Use for exact, exhaustive questions about an author's coauthors.",
        ),
        FunctionTool.from_defaults(
            fn=get_author_document_stats,
            name="get_author_document_stats",
            description=(
                "Use to distinguish every document credit from sole-authored and coauthored works. "
                "This is especially important when selecting a pure-voice style corpus."
            ),
        ),
        FunctionTool.from_defaults(
            fn=list_corpus_documents,
            name="list_corpus_documents",
            description=("Use to exhaustively list logical documents matching exact author or document-type filters."),
        ),
    ]


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
