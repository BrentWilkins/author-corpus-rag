"""Tests for lexical caching and inspectable reciprocal-rank fusion."""

from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from author_corpus.hybrid import ReciprocalRankFusionRetriever, RetrieverArm, bm25_index_exists, build_retrieval_profiles
from author_corpus.persistence import CacheLayout
from author_corpus.retrieval import SemanticCorpusSearch
from author_corpus.scope import AuthorScope


class RankedRetriever(BaseRetriever):
    """Return a fixed ranking with method-specific scores."""

    def __init__(self, ranking: tuple[tuple[str, float], ...]) -> None:
        """Store a deterministic synthetic ranking."""
        self.ranking = ranking
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Return candidates carrying stable node and document IDs."""
        assert query_bundle.query_str
        return [
            NodeWithScore(
                node=TextNode(
                    id_=node_id,
                    text=f"Evidence from {node_id}.",
                    metadata={
                        "document_id": node_id,
                        "title": node_id.title(),
                        "document_type": "article",
                        "passage_voice": "document_author",
                        "document_author_fraction": 1.0,
                        "quoted_speech_fraction": 0.0,
                    },
                ),
                score=score,
            )
            for node_id, score in self.ranking
        ]


def test_rrf_rewards_candidates_supported_by_both_retrievers() -> None:
    """Rank a consensus candidate first without comparing raw score scales."""
    dense = RankedRetriever((("dense-only", 0.99), ("consensus", 0.61)))
    lexical = RankedRetriever((("lexical-only", 14.0), ("consensus", 2.5)))
    fused = ReciprocalRankFusionRetriever(
        (
            RetrieverArm("dense", dense, "cosine_similarity"),
            RetrieverArm("lexical", lexical, "bm25"),
        ),
        similarity_top_k=3,
    )
    search = SemanticCorpusSearch(
        fused,
        default_limit=3,
        strategy="hybrid_rrf",
        score_kind="reciprocal_rank_fusion",
    )

    result = search.search("consensus evidence")

    assert result.passages[0].document_id == "consensus"
    assert result.passages[0].score_kind == "reciprocal_rank_fusion"
    assert [item.method for item in result.passages[0].retrieval_contributions] == ["dense", "lexical"]
    assert [item.rank for item in result.passages[0].retrieval_contributions] == [2, 2]
    assert [item.score for item in result.passages[0].retrieval_contributions] == [0.61, 2.5]


def test_rrf_validates_configuration() -> None:
    """Reject fusion settings that cannot produce a meaningful ranking."""
    retriever = RankedRetriever((("one", 1.0),))

    try:
        ReciprocalRankFusionRetriever((RetrieverArm("dense", retriever, "cosine_similarity"),))
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("Expected a one-arm fusion configuration to fail.")


def test_bm25_cache_requires_complete_current_manifest(tmp_path: Path) -> None:
    """Reject partial and stale lexical indexes before third-party loading."""
    layout = CacheLayout(tmp_path, "fingerprint")
    layout.bm25_index_dir.mkdir(parents=True)
    filenames = {
        "corpus.jsonl",
        "data.csc.index.npy",
        "indices.csc.index.npy",
        "indptr.csc.index.npy",
        "params.index.json",
        "retriever.json",
        "vocab.index.json",
    }
    for filename in filenames:
        (layout.bm25_index_dir / filename).touch()

    assert bm25_index_exists(layout) is False

    manifest = layout.bm25_index_dir / "author-corpus-bm25.json"
    manifest.write_text('{"version":"obsolete","node_count":2}', encoding="utf-8")
    assert bm25_index_exists(layout) is False

    manifest.write_text('{"version":"bm25s-en-stemmed-v1","node_count":2}', encoding="utf-8")
    assert bm25_index_exists(layout) is True
    assert bm25_index_exists(layout, expected_node_count=2) is True
    assert bm25_index_exists(layout, expected_node_count=3) is False


def test_all_runtime_strategies_prefilter_a_rare_author_before_ranking(tmp_path: Path) -> None:
    """Give a one-node author a fair ranking pool instead of filtering only the global top 30."""
    nodes = [
        TextNode(
            id_=f"avery-{ordinal}",
            text="Generic advice without identifying terms.",
            metadata={
                "document_id": f"avery-{ordinal}",
                "title": f"Avery {ordinal}",
                "authors": '["Avery Stone"]',
                "author_keys": ["avery stone"],
                "document_type": "article",
            },
        )
        for ordinal in range(40)
    ]
    nodes.append(
        TextNode(
            id_="jamie-only",
            text="Generic advice without identifying terms.",
            metadata={
                "document_id": "jamie-only",
                "title": "Jamie Only",
                "authors": '["Jamie River"]',
                "author_keys": ["jamie river"],
                "document_type": "article",
            },
        )
    )
    index = VectorStoreIndex(nodes, embed_model=MockEmbedding(embed_dim=8))
    profiles, _loaded = build_retrieval_profiles(
        index,
        CacheLayout(tmp_path, "scoped"),
        candidate_pool_size=30,
    )
    scope = AuthorScope.for_author("Jamie River")

    for search in (
        profiles.discovery,
        profiles.dense_evidence,
        profiles.lexical_evidence,
        profiles.hybrid_evidence,
    ):
        result = search.search("generic advice", author_scope=scope)
        assert [passage.document_id for passage in result.passages] == ["jamie-only"]
        assert result.discarded_by_author_filter == 0
