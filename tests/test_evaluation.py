"""Tests for repeatable semantic retrieval evaluation."""

from pathlib import Path

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from author_corpus.evaluation import evaluate_retrieval, load_retrieval_cases
from author_corpus.retrieval import SemanticCorpusSearch


class EvaluationRetriever(BaseRetriever):
    """Return query-dependent synthetic rankings."""

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Return one hit at rank two and one query with no relevant hit."""
        document_ids = ("unrelated", "relevant") if "successful" in query_bundle.query_str else ("unrelated",)
        return [_candidate(document_id, score=1.0 / rank) for rank, document_id in enumerate(document_ids, start=1)]


def test_loads_and_measures_retrieval_cases(tmp_path: Path) -> None:
    """Validate local cases and calculate hit rate and reciprocal rank."""
    path = tmp_path / "retrieval-eval.local.yaml"
    path.write_text(
        """\
cases:
  - name: successful-case
    query: Find the successful evidence.
    relevant_document_ids:
      - relevant
  - name: missed-case
    query: Find missing evidence.
    relevant_document_ids:
      - missing
""",
        encoding="utf-8",
    )
    cases = load_retrieval_cases(path)
    search = SemanticCorpusSearch(
        EvaluationRetriever(),
        default_limit=2,
        max_passages_per_document=1,
    )

    evaluation = evaluate_retrieval(search, cases, top_k=2)

    assert evaluation.hit_rate == 0.5
    assert evaluation.mean_reciprocal_rank == 0.25
    assert evaluation.cases[0].first_relevant_rank == 2
    assert evaluation.cases[1].first_relevant_rank is None


def _candidate(document_id: str, *, score: float) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(
            text=f"Evidence from {document_id}.",
            metadata={
                "document_id": document_id,
                "title": document_id.title(),
                "document_type": "article",
            },
        ),
        score=score,
    )
