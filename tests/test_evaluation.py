"""Tests for repeatable semantic retrieval evaluation."""

from pathlib import Path

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from author_corpus.evaluation import evaluate_retrieval, evaluate_retrieval_strategies, load_retrieval_cases
from author_corpus.retrieval import SemanticCorpusSearch


class EvaluationRetriever(BaseRetriever):
    """Return query-dependent synthetic rankings."""

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Return one hit at rank two and one query with no relevant hit."""
        document_ids = ("unrelated", "relevant") if "successful" in query_bundle.query_str else ("unrelated",)
        return [_candidate(document_id, score=1.0 / rank) for rank, document_id in enumerate(document_ids, start=1)]


class DuplicateEvaluationRetriever(BaseRetriever):
    """Return repeated chunks from one relevant logical document."""

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Return two chunks that must count as one document-level gain."""
        assert query_bundle.query_str
        return [_candidate("relevant", score=1.0), _candidate("relevant", score=0.9)]


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
    assert evaluation.mean_normalized_discounted_cumulative_gain > evaluation.mean_reciprocal_rank
    assert evaluation.cases[0].first_relevant_rank == 2
    assert evaluation.cases[1].first_relevant_rank is None
    assert evaluation.passage_case_count == 0
    assert evaluation.passage_hit_rate is None


def test_passage_evaluation_rejects_right_document_with_wrong_evidence(tmp_path: Path) -> None:
    """Fail a passage label when only the surrounding document ID is correct."""
    path = tmp_path / "retrieval-eval.local.yaml"
    path.write_text(
        """\
cases:
  - name: wrong-passage
    query: Find the successful evidence.
    relevant_document_ids:
      - relevant
    relevant_passages:
      - document_id: relevant
        contains:
          - Expected authorial advice.
        minimum_document_author_fraction: 0.8
""",
        encoding="utf-8",
    )
    search = SemanticCorpusSearch(EvaluationRetriever(), default_limit=2, max_passages_per_document=1)

    evaluation = evaluate_retrieval(search, load_retrieval_cases(path), top_k=2)

    assert evaluation.hit_rate == 1.0
    assert evaluation.passage_hit_rate == 0.0
    assert evaluation.cases[0].first_relevant_rank == 2
    assert evaluation.cases[0].first_relevant_passage_rank is None
    assert evaluation.passage_mean_normalized_discounted_cumulative_gain == 0.0


def test_passage_evaluation_checks_text_section_and_voice() -> None:
    """Accept evidence only when content, structure, and voice provenance agree."""
    cases_path = Path(__file__).parent / "fixtures" / "retrieval_passages.yaml"
    search = SemanticCorpusSearch(EvaluationRetriever(), default_limit=2, max_passages_per_document=1)

    evaluation = evaluate_retrieval(search, load_retrieval_cases(cases_path), top_k=2)

    assert evaluation.passage_hit_rate == 1.0
    assert evaluation.passage_mean_reciprocal_rank == 1.0
    assert evaluation.passage_mean_normalized_discounted_cumulative_gain == 1.0
    assert evaluation.cases[0].first_relevant_rank == 1


def test_compares_named_strategies_with_elapsed_time(tmp_path: Path) -> None:
    """Evaluate identical labels across strategies and retain timing."""
    path = tmp_path / "retrieval-eval.local.yaml"
    path.write_text(
        """\
cases:
  - name: successful-case
    query: Find the successful evidence.
    relevant_document_ids:
      - relevant
""",
        encoding="utf-8",
    )
    search = SemanticCorpusSearch(EvaluationRetriever(), default_limit=2)

    benchmark = evaluate_retrieval_strategies(
        {"dense": search, "hybrid": search},
        load_retrieval_cases(path),
        top_k=2,
    )

    assert [item.strategy for item in benchmark.strategies] == ["dense", "hybrid"]
    assert all(item.elapsed_seconds >= 0.0 for item in benchmark.strategies)
    assert all(item.evaluation.hit_rate == 1.0 for item in benchmark.strategies)


def test_ndcg_counts_a_relevant_document_only_once(tmp_path: Path) -> None:
    """Keep nDCG bounded when multiple chunks come from one relevant document."""
    path = tmp_path / "retrieval-eval.local.yaml"
    path.write_text(
        """\
cases:
  - name: duplicate-document
    query: Find the evidence.
    relevant_document_ids:
      - relevant
""",
        encoding="utf-8",
    )
    search = SemanticCorpusSearch(
        DuplicateEvaluationRetriever(),
        default_limit=2,
        max_passages_per_document=2,
    )

    evaluation = evaluate_retrieval(search, load_retrieval_cases(path), top_k=2)

    assert evaluation.mean_normalized_discounted_cumulative_gain == 1.0


def test_evaluation_can_measure_the_runtime_query_transform(tmp_path: Path) -> None:
    """Keep human wording in labels while retrieving with normalized runtime terms."""
    path = tmp_path / "retrieval-eval.local.yaml"
    path.write_text(
        """\
cases:
  - name: transformed-query
    query: Raw author wording.
    relevant_document_ids:
      - relevant
""",
        encoding="utf-8",
    )
    search = SemanticCorpusSearch(EvaluationRetriever(), default_limit=2)

    evaluation = evaluate_retrieval(
        search,
        load_retrieval_cases(path),
        top_k=2,
        query_transform=lambda query: f"successful {query}",
    )

    assert evaluation.hit_rate == 1.0
    assert evaluation.cases[0].query == "Raw author wording."


def _candidate(document_id: str, *, score: float) -> NodeWithScore:
    document_author_fraction = 0.2 if document_id == "unrelated" else 0.95
    return NodeWithScore(
        node=TextNode(
            text=f"Evidence from {document_id}.",
            metadata={
                "document_id": document_id,
                "title": document_id.title(),
                "document_type": "article",
                "section_path": '["Guide", "Advice"]',
                "passage_voice": "mixed",
                "document_author_fraction": document_author_fraction,
                "quoted_speech_fraction": 1.0 - document_author_fraction,
            },
        ),
        score=score,
    )
