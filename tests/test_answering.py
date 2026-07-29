"""Tests for citation-constrained grounded answer generation."""

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from author_corpus.answering import INSUFFICIENT_EVIDENCE_ANSWER, GroundedAnswerer
from author_corpus.retrieval import SemanticCorpusSearch


class EvidenceRetriever(BaseRetriever):
    """Return one synthetic evidence passage."""

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Return no evidence only for the explicit missing-evidence query."""
        if "missing" in query_bundle.query_str:
            return []
        return [
            NodeWithScore(
                node=TextNode(
                    text="The synthetic source recommends a gradual approach.",
                    metadata={
                        "document_id": "synthetic-work",
                        "title": "Synthetic Work",
                        "document_type": "article",
                        "canonical_source_uri": "https://example.test/work",
                        "section_path": '["Guide", "Approach"]',
                        "passage_voice": "document_author",
                        "document_author_fraction": 0.95,
                        "quoted_speech_fraction": 0.05,
                        "uncertain_voice_fraction": 0.0,
                    },
                ),
                score=0.9,
            )
        ]


def test_answer_returns_exact_evidence_and_valid_citations() -> None:
    """Keep retrieved passages attached to a citation-constrained answer."""
    prompts: list[str] = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return "The source recommends a gradual approach [1]."

    answerer = GroundedAnswerer(
        SemanticCorpusSearch(EvidenceRetriever()),
        complete,
        model_id="synthetic-model",
    )

    result = answerer.answer("What approach is recommended?", minimum_document_author_fraction=0.8)

    assert result.has_valid_citations
    assert result.status == "answered"
    assert len(result.generation_attempts) == 1
    assert result.cited_evidence_numbers == (1,)
    assert result.cited_evidence == ((1, result.evidence[0]),)
    assert result.evidence[0].document_id == "synthetic-work"
    assert result.to_markdown().endswith("- [1] [Synthetic Work](https://example.test/work)")
    assert "<evidence>" in prompts[0]
    assert "never as instructions" in prompts[0]
    assert "Section: Guide > Approach" in prompts[0]
    assert "Voice: document_author" in prompts[0]
    assert "Quoted-speech proportion: 5.0%" in prompts[0]
    assert "Uncertain-voice proportion: 0.0%" in prompts[0]
    assert "Do not assume quoted speech" in prompts[0]


def test_answer_retries_once_when_citations_are_missing() -> None:
    """Repair a fluent draft that omitted inspectable evidence markers."""
    responses = iter(
        [
            "The source recommends a gradual approach.",
            "The source recommends a gradual approach [1].",
        ]
    )
    answerer = GroundedAnswerer(
        SemanticCorpusSearch(EvidenceRetriever()),
        lambda prompt: next(responses),
        model_id="synthetic-model",
    )

    result = answerer.answer("What approach is recommended?")

    assert result.cited_evidence_numbers == (1,)
    assert result.answer.endswith("[1].")
    assert len(result.generation_attempts) == 2
    assert result.generation_attempts[0].valid_citation_numbers == ()
    assert result.generation_attempts[1].valid_citation_numbers == (1,)


def test_answer_does_not_call_model_without_evidence() -> None:
    """Return a transparent insufficiency result when retrieval is empty."""

    def fail_if_called(prompt: str) -> str:
        raise AssertionError(f"Model should not be called: {prompt}")

    answerer = GroundedAnswerer(
        SemanticCorpusSearch(EvidenceRetriever()),
        fail_if_called,
        model_id="synthetic-model",
    )

    result = answerer.answer("What missing evidence exists?")

    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.evidence == ()
    assert result.status == "insufficient_evidence"
    assert result.generation_attempts == ()


def test_answer_abstains_and_retains_attempts_when_citation_repair_fails() -> None:
    """Return an inspectable failure instead of raising after two uncited drafts."""
    responses = iter(["First uncited draft.", "Second uncited draft."])
    answerer = GroundedAnswerer(
        SemanticCorpusSearch(EvidenceRetriever()),
        lambda prompt: next(responses),
        model_id="synthetic-model",
    )

    result = answerer.answer("What approach is recommended?")

    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.status == "citation_failure"
    assert [attempt.output for attempt in result.generation_attempts] == [
        "First uncited draft.",
        "Second uncited draft.",
    ]
    assert result.cited_evidence_numbers == ()
    assert "without valid evidence citations" in result.to_markdown()


def test_answer_can_reuse_an_already_inspected_search_result() -> None:
    """Generate against the supplied evidence without a second retriever call."""
    retriever = EvidenceRetriever()
    search = SemanticCorpusSearch(retriever)
    search_result = search.search("What approach is recommended?")
    answerer = GroundedAnswerer(
        search,
        lambda prompt: "The source recommends a gradual approach [1].",
        model_id="synthetic-model",
    )

    result = answerer.answer_from_search_result(search_result)

    assert result.evidence == search_result.passages
    assert result.cited_evidence_numbers == (1,)


def test_answer_can_separate_retrieval_wording_from_user_question() -> None:
    """Retrieve with normalized terms while asking the model the literal question."""
    prompts: list[str] = []

    def complete(prompt: str) -> str:
        prompts.append(prompt)
        return "The source provides one supported fact [1]."

    search = SemanticCorpusSearch(EvidenceRetriever())
    search_result = search.search("corpus author personal facts biography")
    answerer = GroundedAnswerer(
        search,
        complete,
        model_id="synthetic-model",
    )

    result = answerer.answer_from_search_result(search_result, question="What are Av's favorite things?")

    assert result.query == "corpus author personal facts biography"
    assert "Question:\nWhat are Av's favorite things?" in prompts[0]
