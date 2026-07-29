"""Static safety checks for the interactive notebook."""

from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parents[1] / "notebooks" / "author_corpus_analyzer.ipynb"


def test_notebook_requires_explicit_expensive_operation_flags() -> None:
    """Keep model generation and missing-index builds opt-in."""
    notebook_text = NOTEBOOK_PATH.read_text(encoding="utf-8")

    assert "AUTHOR_CORPUS_ALLOW_INDEX_BUILD" in notebook_text
    assert "AUTHOR_CORPUS_RUN_GROUNDED_ANSWER" in notebook_text


def test_notebook_cannot_resume_broad_synthesis() -> None:
    """Prevent Run All from invoking the unaudited synthesis function."""
    notebook_text = NOTEBOOK_PATH.read_text(encoding="utf-8")

    assert "build_cached_knowledge" not in notebook_text
    assert "knowledge.synthesis" not in notebook_text
    assert "--include-experimental-synthesis" in notebook_text


def test_notebook_uses_separate_discovery_and_grounded_evidence_profiles() -> None:
    """Select diverse discovery or deeper evidence from an inspectable route."""
    notebook_text = NOTEBOOK_PATH.read_text(encoding="utf-8")

    assert "discovery_search = retrieval_profiles.discovery" in notebook_text
    assert "evidence_search = retrieval_profiles.hybrid_evidence" in notebook_text
    assert "route_decision = route_query(semantic_query)" in notebook_text
    assert (
        "selected_search = discovery_search if route_decision.route is QueryRoute.BROAD_DISCOVERY else evidence_search"
        in notebook_text
    )
    assert "evidence_limit=selected_search.default_limit" in notebook_text


def test_notebook_does_not_send_exact_catalog_questions_to_semantic_search() -> None:
    """Stop exact metadata questions before non-exhaustive retrieval or generation."""
    notebook_text = NOTEBOOK_PATH.read_text(encoding="utf-8")

    assert "if route_decision.route is QueryRoute.EXACT_CATALOG" in notebook_text
    assert "No semantic search was run" in notebook_text
    assert "Grounded generation skipped because this question belongs to the exact catalog" in notebook_text
