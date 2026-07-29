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
    assert "grounded_answerer.answer_from_search_result(" in notebook_text
    assert "question=author_resolution.grounding_question" in notebook_text


def test_notebook_uses_configured_author_identity_for_retrieval_and_grounding() -> None:
    """Route literally while using separately normalized retrieval and grounding text."""
    notebook_text = NOTEBOOK_PATH.read_text(encoding="utf-8")

    assert "AUTHOR_CORPUS_DEFAULT_AUTHOR_ALIASES" in notebook_text
    assert "route_decision = route_query(semantic_query)" in notebook_text
    assert "author_resolution = resolve_author_query(semantic_query, author_identity)" in notebook_text
    assert "author_resolution.retrieval_query" in notebook_text
    assert "question=author_resolution.grounding_question" in notebook_text


def test_notebook_reports_exact_cited_span_coverage() -> None:
    """Expose whether persisted citations resolve to versioned source ranges."""
    notebook_text = NOTEBOOK_PATH.read_text(encoding="utf-8")

    assert "query_trace.cited_span_coverage" in notebook_text
    assert "exact cited evidence spans:" in notebook_text


def test_notebook_measures_claim_classification_without_mutating_audits() -> None:
    """Run the labeled offline benchmark without promoting predictions to claim status."""
    notebook_text = NOTEBOOK_PATH.read_text(encoding="utf-8")

    assert "claim-classification.yaml" in notebook_text
    assert "evaluate_claim_classifier(" in notebook_text
    assert "claim false-support rate:" in notebook_text
    assert "validate_claim_classification_sources(" in notebook_text
    assert "claim provenance coverage:" in notebook_text
    assert "AuditedClaim(" not in notebook_text


def test_notebook_evaluates_private_generated_claims_read_only() -> None:
    """Measure trace-derived labels without creating reviews or training data."""
    notebook_text = NOTEBOOK_PATH.read_text(encoding="utf-8")

    assert "AUTHOR_CORPUS_GENERATED_CLAIM_EVAL" in notebook_text
    assert "evaluate_generated_claims(" in notebook_text
    assert 'timings.finish(\\"Generated-claim evaluation\\"' in notebook_text
    assert "export_reviewed_training_examples(" not in notebook_text


def test_notebook_preserves_author_scope_and_only_inspects_synthesis() -> None:
    """Keep future multi-author scope explicit without building derived truth on Run All."""
    notebook_text = NOTEBOOK_PATH.read_text(encoding="utf-8")

    assert "default_scope = AuthorScope()" in notebook_text
    assert "default_scope = AuthorScope.for_author(author_identity.canonical_name)" in notebook_text
    assert "author_scope=default_scope" in notebook_text
    assert "EvidenceSynthesisStore(layout.evidence_synthesis_path)" in notebook_text
    assert "build_evidence_bound_synthesis(" not in notebook_text


def test_notebook_prepares_review_proposals_without_accepting_them() -> None:
    """Expose review candidates while leaving every consequential action explicit."""
    notebook_text = NOTEBOOK_PATH.read_text(encoding="utf-8")

    assert "ClaimReviewProposal.from_decision(" in notebook_text
    assert "ClaimReviewStore(layout.claim_review_path)" in notebook_text
    assert "No review action was taken and no AuditedClaim was created." in notebook_text
    assert "review_claim_proposal(" not in notebook_text
    assert ".put(review" not in notebook_text
    assert "apply_claim_review(" not in notebook_text


def test_notebook_does_not_send_exact_catalog_questions_to_semantic_search() -> None:
    """Execute exact metadata safely without non-exhaustive retrieval or generation."""
    notebook_text = NOTEBOOK_PATH.read_text(encoding="utf-8")

    assert "if route_decision.route is QueryRoute.EXACT_CATALOG" in notebook_text
    assert "exact_result = execute_catalog_query(" in notebook_text
    assert r"default_author=os.getenv(\"AUTHOR_CORPUS_DEFAULT_AUTHOR\")" in notebook_text
    assert r"timings.finish(\"Exact catalog execution\", exact_started)" in notebook_text
    assert "No semantic search or model generation was run" in notebook_text
    assert "Grounded generation skipped because this question belongs to the exact catalog" in notebook_text
