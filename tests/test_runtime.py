"""Tests for private environment validation at interactive runtime startup."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from author_corpus.audit import EvidenceSpan
from author_corpus.claim_classification import ClaimClassificationCase
from author_corpus.ingestion import load_corpus
from author_corpus.review import ClaimReviewStore
from author_corpus.runtime import RuntimeSettings, _load_review_workspace


def test_runtime_settings_resolve_private_relative_paths(tmp_path: Path) -> None:
    """Keep author and corpus specifics in environment-backed local settings."""
    settings = RuntimeSettings.from_environment(
        {
            "AUTHOR_CORPUS_CONFIG": "corpus.local.yaml",
            "AUTHOR_CORPUS_DEFAULT_AUTHOR": "Avery Stone",
            "AUTHOR_CORPUS_DEFAULT_AUTHOR_ALIASES": '["Avery", "A. Stone"]',
            "AUTHOR_CORPUS_CLAIM_EVAL": "claim-classification.local.yaml",
            "OLLAMA_MODEL": "synthetic-model",
            "OLLAMA_REASONING_EFFORT": "none",
            "EMBEDDING_MODEL": "synthetic-embedding",
        },
        project_root=tmp_path,
    )

    assert settings.config_path == (tmp_path / "corpus.local.yaml").resolve()
    assert settings.default_author == "Avery Stone"
    assert settings.author_aliases == ("Avery", "A. Stone")
    assert settings.claim_evaluation_path == (tmp_path / "claim-classification.local.yaml").resolve()
    assert settings.model.model_id == "synthetic-model"
    assert settings.embedding_model == "synthetic-embedding"


def test_runtime_settings_require_config_and_model(tmp_path: Path) -> None:
    """Fail startup early when private runtime settings are incomplete."""
    with pytest.raises(ValueError, match="AUTHOR_CORPUS_CONFIG"):
        RuntimeSettings.from_environment({}, project_root=tmp_path)
    with pytest.raises(ValueError, match="OLLAMA_MODEL"):
        RuntimeSettings.from_environment(
            {"AUTHOR_CORPUS_CONFIG": "corpus.local.yaml"},
            project_root=tmp_path,
        )


def test_runtime_settings_validate_reasoning_effort(tmp_path: Path) -> None:
    """Reject unsupported model settings rather than passing them downstream."""
    with pytest.raises(ValidationError, match="reasoning_effort"):
        RuntimeSettings.from_environment(
            {
                "AUTHOR_CORPUS_CONFIG": "corpus.local.yaml",
                "OLLAMA_MODEL": "synthetic-model",
                "OLLAMA_REASONING_EFFORT": "guess",
            },
            project_root=tmp_path,
        )


def test_runtime_loads_stable_source_bound_review_proposals(tmp_path: Path) -> None:
    """Populate the UI queue only from provenance that matches the current corpus."""
    source_path = tmp_path / "article.md"
    evidence = "The bridge closes during extreme heat."
    source_path.write_text(evidence, encoding="utf-8")
    load_result = load_corpus([source_path])
    document = load_result.documents[0]
    span = EvidenceSpan.from_document(document, evidence)
    case = ClaimClassificationCase(
        name="heat-closure",
        claim=evidence,
        evidence=evidence,
        expected_label="supports",
        category="support",
        evidence_span=span,
    )
    evaluation_path = tmp_path / "claims.local.yaml"
    evaluation_path.write_text(
        yaml.safe_dump({"cases": [case.model_dump(mode="json")]}),
        encoding="utf-8",
    )

    first = _load_review_workspace(
        evaluation_path,
        load_result=load_result,
        fingerprint="fingerprint",
        store=ClaimReviewStore(tmp_path / "reviews.sqlite3"),
    )
    second = _load_review_workspace(
        evaluation_path,
        load_result=load_result,
        fingerprint="fingerprint",
        store=ClaimReviewStore(tmp_path / "reviews.sqlite3"),
    )

    assert first is not None
    assert second is not None
    assert len(first.proposals) == 1
    assert first.proposals[0].proposal_id == second.proposals[0].proposal_id
