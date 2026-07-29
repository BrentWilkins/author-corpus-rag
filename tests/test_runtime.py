"""Tests for private environment validation at interactive runtime startup."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from author_corpus.runtime import RuntimeSettings


def test_runtime_settings_resolve_private_relative_paths(tmp_path: Path) -> None:
    """Keep author and corpus specifics in environment-backed local settings."""
    settings = RuntimeSettings.from_environment(
        {
            "AUTHOR_CORPUS_CONFIG": "corpus.local.yaml",
            "AUTHOR_CORPUS_DEFAULT_AUTHOR": "Avery Stone",
            "OLLAMA_MODEL": "synthetic-model",
            "OLLAMA_REASONING_EFFORT": "none",
            "EMBEDDING_MODEL": "synthetic-embedding",
        },
        project_root=tmp_path,
    )

    assert settings.config_path == (tmp_path / "corpus.local.yaml").resolve()
    assert settings.default_author == "Avery Stone"
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
