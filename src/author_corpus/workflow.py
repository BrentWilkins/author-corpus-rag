"""Validated execution switches for interactive and automated workflows."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


class NotebookRunOptions(BaseModel):
    """Control operations that may be slow or invoke a language model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    allow_index_build: bool = False
    run_grounded_answer: bool = False

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> NotebookRunOptions:
        """Load explicit opt-in switches from environment variables."""
        return cls(
            allow_index_build=_environment_flag(environment, "AUTHOR_CORPUS_ALLOW_INDEX_BUILD"),
            run_grounded_answer=_environment_flag(environment, "AUTHOR_CORPUS_RUN_GROUNDED_ANSWER"),
        )


def _environment_flag(environment: Mapping[str, str], name: str) -> bool:
    value = environment.get(name, "").strip().casefold()
    if not value:
        return False
    if value not in _TRUE_VALUES:
        expected = ", ".join(sorted(_TRUE_VALUES))
        raise ValueError(f"{name} must be one of: {expected}.")
    return True
