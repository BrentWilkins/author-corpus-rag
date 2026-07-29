"""Typed records shared by bounded reasoning, services, and durable traces."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from author_corpus.answering import GroundedAnswer
from author_corpus.claim_classification import ClaimEvidenceDecision
from author_corpus.retrieval import SemanticSearchResult
from author_corpus.scope import AuthorScope

REASONING_PROMPT_VERSION = "claim-reasoning-v1"
VerificationStatus = Literal["supported", "qualified", "contradicted", "unsupported", "uncertain"]


class ReasoningStep(BaseModel):
    """One explicit retrieval question in a bounded reasoning plan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ordinal: int = Field(ge=1)
    question: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class ReasoningPlan(BaseModel):
    """Inspectable query decomposition with an explicit author scope."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str = Field(min_length=1)
    author_scope: AuthorScope = Field(default_factory=AuthorScope)
    steps: tuple[ReasoningStep, ...] = Field(min_length=1)
    maximum_rounds: int = Field(default=2, ge=1, le=2)

    @model_validator(mode="after")
    def require_ordered_steps(self) -> ReasoningPlan:
        """Require stable one-based step ordering."""
        if tuple(step.ordinal for step in self.steps) != tuple(range(1, len(self.steps) + 1)):
            raise ValueError("Reasoning steps must use contiguous one-based ordinals.")
        return self


class ClaimVerification(BaseModel):
    """One generated claim and the evidence-role decisions behind its status."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    citation_numbers: tuple[int, ...] = Field(min_length=1)
    status: VerificationStatus
    decisions: tuple[ClaimEvidenceDecision, ...] = ()
    rationale: str = Field(min_length=1)

    @property
    def may_answer(self) -> bool:
        """Return whether this claim may appear in an automatic final answer."""
        return self.status in {"supported", "qualified"}


class ReasoningRound(BaseModel):
    """One retrieval, drafting, and verification pass."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    round_number: int = Field(ge=1, le=2)
    retrieval_queries: tuple[str, ...] = Field(min_length=1)
    retrieved_passages: int = Field(ge=0)
    verifications: tuple[ClaimVerification, ...]
    retrieval_seconds: float = Field(ge=0.0)
    generation_seconds: float = Field(ge=0.0)
    verification_seconds: float = Field(ge=0.0)


class ReasoningTrace(BaseModel):
    """Serializable intermediate work retained with a generated answer trace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_version: str = REASONING_PROMPT_VERSION
    plan: ReasoningPlan
    rounds: tuple[ReasoningRound, ...] = Field(min_length=1)


class ReasonedAnswer(BaseModel):
    """A final safe answer plus its bounded reasoning trace and evidence pool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    grounded_answer: GroundedAnswer
    search_result: SemanticSearchResult
    reasoning: ReasoningTrace

    @model_validator(mode="after")
    def require_same_evidence(self) -> ReasonedAnswer:
        """Require the rendered answer to retain the inspected evidence pool."""
        if self.grounded_answer.evidence != self.search_result.passages:
            raise ValueError("A reasoned answer must retain the evidence returned for inspection.")
        return self
