"""Private holdout evaluation for aggregate claim-verifier decisions."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from author_corpus.claim_extraction import AnswerClaimCandidate, extract_answer_claims
from author_corpus.models import CorpusDocument
from author_corpus.reasoning import ClaimVerifier
from author_corpus.reasoning_models import ClaimVerification, VerificationStatus
from author_corpus.tracing import QueryTraceStore


class VerifierEvaluationCase(BaseModel):
    """One human decision for a complete generated claim and all its citations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    expected_status: VerificationStatus
    category: str = Field(min_length=1)


class VerifierCaseResult(BaseModel):
    """One expected aggregate status and verifier output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case: VerifierEvaluationCase
    verification: ClaimVerification

    @property
    def correct(self) -> bool:
        """Return whether the aggregate status matches the human label."""
        return self.case.expected_status == self.verification.status


class VerifierEvaluation(BaseModel):
    """Safety-focused metrics for one verifier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verifier_name: str = Field(min_length=1)
    cases: tuple[VerifierCaseResult, ...]
    elapsed_seconds: float = Field(ge=0.0)

    @property
    def accuracy(self) -> float:
        """Return exact aggregate-status accuracy."""
        return 0.0 if not self.cases else sum(case.correct for case in self.cases) / len(self.cases)

    @property
    def coverage(self) -> float:
        """Return the fraction not explicitly marked uncertain."""
        return 0.0 if not self.cases else sum(case.verification.status != "uncertain" for case in self.cases) / len(self.cases)

    @property
    def selective_accuracy(self) -> float | None:
        """Return accuracy among non-abstained cases."""
        resolved = tuple(case for case in self.cases if case.verification.status != "uncertain")
        return None if not resolved else sum(case.correct for case in resolved) / len(resolved)

    @property
    def false_acceptance_rate(self) -> float | None:
        """Return unsafe answer admission among claims expected not to be answerable."""
        unsafe = tuple(case for case in self.cases if case.case.expected_status not in {"supported", "qualified"})
        if not unsafe:
            return None
        return sum(case.verification.may_answer for case in unsafe) / len(unsafe)


class VerifierComparison(BaseModel):
    """Comparable results for multiple verifiers on identical cases."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluations: tuple[VerifierEvaluation, ...] = Field(min_length=1)


class _VerifierEvaluationFile(BaseModel):
    """Validated private verifier-evaluation YAML."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cases: tuple[VerifierEvaluationCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_candidates(self) -> _VerifierEvaluationFile:
        """Keep one human aggregate label per generated candidate."""
        keys = [(case.trace_id, case.candidate_id) for case in self.cases]
        if len(set(keys)) != len(keys):
            raise ValueError("Verifier evaluation cannot repeat a trace candidate.")
        return self


def load_verifier_cases(path: str | Path) -> tuple[VerifierEvaluationCase, ...]:
    """Load private aggregate verifier labels from YAML."""
    value: object = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    return _VerifierEvaluationFile.model_validate(value).cases


def evaluate_verifiers(
    cases: tuple[VerifierEvaluationCase, ...],
    trace_store: QueryTraceStore,
    *,
    documents: Mapping[str, CorpusDocument],
    verifiers: Mapping[str, ClaimVerifier],
) -> VerifierComparison:
    """Compare verifiers without changing traces, reviews, or audited claims."""
    if not verifiers:
        raise ValueError("At least one claim verifier is required.")
    candidates = tuple(_resolve_candidate(case, trace_store, documents=documents) for case in cases)
    evaluations: list[VerifierEvaluation] = []
    for name, verifier in verifiers.items():
        normalized_name = " ".join(name.strip().split())
        if not normalized_name:
            raise ValueError("Verifier names must not be blank.")
        started = perf_counter()
        verifications = verifier.verify_many(candidates)
        if len(verifications) != len(cases):
            raise ValueError(f"Verifier {normalized_name!r} returned the wrong number of decisions.")
        evaluations.append(
            VerifierEvaluation(
                verifier_name=normalized_name,
                cases=tuple(
                    VerifierCaseResult(case=case, verification=verification)
                    for case, verification in zip(cases, verifications, strict=True)
                ),
                elapsed_seconds=perf_counter() - started,
            )
        )
    return VerifierComparison(evaluations=tuple(evaluations))


def _resolve_candidate(
    case: VerifierEvaluationCase,
    store: QueryTraceStore,
    *,
    documents: Mapping[str, CorpusDocument],
) -> AnswerClaimCandidate:
    trace = store.get(case.trace_id)
    if trace is None:
        raise ValueError(f"Unknown verifier-evaluation trace: {case.trace_id!r}.")
    freshness = trace.check_freshness(documents)
    if not freshness.is_current:
        raise ValueError(f"Verifier case {case.name!r} does not have current exact source evidence.")
    candidate = next(
        (item for item in extract_answer_claims(trace).candidates if item.candidate_id == case.candidate_id),
        None,
    )
    if candidate is None:
        raise ValueError(f"Unknown candidate {case.candidate_id!r} in trace {case.trace_id!r}.")
    if not candidate.is_source_bound:
        raise ValueError(f"Verifier case {case.name!r} is not source-bound.")
    return candidate
