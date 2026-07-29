"""Offline evaluation of source-bound claims extracted from generated traces."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

import yaml
from pydantic import BaseModel, ConfigDict, Field

from author_corpus.claim_classification import CLAIM_EVIDENCE_LABELS, ClaimEvidenceDecision, ClaimEvidenceLabel
from author_corpus.claim_extraction import extract_answer_claims
from author_corpus.models import CorpusDocument
from author_corpus.tracing import QueryTraceStore


class GeneratedClaimEvaluationCase(BaseModel):
    """One private human label for a generated claim and cited evidence pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    evidence_number: int = Field(ge=1)
    expected_label: ClaimEvidenceLabel
    category: str = Field(min_length=1)


class GeneratedClaimCaseResult(BaseModel):
    """One resolved trace case and its current classifier decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case: GeneratedClaimEvaluationCase
    decision: ClaimEvidenceDecision
    exact_span_current: bool

    @property
    def correct(self) -> bool:
        """Return whether the classifier matched the human label."""
        return self.case.expected_label == self.decision.label


class GeneratedClaimLabelMetrics(BaseModel):
    """Per-label support, precision, and recall for generated claims."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: ClaimEvidenceLabel
    support: int = Field(ge=0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)


class GeneratedClaimEvaluation(BaseModel):
    """Safety-focused metrics over private trace-derived claim labels."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cases: tuple[GeneratedClaimCaseResult, ...]
    label_metrics: tuple[GeneratedClaimLabelMetrics, ...]
    elapsed_seconds: float = Field(ge=0.0)

    @property
    def accuracy(self) -> float:
        """Return exact-label accuracy."""
        return 0.0 if not self.cases else sum(case.correct for case in self.cases) / len(self.cases)

    @property
    def exact_span_coverage(self) -> float:
        """Return the fraction whose frozen source spans still match the corpus."""
        return 0.0 if not self.cases else sum(case.exact_span_current for case in self.cases) / len(self.cases)

    @property
    def false_support_rate(self) -> float | None:
        """Return how often non-support evidence is incorrectly promoted to support."""
        non_support = tuple(case for case in self.cases if case.case.expected_label != "supports")
        if not non_support:
            return None
        return sum(case.decision.label == "supports" for case in non_support) / len(non_support)

    @property
    def abstention_rate(self) -> float:
        """Return the fraction classified as uncertain."""
        return 0.0 if not self.cases else sum(case.decision.label == "uncertain" for case in self.cases) / len(self.cases)


class _GeneratedClaimEvaluationFile(BaseModel):
    """Validated private YAML representation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cases: tuple[GeneratedClaimEvaluationCase, ...] = Field(min_length=1)


def load_generated_claim_cases(path: str | Path) -> tuple[GeneratedClaimEvaluationCase, ...]:
    """Load private trace-derived evaluation labels from YAML."""
    value: object = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    return _GeneratedClaimEvaluationFile.model_validate(value).cases


def evaluate_generated_claims(
    cases: tuple[GeneratedClaimEvaluationCase, ...],
    store: QueryTraceStore,
    *,
    documents: Mapping[str, CorpusDocument],
) -> GeneratedClaimEvaluation:
    """Evaluate trace claims without creating reviews or audited claims."""
    started = perf_counter()
    results = tuple(_evaluate_case(case, store, documents=documents) for case in cases)
    return GeneratedClaimEvaluation(
        cases=results,
        label_metrics=tuple(_label_metrics(label, results) for label in CLAIM_EVIDENCE_LABELS),
        elapsed_seconds=perf_counter() - started,
    )


def _evaluate_case(
    case: GeneratedClaimEvaluationCase,
    store: QueryTraceStore,
    *,
    documents: Mapping[str, CorpusDocument],
) -> GeneratedClaimCaseResult:
    trace = store.get(case.trace_id)
    if trace is None:
        raise ValueError(f"Unknown generated-claim trace: {case.trace_id!r}.")
    candidate = next(
        (item for item in extract_answer_claims(trace).candidates if item.candidate_id == case.candidate_id),
        None,
    )
    if candidate is None:
        raise ValueError(f"Unknown candidate {case.candidate_id!r} in trace {case.trace_id!r}.")
    assessment = next((item for item in candidate.evidence if item.evidence_number == case.evidence_number), None)
    if assessment is None:
        raise ValueError(f"Candidate {case.candidate_id!r} does not cite evidence {case.evidence_number}.")
    if assessment.evidence_span is None or assessment.classifier_decision is None:
        raise ValueError(f"Evaluation case {case.name!r} is not tied to an exact source span.")
    source_issues = trace.check_freshness(documents).evidence_issues
    stale_span_ids = {issue.span_id for issue in source_issues}
    return GeneratedClaimCaseResult(
        case=case,
        decision=assessment.classifier_decision,
        exact_span_current=assessment.evidence_span.span_id not in stale_span_ids,
    )


def _label_metrics(
    label: ClaimEvidenceLabel,
    results: tuple[GeneratedClaimCaseResult, ...],
) -> GeneratedClaimLabelMetrics:
    expected = tuple(result for result in results if result.case.expected_label == label)
    predicted = tuple(result for result in results if result.decision.label == label)
    true_positive = sum(result.correct for result in predicted)
    return GeneratedClaimLabelMetrics(
        label=label,
        support=len(expected),
        precision=0.0 if not predicted else true_positive / len(predicted),
        recall=0.0 if not expected else true_positive / len(expected),
    )
