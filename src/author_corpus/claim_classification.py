"""Conservative, inspectable classification of claim-to-evidence relationships."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from author_corpus.audit import EvidenceSpan, EvidenceValidationIssue, validate_evidence_spans
from author_corpus.models import CorpusDocument

ClaimEvidenceLabel = Literal[
    "supports",
    "qualifies",
    "contradicts",
    "updates",
    "attributed_report",
    "insufficient",
    "uncertain",
]
EvidenceVoice = Literal["document_author", "quoted_speech", "mixed", "uncertain", "unknown"]

CLAIM_EVIDENCE_LABELS: tuple[ClaimEvidenceLabel, ...] = (
    "supports",
    "qualifies",
    "contradicts",
    "updates",
    "attributed_report",
    "insufficient",
    "uncertain",
)

_WORD_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_DIRECT_QUOTE_PATTERN = re.compile(r"[“”\"]")
_NEGATIONS = frozenset({"cannot", "never", "no", "not", "without"})
_NUMBER_WORDS = frozenset(
    {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
    }
)
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "there",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
    }
)
_UPDATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("after", re.compile(r"\bafter\b")),
    ("as_of", re.compile(r"\bas of\b")),
    ("corrected", re.compile(r"\bcorrect(?:ed|ion)\b")),
    ("increased_from", re.compile(r"\b(?:increased|decreased|changed)\s+from\b")),
    ("later", re.compile(r"\blater\b")),
    ("no_longer", re.compile(r"\bno longer\b")),
    ("now", re.compile(r"\bnow\b")),
    ("previously", re.compile(r"\bpreviously\b")),
    ("recently", re.compile(r"\brecently\b")),
    ("retracted", re.compile(r"\bretract(?:ed|ion)\b")),
    ("revised", re.compile(r"\brevis(?:ed|ion)\b")),
    ("updated", re.compile(r"\bupdat(?:ed|e|ing)\b")),
)
_CONTRADICTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("contrary", re.compile(r"\bcontrary\b")),
    ("denied", re.compile(r"\bdeni(?:ed|es|al)\b")),
    ("disagreed", re.compile(r"\bdisagree(?:d|s|ment)?\b")),
    ("false", re.compile(r"\bfalse\b")),
    ("however", re.compile(r"\bhowever\b")),
    ("incorrect", re.compile(r"\bincorrect\b")),
    ("instead", re.compile(r"\binstead\b")),
)
_QUALIFICATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("can", re.compile(r"\bcan\b")),
    ("except", re.compile(r"\bexcept\b")),
    ("generally", re.compile(r"\bgenerally\b")),
    ("if", re.compile(r"\bif\b")),
    ("may", re.compile(r"\bmay\b")),
    ("might", re.compile(r"\bmight\b")),
    ("not_always", re.compile(r"\bnot always\b")),
    ("only", re.compile(r"\bonly\b")),
    ("possible", re.compile(r"\bpossible|possibly\b")),
    ("sometimes", re.compile(r"\bsometimes\b")),
    ("typically", re.compile(r"\btypically\b")),
    ("unless", re.compile(r"\bunless\b")),
    ("when", re.compile(r"\bwhen\b")),
)
_ATTRIBUTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("according_to", re.compile(r"\baccording to\b")),
    ("argued", re.compile(r"\bargu(?:ed|es)\b")),
    ("claimed", re.compile(r"\bclaim(?:ed|s)\b")),
    ("described", re.compile(r"\bdescrib(?:ed|es)\b")),
    ("reported", re.compile(r"\breport(?:ed|s)\b")),
    ("said", re.compile(r"\bsaid\b")),
    ("stated", re.compile(r"\bstat(?:ed|es)\b")),
    ("says", re.compile(r"\bsays\b")),
    ("told", re.compile(r"\btold\b")),
)
_INSUFFICIENCY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cannot_determine", re.compile(r"\bcannot determine\b")),
    ("insufficient_evidence", re.compile(r"\binsufficient evidence\b")),
    ("no_clear_evidence", re.compile(r"\bno clear evidence\b")),
    ("not_established", re.compile(r"\bnot (?:been )?established\b")),
    ("unclear_whether", re.compile(r"\bunclear whether\b")),
    ("unknown_whether", re.compile(r"\bunknown whether\b")),
)


class ClaimEvidenceSignals(BaseModel):
    """Inspectable lexical and discourse signals used by one decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_terms: tuple[str, ...]
    evidence_terms: tuple[str, ...]
    shared_terms: tuple[str, ...]
    claim_term_coverage: float = Field(ge=0.0, le=1.0)
    negation_mismatch: bool
    claim_numbers: tuple[str, ...] = ()
    evidence_numbers: tuple[str, ...] = ()
    numeric_mismatch: bool = False
    update_markers: tuple[str, ...] = ()
    contradiction_markers: tuple[str, ...] = ()
    qualification_markers: tuple[str, ...] = ()
    attribution_markers: tuple[str, ...] = ()
    insufficiency_markers: tuple[str, ...] = ()
    evidence_voice: EvidenceVoice = "unknown"


class ClaimEvidenceDecision(BaseModel):
    """One heuristic evidence-role prediction with an auditable rationale."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    label: ClaimEvidenceLabel
    rationale: str = Field(min_length=1)
    signals: ClaimEvidenceSignals


class ClaimClassifier(Protocol):
    """Callable boundary for comparing alternative claim classifiers."""

    def __call__(
        self,
        claim: str,
        evidence: str,
        *,
        evidence_voice: EvidenceVoice = "unknown",
    ) -> ClaimEvidenceDecision:
        """Return one inspectable claim/evidence decision."""
        ...


class ClaimClassificationCase(BaseModel):
    """One human-labeled claim/evidence pair."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    expected_label: ClaimEvidenceLabel
    evidence_voice: EvidenceVoice = "document_author"
    category: str = Field(min_length=1)
    evidence_span: EvidenceSpan | None = None

    @model_validator(mode="after")
    def validate_evidence_span(self) -> ClaimClassificationCase:
        """Require optional exact provenance to freeze this case's evidence text."""
        if self.evidence_span is not None and self.evidence_span.text != self.evidence:
            raise ValueError("Claim-classification evidence must exactly equal its evidence span text.")
        return self


class ClaimClassificationCaseResult(BaseModel):
    """One expected label, prediction, and inspectable classifier output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    category: str
    expected_label: ClaimEvidenceLabel
    decision: ClaimEvidenceDecision
    evidence_span_id: str | None = None

    @property
    def correct(self) -> bool:
        """Return whether the prediction matches the human label."""
        return self.expected_label == self.decision.label


class ClaimLabelMetrics(BaseModel):
    """Per-label classification support, precision, and recall."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: ClaimEvidenceLabel
    support: int = Field(ge=0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)


class ClaimClassificationEvaluation(BaseModel):
    """Safety-focused aggregate metrics and every evaluated prediction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cases: tuple[ClaimClassificationCaseResult, ...]
    label_metrics: tuple[ClaimLabelMetrics, ...]
    elapsed_seconds: float = Field(ge=0.0)

    @property
    def accuracy(self) -> float:
        """Return overall exact-label accuracy."""
        return 0.0 if not self.cases else sum(case.correct for case in self.cases) / len(self.cases)

    @property
    def coverage(self) -> float:
        """Return the fraction of cases where the classifier did not abstain."""
        return 0.0 if not self.cases else sum(case.decision.label != "uncertain" for case in self.cases) / len(self.cases)

    @property
    def selective_accuracy(self) -> float | None:
        """Return accuracy among non-abstained decisions."""
        resolved = tuple(case for case in self.cases if case.decision.label != "uncertain")
        return None if not resolved else sum(case.correct for case in resolved) / len(resolved)

    @property
    def false_support_rate(self) -> float | None:
        """Return how often non-support evidence is incorrectly promoted to support."""
        non_support = tuple(case for case in self.cases if case.expected_label != "supports")
        if not non_support:
            return None
        return sum(case.decision.label == "supports" for case in non_support) / len(non_support)

    @property
    def provenance_coverage(self) -> float:
        """Return the fraction of labels tied to exact versioned source spans."""
        return 0.0 if not self.cases else sum(case.evidence_span_id is not None for case in self.cases) / len(self.cases)

    @property
    def mistakes(self) -> tuple[ClaimClassificationCaseResult, ...]:
        """Return all classification errors for inspection."""
        return tuple(case for case in self.cases if not case.correct)


class _ClaimClassificationFile(BaseModel):
    """Validated on-disk representation of claim-classification cases."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cases: tuple[ClaimClassificationCase, ...] = Field(min_length=1)


def classify_claim_evidence(
    claim: str,
    evidence: str,
    *,
    evidence_voice: EvidenceVoice = "unknown",
) -> ClaimEvidenceDecision:
    """Classify one claim/evidence pair conservatively using explicit signals."""
    normalized_claim = _required_text(claim, name="Claim")
    normalized_evidence = _required_text(evidence, name="Evidence")
    claim_terms = _content_terms(normalized_claim)
    evidence_terms = _content_terms(normalized_evidence)
    shared_terms = tuple(sorted(set(claim_terms).intersection(evidence_terms)))
    coverage = 0.0 if not claim_terms else len(shared_terms) / len(set(claim_terms))
    claim_negated = bool(set(claim_terms).intersection(_NEGATIONS))
    evidence_negated = bool(set(evidence_terms).intersection(_NEGATIONS))
    claim_numbers = _number_terms(claim_terms)
    evidence_numbers = _number_terms(evidence_terms)
    numeric_mismatch = bool(claim_numbers and evidence_numbers and not set(claim_numbers).intersection(evidence_numbers))
    update_markers = _markers(normalized_evidence, _UPDATE_PATTERNS)
    contradiction_markers = _markers(normalized_evidence, _CONTRADICTION_PATTERNS)
    qualification_markers = _markers(normalized_evidence, _QUALIFICATION_PATTERNS)
    attribution_markers = _markers(normalized_evidence, _ATTRIBUTION_PATTERNS)
    insufficiency_markers = _markers(normalized_evidence, _INSUFFICIENCY_PATTERNS)
    if evidence_voice != "document_author" and _DIRECT_QUOTE_PATTERN.search(normalized_evidence):
        attribution_markers = (*attribution_markers, "direct_quote")
    signals = ClaimEvidenceSignals(
        claim_terms=claim_terms,
        evidence_terms=evidence_terms,
        shared_terms=shared_terms,
        claim_term_coverage=coverage,
        negation_mismatch=claim_negated != evidence_negated,
        claim_numbers=claim_numbers,
        evidence_numbers=evidence_numbers,
        numeric_mismatch=numeric_mismatch,
        update_markers=update_markers,
        contradiction_markers=contradiction_markers,
        qualification_markers=qualification_markers,
        attribution_markers=attribution_markers,
        insufficiency_markers=insufficiency_markers,
        evidence_voice=evidence_voice,
    )

    if not shared_terms or coverage < 0.25:
        return _decision(
            normalized_claim,
            normalized_evidence,
            "insufficient",
            "The evidence does not share enough substantive claim terms to assign a relationship.",
            signals,
        )
    if insufficiency_markers:
        return _decision(
            normalized_claim,
            normalized_evidence,
            "insufficient",
            "The passage explicitly says the asserted relationship is unknown or lacks evidence.",
            signals,
        )
    if evidence_voice == "quoted_speech" or attribution_markers:
        return _decision(
            normalized_claim,
            normalized_evidence,
            "attributed_report",
            "The passage presents a person's or source's report, which must not be promoted to narrator-established fact.",
            signals,
        )
    if evidence_voice in {"mixed", "uncertain"}:
        return _decision(
            normalized_claim,
            normalized_evidence,
            "uncertain",
            "Mixed or uncertain voice provenance prevents a fact-level relationship from being assigned safely.",
            signals,
        )
    if signals.numeric_mismatch:
        return _decision(
            normalized_claim,
            normalized_evidence,
            "contradicts",
            "The claim and evidence contain incompatible quantities.",
            signals,
        )
    if update_markers:
        return _decision(
            normalized_claim,
            normalized_evidence,
            "updates",
            "Temporal or correction language marks the evidence as a later state or revision.",
            signals,
        )
    if contradiction_markers or (signals.negation_mismatch and coverage >= 0.4):
        return _decision(
            normalized_claim,
            normalized_evidence,
            "contradicts",
            "Explicit opposition or a claim-level negation mismatch conflicts with the assertion.",
            signals,
        )
    if qualification_markers and coverage >= 0.4:
        return _decision(
            normalized_claim,
            normalized_evidence,
            "qualifies",
            "Scope or modal language supports only a narrower assertion.",
            signals,
        )
    if coverage >= 0.65:
        return _decision(
            normalized_claim,
            normalized_evidence,
            "supports",
            "Most substantive claim terms occur in evidence without conflict, attribution, revision, or scope markers.",
            signals,
        )
    if coverage < 0.35:
        return _decision(
            normalized_claim,
            normalized_evidence,
            "insufficient",
            "The evidence has topical overlap but does not cover enough of the assertion.",
            signals,
        )
    return _decision(
        normalized_claim,
        normalized_evidence,
        "uncertain",
        "The evidence overlaps the claim, but no conservative relationship rule resolves it.",
        signals,
    )


def load_claim_classification_cases(path: str | Path) -> tuple[ClaimClassificationCase, ...]:
    """Load and validate a YAML claim/evidence evaluation set."""
    value: object = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    return _ClaimClassificationFile.model_validate(value).cases


def evaluate_claim_classifier(
    cases: tuple[ClaimClassificationCase, ...],
    *,
    classifier: ClaimClassifier = classify_claim_evidence,
) -> ClaimClassificationEvaluation:
    """Evaluate a classifier on fixed labels without converting predictions into audited truth."""
    started = perf_counter()
    results = tuple(
        ClaimClassificationCaseResult(
            name=case.name,
            category=case.category,
            expected_label=case.expected_label,
            evidence_span_id=case.evidence_span.span_id if case.evidence_span is not None else None,
            decision=classifier(
                case.claim,
                case.evidence,
                evidence_voice=case.evidence_voice,
            ),
        )
        for case in cases
    )
    metrics = tuple(_label_metrics(label, results) for label in CLAIM_EVIDENCE_LABELS)
    return ClaimClassificationEvaluation(
        cases=results,
        label_metrics=metrics,
        elapsed_seconds=perf_counter() - started,
    )


def validate_claim_classification_sources(
    cases: tuple[ClaimClassificationCase, ...],
    documents: Mapping[str, CorpusDocument],
) -> tuple[EvidenceValidationIssue, ...]:
    """Validate every provenance-bearing case against the current corpus."""
    spans = tuple(case.evidence_span for case in cases if case.evidence_span is not None)
    return validate_evidence_spans(spans, documents)


def _decision(
    claim: str,
    evidence: str,
    label: ClaimEvidenceLabel,
    rationale: str,
    signals: ClaimEvidenceSignals,
) -> ClaimEvidenceDecision:
    return ClaimEvidenceDecision(
        claim=claim,
        evidence=evidence,
        label=label,
        rationale=rationale,
        signals=signals,
    )


def _required_text(value: str, *, name: str) -> str:
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise ValueError(f"{name} must not be empty.")
    return normalized


def _content_terms(text: str) -> tuple[str, ...]:
    normalized = text.casefold().replace("n't", " not").replace("n’t", " not")
    return tuple(_stem(token) for token in _WORD_PATTERN.findall(normalized) if token not in _STOPWORDS)


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ied"):
        return f"{token[:-3]}y"
    if len(token) > 4 and token.endswith("ed"):
        return token[:-1] if token[-3] == "e" else token[:-2]
    if len(token) > 5 and token.endswith("ing"):
        base = token[:-3]
        return base[:-1] if len(base) > 2 and base[-1] == base[-2] else base
    if len(token) > 4 and token.endswith("es"):
        return token[:-1]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _number_terms(terms: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(term for term in terms if term.isdigit() or term in _NUMBER_WORDS)


def _markers(
    text: str,
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
) -> tuple[str, ...]:
    normalized = text.casefold()
    return tuple(name for name, pattern in patterns if pattern.search(normalized))


def _label_metrics(
    label: ClaimEvidenceLabel,
    results: tuple[ClaimClassificationCaseResult, ...],
) -> ClaimLabelMetrics:
    true_positive = sum(case.expected_label == label and case.decision.label == label for case in results)
    false_positive = sum(case.expected_label != label and case.decision.label == label for case in results)
    false_negative = sum(case.expected_label == label and case.decision.label != label for case in results)
    support = true_positive + false_negative
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    return ClaimLabelMetrics(
        label=label,
        support=support,
        precision=0.0 if precision_denominator == 0 else true_positive / precision_denominator,
        recall=0.0 if recall_denominator == 0 else true_positive / recall_denominator,
    )
