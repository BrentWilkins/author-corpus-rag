"""Tests for conservative claim/evidence classification and evaluation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from author_corpus.audit import EvidenceSpan
from author_corpus.claim_classification import (
    ClaimClassificationCase,
    ClaimEvidenceDecision,
    EvidenceVoice,
    classify_claim_evidence,
    evaluate_claim_classifier,
    load_claim_classification_cases,
    validate_claim_classification_sources,
)
from author_corpus.ingestion import load_corpus

EVALUATION_PATH = Path(__file__).parents[1] / "evaluation" / "claim-classification.yaml"


def test_committed_adversarial_claim_regressions() -> None:
    """Measure every supported evidence role and retain safe abstentions."""
    cases = load_claim_classification_cases(EVALUATION_PATH)

    evaluation = evaluate_claim_classifier(cases)

    assert len(cases) == 29
    assert evaluation.accuracy >= 0.9
    assert evaluation.coverage < 1.0
    assert evaluation.selective_accuracy is not None
    assert evaluation.selective_accuracy >= 0.9
    assert evaluation.false_support_rate is not None
    assert evaluation.false_support_rate <= 0.05
    assert {case.name for case in evaluation.mistakes} <= {"entity-role-reversal"}
    assert {metrics.label for metrics in evaluation.label_metrics} == {
        "supports",
        "qualifies",
        "contradicts",
        "updates",
        "attributed_report",
        "insufficient",
        "uncertain",
    }
    assert all(metrics.support > 0 for metrics in evaluation.label_metrics)


def test_quoted_report_is_not_promoted_to_narrator_fact() -> None:
    """Prefer attribution over support when a source merely reports a statement."""
    decision = classify_claim_evidence(
        "The route is comfortable.",
        'A participant said, "The route is comfortable."',
        evidence_voice="quoted_speech",
    )

    assert decision.label == "attributed_report"
    assert "said" in decision.signals.attribution_markers
    assert "direct_quote" in decision.signals.attribution_markers


def test_negation_mismatch_is_an_inspectable_contradiction() -> None:
    """Detect a direct conflict without treating overlap as semantic confidence."""
    decision = classify_claim_evidence(
        "The gate is open overnight.",
        "The gate is not open overnight.",
        evidence_voice="document_author",
    )

    assert decision.label == "contradicts"
    assert decision.signals.negation_mismatch is True
    assert decision.signals.claim_term_coverage == 1.0


def test_curly_apostrophe_negation_is_an_inspectable_contradiction() -> None:
    """Normalize typographic contractions before extracting negation signals."""
    decision = classify_claim_evidence(
        "The gate is open overnight.",
        "The gate isn’t open overnight.",
        evidence_voice="document_author",
    )

    assert decision.label == "contradicts"
    assert decision.signals.negation_mismatch is True


def test_quoted_terminology_is_not_treated_as_spoken_attribution() -> None:
    """Do not mistake typographic emphasis inside author narration for speech."""
    decision = classify_claim_evidence(
        "The committee uses a free-choice policy.",
        "More recently, the committee moved away from its “free-choice” policy.",
        evidence_voice="document_author",
    )

    assert decision.label == "updates"
    assert decision.signals.update_markers == ("recently",)
    assert decision.signals.attribution_markers == ()


def test_explicit_lack_of_evidence_blocks_positive_support() -> None:
    """Honor an explicit insufficiency statement even with strong term overlap."""
    decision = classify_claim_evidence(
        "Students choose the program because of its curriculum.",
        "There is no clear evidence that students choose the program because of its curriculum.",
        evidence_voice="document_author",
    )

    assert decision.label == "insufficient"
    assert decision.signals.insufficiency_markers == ("no_clear_evidence",)


def test_incompatible_quantities_are_not_supported_by_high_overlap() -> None:
    """Detect a numeric conflict even when nearly every non-numeric term matches."""
    decision = classify_claim_evidence(
        "The entry fee is twenty dollars.",
        "The entry fee is thirty dollars.",
        evidence_voice="document_author",
    )

    assert decision.label == "contradicts"
    assert decision.signals.claim_numbers == ("twenty",)
    assert decision.signals.evidence_numbers == ("thirty",)
    assert decision.signals.numeric_mismatch is True


def test_low_overlap_abstains_as_insufficient_before_discourse_markers() -> None:
    """Do not let a word such as however manufacture a relationship."""
    decision = classify_claim_evidence(
        "The bridge closes during extreme heat.",
        "However, the cafeteria changed its breakfast menu.",
        evidence_voice="document_author",
    )

    assert decision.label == "insufficient"
    assert decision.signals.shared_terms == ()
    assert decision.signals.contradiction_markers == ("however",)


def test_mixed_voice_forces_uncertainty_without_explicit_attribution() -> None:
    """Fail closed when a passage blends narration and quoted speech."""
    decision = classify_claim_evidence(
        "The permit allows overnight parking.",
        "The permit allows overnight parking.",
        evidence_voice="mixed",
    )

    assert decision.label == "uncertain"
    assert decision.signals.evidence_voice == "mixed"


def test_false_support_rate_exposes_an_unsafe_classifier() -> None:
    """Make indiscriminate support predictions visible in benchmark metrics."""
    cases = (
        ClaimClassificationCase(
            name="unsupported",
            category="unsupported",
            claim="The bridge closes.",
            evidence="The office opens.",
            expected_label="insufficient",
        ),
    )

    def always_support(
        claim: str,
        evidence: str,
        *,
        evidence_voice: EvidenceVoice = "unknown",
    ) -> ClaimEvidenceDecision:
        decision = classify_claim_evidence(claim, evidence, evidence_voice=evidence_voice)
        return decision.model_copy(update={"label": "supports", "rationale": "Unsafe synthetic classifier."})

    evaluation = evaluate_claim_classifier(cases, classifier=always_support)

    assert evaluation.accuracy == 0.0
    assert evaluation.false_support_rate == 1.0
    assert len(evaluation.mistakes) == 1


def test_case_file_rejects_unknown_labels(tmp_path: Path) -> None:
    """Fail validation when a benchmark invents an unsupported relationship."""
    path = tmp_path / "claims.yaml"
    path.write_text(
        """\
cases:
  - name: invalid
    category: invalid
    claim: A claim.
    evidence: Some evidence.
    expected_label: maybe_true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_claim_classification_cases(path)


def test_private_case_provenance_validates_against_current_source(tmp_path: Path) -> None:
    """Tie a real-world label to one exact document version and character range."""
    source_path = tmp_path / "article.md"
    source_path.write_text("The bridge closes only during extreme heat.", encoding="utf-8")
    document = load_corpus([source_path]).documents[0]
    span = EvidenceSpan.from_document(document, "The bridge closes only during extreme heat.")
    case = ClaimClassificationCase(
        name="versioned-qualification",
        category="qualification",
        claim="The bridge closes.",
        evidence=span.text,
        evidence_span=span,
        expected_label="qualifies",
    )

    evaluation = evaluate_claim_classifier((case,))

    assert evaluation.provenance_coverage == 1.0
    assert evaluation.cases[0].evidence_span_id == span.span_id
    assert validate_claim_classification_sources((case,), {document.document_id: document}) == ()

    changed = document.model_copy(update={"content_hash": "changed"})
    issues = validate_claim_classification_sources((case,), {document.document_id: changed})
    assert issues[0].code == "document_changed"


def test_case_rejects_evidence_that_differs_from_its_exact_span(tmp_path: Path) -> None:
    """Prevent labels from drifting away from their frozen source evidence."""
    source_path = tmp_path / "article.md"
    source_path.write_text("Exact evidence.", encoding="utf-8")
    document = load_corpus([source_path]).documents[0]
    span = EvidenceSpan.from_document(document, "Exact evidence.")

    with pytest.raises(ValidationError, match="exactly equal"):
        ClaimClassificationCase(
            name="drifted",
            category="support",
            claim="A claim.",
            evidence="Edited evidence.",
            evidence_span=span,
            expected_label="supports",
        )
