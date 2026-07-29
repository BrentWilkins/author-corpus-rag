"""Bounded, claim-verified reasoning over source-bound corpus evidence."""

from __future__ import annotations

import re
from collections.abc import Sequence
from time import perf_counter
from typing import Protocol

from author_corpus.answering import INSUFFICIENT_EVIDENCE_ANSWER, GroundedAnswer, GroundedAnswerer, TextCompleter
from author_corpus.claim_classification import ClaimEvidenceDecision
from author_corpus.claim_extraction import AnswerClaimCandidate, extract_grounded_answer_claims
from author_corpus.reasoning_models import (
    REASONING_PROMPT_VERSION,
    ClaimVerification,
    ReasonedAnswer,
    ReasoningPlan,
    ReasoningRound,
    ReasoningStep,
    ReasoningTrace,
    VerificationStatus,
)
from author_corpus.retrieval import RetrievedPassage, SemanticCorpusSearch, SemanticSearchResult
from author_corpus.scope import AuthorScope

_COMPOSITIONAL_SIGNAL = re.compile(
    r"\b(compare|contrast|versus|vs\.?|changed? over time|evolved?|relationship between|"
    r"reconcile|contradict|qualified?|across (?:the )?(?:corpus|articles|works)|"
    r"both .+ and|first .+ then)\b",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY = re.compile(r"\s+(?:versus|vs\.?|but|while|whereas)\s+|;\s*", re.IGNORECASE)


class ClaimVerifier(Protocol):
    """Replaceable boundary for evaluated claim/evidence verification."""

    def verify(self, candidate: AnswerClaimCandidate) -> ClaimVerification:
        """Verify one citation-bound generated claim."""
        ...


class ConservativeClaimVerifier:
    """Aggregate existing inspectable evidence-role decisions conservatively."""

    def verify(self, candidate: AnswerClaimCandidate) -> ClaimVerification:
        """Reject missing spans, conflicts, updates, and mixed evidence roles."""
        decisions = tuple(
            assessment.classifier_decision for assessment in candidate.evidence if assessment.classifier_decision is not None
        )
        if not candidate.is_source_bound or len(decisions) != len(candidate.evidence):
            return _verification(candidate, "uncertain", decisions, "At least one citation lacks a current exact source span.")
        labels = {decision.label for decision in decisions}
        if "contradicts" in labels:
            return _verification(candidate, "contradicted", decisions, "At least one cited passage contradicts the claim.")
        if "insufficient" in labels:
            return _verification(candidate, "unsupported", decisions, "At least one cited passage is insufficient for the claim.")
        if labels <= {"supports"}:
            return _verification(candidate, "supported", decisions, "Every cited passage conservatively supports the claim.")
        if labels <= {"supports", "qualifies"} and "qualifies" in labels:
            return _verification(candidate, "qualified", decisions, "The evidence supports only a qualified form of the claim.")
        return _verification(
            candidate,
            "uncertain",
            decisions,
            f"Evidence roles {sorted(labels)} do not permit automatic factual promotion.",
        )


class BoundedReasoningEngine:
    """Plan, retrieve, draft, verify, and optionally correct one hard question."""

    def __init__(
        self,
        search: SemanticCorpusSearch,
        complete: TextCompleter,
        *,
        model_id: str,
        verifier: ClaimVerifier | None = None,
        evidence_limit: int = 10,
        maximum_rounds: int = 2,
    ) -> None:
        """Initialize a reasoning engine with strict latency and evidence bounds."""
        if evidence_limit < 1:
            raise ValueError("evidence_limit must be at least 1.")
        if maximum_rounds not in {1, 2}:
            raise ValueError("maximum_rounds must be one or two.")
        self.search = search
        self.complete = complete
        self.model_id = model_id
        self.verifier = verifier or ConservativeClaimVerifier()
        self.evidence_limit = evidence_limit
        self.maximum_rounds = maximum_rounds

    def answer(
        self,
        question: str,
        *,
        retrieval_question: str | None = None,
        author_scope: AuthorScope | None = None,
        minimum_document_author_fraction: float | None = None,
    ) -> ReasonedAnswer:
        """Return only source-bound supported or qualified generated claims."""
        scope = author_scope or AuthorScope()
        plan = plan_reasoning(
            question,
            retrieval_question=retrieval_question,
            author_scope=scope,
            maximum_rounds=self.maximum_rounds,
        )
        passages: list[RetrievedPassage] = []
        rounds: list[ReasoningRound] = []
        queries = tuple(step.question for step in plan.steps)
        final_draft: GroundedAnswer | None = None
        final_verifications: tuple[ClaimVerification, ...] = ()

        for round_number in range(1, self.maximum_rounds + 1):
            retrieval_started = perf_counter()
            inspected_candidates = 0
            discarded = 0
            for query in queries:
                result = self.search.search(
                    query,
                    limit=self.evidence_limit,
                    minimum_document_author_fraction=minimum_document_author_fraction,
                )
                passages = _merge_passages(passages, result.passages, limit=self.evidence_limit)
                inspected_candidates += result.inspected_candidates
                discarded += result.discarded_by_voice_filter
            retrieval_seconds = perf_counter() - retrieval_started
            merged = SemanticSearchResult(
                query=question,
                strategy=f"bounded_reasoning:{self.search.strategy}",
                score_kind="unknown",
                passages=tuple(passage.model_copy(update={"rank": rank}) for rank, passage in enumerate(passages, start=1)),
                inspected_candidates=inspected_candidates,
                discarded_by_voice_filter=discarded,
            )

            generation_started = perf_counter()
            final_draft = GroundedAnswerer(
                self.search,
                self.complete,
                model_id=self.model_id,
                evidence_limit=self.evidence_limit,
                prompt_version=REASONING_PROMPT_VERSION,
            ).answer_from_search_result(merged, question=question)
            generation_seconds = perf_counter() - generation_started

            verification_started = perf_counter()
            extraction = extract_grounded_answer_claims(
                final_draft,
                extraction_id=f"reasoning-round-{round_number}",
            )
            final_verifications = tuple(self.verifier.verify(candidate) for candidate in extraction.candidates)
            verification_seconds = perf_counter() - verification_started
            rounds.append(
                ReasoningRound(
                    round_number=round_number,
                    retrieval_queries=queries,
                    retrieved_passages=len(merged.passages),
                    verifications=final_verifications,
                    retrieval_seconds=retrieval_seconds,
                    generation_seconds=generation_seconds,
                    verification_seconds=verification_seconds,
                )
            )
            unresolved = tuple(verification.statement for verification in final_verifications if not verification.may_answer)
            if not unresolved or round_number == self.maximum_rounds:
                break
            queries = unresolved[:3]

        if final_draft is None:
            raise RuntimeError("Reasoning completed without producing a draft.")
        final_answer = _safe_answer(final_draft, final_verifications)
        final_search = SemanticSearchResult(
            query=question,
            strategy=f"bounded_reasoning:{self.search.strategy}",
            score_kind="unknown",
            passages=final_answer.evidence,
            inspected_candidates=sum(round_.retrieved_passages for round_ in rounds),
            discarded_by_voice_filter=0,
        )
        return ReasonedAnswer(
            grounded_answer=final_answer,
            search_result=final_search,
            reasoning=ReasoningTrace(plan=plan, rounds=tuple(rounds)),
        )


def requires_multistep_reasoning(question: str) -> bool:
    """Return whether deterministic signals indicate a compositional question."""
    return _COMPOSITIONAL_SIGNAL.search(" ".join(question.strip().split())) is not None


def plan_reasoning(
    question: str,
    *,
    retrieval_question: str | None = None,
    author_scope: AuthorScope | None = None,
    maximum_rounds: int = 2,
) -> ReasoningPlan:
    """Create a deterministic, source-agnostic retrieval decomposition."""
    normalized = " ".join(question.strip().split())
    if not normalized:
        raise ValueError("Question must not be empty.")
    normalized_retrieval = " ".join((retrieval_question or normalized).strip().split())
    if not normalized_retrieval:
        raise ValueError("Retrieval question must not be empty.")
    clauses = tuple(clause.strip(" ,") for clause in _CLAUSE_BOUNDARY.split(normalized_retrieval) if clause.strip(" ,"))
    questions = clauses if len(clauses) > 1 else (normalized_retrieval,)
    steps = tuple(
        ReasoningStep(
            ordinal=ordinal,
            question=part,
            rationale="Retrieve this explicit component independently before composing the answer.",
        )
        for ordinal, part in enumerate(questions, start=1)
    )
    return ReasoningPlan(
        question=normalized,
        author_scope=author_scope or AuthorScope(),
        steps=steps,
        maximum_rounds=maximum_rounds,
    )


def _merge_passages(
    existing: Sequence[RetrievedPassage],
    additions: Sequence[RetrievedPassage],
    *,
    limit: int,
) -> list[RetrievedPassage]:
    merged: list[RetrievedPassage] = list(existing)
    keys = {_passage_key(passage) for passage in existing}
    for passage in additions:
        key = _passage_key(passage)
        if key in keys:
            continue
        merged.append(passage)
        keys.add(key)
        if len(merged) == limit:
            break
    return merged


def _passage_key(passage: RetrievedPassage) -> tuple[str, str]:
    span_id = passage.evidence_span.span_id if passage.evidence_span is not None else passage.text
    return passage.document_id, span_id


def _verification(
    candidate: AnswerClaimCandidate,
    status: VerificationStatus,
    decisions: tuple[ClaimEvidenceDecision, ...],
    rationale: str,
) -> ClaimVerification:
    return ClaimVerification(
        candidate_id=candidate.candidate_id,
        statement=candidate.statement,
        citation_numbers=candidate.citation_numbers,
        status=status,
        decisions=decisions,
        rationale=rationale,
    )


def _safe_answer(draft: GroundedAnswer, verifications: tuple[ClaimVerification, ...]) -> GroundedAnswer:
    accepted = tuple(verification for verification in verifications if verification.may_answer)
    if not accepted:
        return GroundedAnswer(
            query=draft.query,
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            evidence=draft.evidence,
            cited_evidence_numbers=(),
            model_id=draft.model_id,
            prompt_version=draft.prompt_version,
        )
    lines = tuple(
        f"{verification.statement} {' '.join(f'[{number}]' for number in verification.citation_numbers)}"
        for verification in accepted
    )
    cited = tuple(dict.fromkeys(number for verification in accepted for number in verification.citation_numbers))
    return GroundedAnswer(
        query=draft.query,
        answer="\n\n".join(lines),
        evidence=draft.evidence,
        cited_evidence_numbers=cited,
        model_id=draft.model_id,
        prompt_version=draft.prompt_version,
    )
