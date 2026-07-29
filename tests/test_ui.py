"""Tests for the local Gradio interface boundary."""

from pathlib import Path
from typing import cast

import gradio as gr
import pytest

from author_corpus.audit import EvidenceSpan
from author_corpus.claim_classification import classify_claim_evidence
from author_corpus.ingestion import load_corpus
from author_corpus.persistence import corpus_fingerprint
from author_corpus.review import ClaimReviewProposal, ClaimReviewStore, ClaimReviewWorkspace
from author_corpus.service import CorpusQueryService
from author_corpus.ui import (
    _user_history,
    build_chat_interface,
    build_corpus_interface,
    claim_review_choices,
    claim_review_form,
    submit_claim_review,
)


def test_builds_gradio_chat_without_starting_a_server() -> None:
    """Construct the UI independently from corpus loading and server launch."""
    service = cast(CorpusQueryService, object())

    interface = build_chat_interface(service, corpus_name="Synthetic Corpus")

    assert isinstance(interface, gr.ChatInterface)
    assert interface.title == "Synthetic Corpus explorer"
    assert interface.analytics_enabled is False
    assert interface.save_history is False


def test_gradio_history_keeps_user_text_and_discards_assistant_output() -> None:
    """Convert both Gradio 6 content blocks and plain strings without model text."""
    messages = _user_history(
        [
            {"role": "user", "content": [{"type": "text", "text": "First question"}]},
            {"role": "assistant", "content": "Generated answer"},
            {"role": "user", "content": "Follow-up question"},
            {"role": "user", "content": [{"type": "image", "path": "/tmp/ignored.png"}]},
        ]
    )

    assert [message.content for message in messages] == ["First question", "Follow-up question"]


def test_builds_combined_chat_and_review_interface(tmp_path: Path) -> None:
    """Expose local review controls without starting a Gradio server."""
    service = cast(CorpusQueryService, object())
    workspace = _review_workspace(tmp_path)

    interface = build_corpus_interface(
        service,
        corpus_name="Synthetic Corpus",
        review_workspace=workspace,
    )

    assert isinstance(interface, gr.Blocks)
    assert interface.title == "Synthetic Corpus explorer"


def test_review_form_shows_exact_evidence_and_stable_proposal_value(tmp_path: Path) -> None:
    """Keep display labels separate from stable IDs and show source text before action."""
    workspace = _review_workspace(tmp_path)
    proposal = workspace.proposals[0]

    choices = claim_review_choices(workspace)
    preview, statement, status = claim_review_form(workspace, proposal.proposal_id)

    assert choices[0][1] == proposal.proposal_id
    assert "pending" in choices[0][0]
    assert proposal.evidence_spans[0].text in preview
    assert statement == proposal.statement
    assert status == "supported"


def test_review_submission_requires_identity_and_records_once(tmp_path: Path) -> None:
    """Make the explicit action durable while refusing anonymous or repeated review."""
    workspace = _review_workspace(tmp_path)
    proposal = workspace.proposals[0]
    arguments = {
        "proposal_id": proposal.proposal_id,
        "action": "accept",
        "revised_statement": proposal.statement,
        "revised_status": "supported",
        "attribution": "",
        "qualifiers": "",
        "notes": "Checked exact evidence.",
    }

    with pytest.raises(ValueError, match="Reviewer"):
        submit_claim_review(workspace, reviewer="", **arguments)

    review = submit_claim_review(workspace, reviewer="Local reviewer", **arguments)

    assert review.action == "accept"
    assert workspace.store.get(review.review_id) == review
    with pytest.raises(ValueError, match="already has"):
        submit_claim_review(workspace, reviewer="Local reviewer", **arguments)


def test_review_submission_can_revise_with_explicit_qualifier(tmp_path: Path) -> None:
    """Turn a qualification into an audited claim only when its scope is written."""
    workspace = _review_workspace(tmp_path, qualified=True)
    proposal = workspace.proposals[0]

    review = submit_claim_review(
        workspace,
        proposal_id=proposal.proposal_id,
        reviewer="Local reviewer",
        action="revise",
        revised_statement="The bridge closes above the posted threshold.",
        revised_status="qualified",
        attribution="",
        qualifiers="Only above the posted threshold.",
        notes="Narrowed the original statement.",
    )

    assert review.audited_claim is not None
    assert review.audited_claim.qualifiers == ("Only above the posted threshold.",)


def _review_workspace(tmp_path: Path, *, qualified: bool = False) -> ClaimReviewWorkspace:
    source_path = tmp_path / "article.md"
    evidence = (
        "The bridge closes only when temperatures exceed the posted threshold."
        if qualified
        else "The bridge closes during extreme heat."
    )
    source_path.write_text(evidence, encoding="utf-8")
    document = load_corpus([source_path]).documents[0]
    span = EvidenceSpan.from_document(document, evidence)
    statement = "The bridge closes." if qualified else evidence
    decision = classify_claim_evidence(statement, evidence, evidence_voice="document_author")
    proposal = ClaimReviewProposal.from_decision(
        decision,
        corpus_fingerprint=corpus_fingerprint([document]),
        claim_id="bridge-closure",
        evidence_spans=(span,),
    )
    return ClaimReviewWorkspace(
        proposals=(proposal,),
        store=ClaimReviewStore(tmp_path / "reviews.sqlite3"),
    )
