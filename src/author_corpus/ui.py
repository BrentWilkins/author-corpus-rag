"""Thin, local Gradio interface over the safe corpus query service."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import gradio as gr

from author_corpus.audit import AuditedClaim
from author_corpus.conversation import ConversationMessage, ConversationRole, ask_conversational
from author_corpus.review import (
    ClaimReviewAction,
    ClaimReviewProposal,
    ClaimReviewRecord,
    ClaimReviewWorkspace,
    ResolvedClaimStatus,
)
from author_corpus.service import CorpusQueryService

_REVIEW_ACTION_CHOICES: tuple[tuple[str, str], ...] = (
    ("Accept unchanged", "accept"),
    ("Revise and record", "revise"),
    ("Reject", "reject"),
)
_REVIEW_ACTIONS: dict[str, ClaimReviewAction] = {
    "accept": "accept",
    "revise": "revise",
    "reject": "reject",
}
_REVIEW_STATUS_CHOICES: tuple[tuple[str, str], ...] = (
    ("Supported", "supported"),
    ("Qualified", "qualified"),
    ("Contradicted", "contradicted"),
    ("Unsupported", "unsupported"),
)
_REVIEW_STATUSES: dict[str, ResolvedClaimStatus] = {
    "supported": "supported",
    "qualified": "qualified",
    "contradicted": "contradicted",
    "unsupported": "unsupported",
}


def build_chat_interface(
    service: CorpusQueryService,
    *,
    corpus_name: str = "Author Corpus",
) -> gr.ChatInterface:
    """Build a per-browser-session chat interface without shared global history."""

    def respond(message: str, history: list[dict[str, object]]) -> str:
        return chat_response(service, message, history)

    return gr.ChatInterface(
        fn=respond,
        title=f"{corpus_name} explorer",
        description=(
            "Exact metadata questions use the exhaustive catalog. Content questions use source-grounded retrieval. "
            "Each answer shows its route, sources, and timing."
        ),
        examples=[
            "How many articles has the author written here?",
            "Who are the other authors?",
            "What themes recur across the corpus?",
            "What practical advice does the author give readers?",
        ],
        flagging_mode="never",
        analytics_enabled=False,
        save_history=False,
        api_visibility="private",
    )


def chat_response(
    service: CorpusQueryService,
    message: str,
    history: list[dict[str, object]],
) -> str:
    """Convert Gradio history and execute one bounded conversational turn."""
    turn = ask_conversational(service, message, _user_history(history))
    return turn.to_markdown()


def build_corpus_interface(
    service: CorpusQueryService,
    *,
    corpus_name: str = "Author Corpus",
    review_workspace: ClaimReviewWorkspace | None = None,
) -> gr.Blocks:
    """Build local chat and explicit claim-review tabs."""
    with gr.Blocks(
        title=f"{corpus_name} explorer",
        analytics_enabled=False,
        fill_height=True,
    ) as interface:
        with gr.Tab("Corpus chat"):
            build_chat_interface(service, corpus_name=corpus_name)
        with gr.Tab("Claim review"):
            _render_claim_review_tab(review_workspace)
    return cast(gr.Blocks, interface)


def launch_chat_interface(
    service: CorpusQueryService,
    *,
    corpus_name: str,
    review_workspace: ClaimReviewWorkspace | None = None,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
    inbrowser: bool = False,
) -> None:
    """Launch local chat and review tabs with sharing disabled by default."""
    interface = build_corpus_interface(
        service,
        corpus_name=corpus_name,
        review_workspace=review_workspace,
    )
    interface.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        inbrowser=inbrowser,
        show_error=True,
    )


def claim_review_choices(workspace: ClaimReviewWorkspace) -> tuple[tuple[str, str], ...]:
    """Return stable Gradio labels and proposal IDs without exposing text as values."""
    choices: list[tuple[str, str]] = []
    for proposal in workspace.proposals:
        state = "reviewed" if workspace.reviews_for(proposal.proposal_id) else "pending"
        statement = _truncate(" ".join(proposal.statement.split()), length=72)
        choices.append((f"{state} · {proposal.claim_id} · {statement}", proposal.proposal_id))
    return tuple(choices)


def claim_review_form(
    workspace: ClaimReviewWorkspace,
    proposal_id: str | None,
) -> tuple[str, str, str]:
    """Render one proposal and safe defaults for an explicit revision."""
    if not proposal_id:
        return "Select a proposal to inspect its exact evidence.", "", "supported"
    proposal = workspace.get(proposal_id)
    if proposal.suggested_status is not None:
        status: ResolvedClaimStatus = proposal.suggested_status
    elif proposal.classifier_decision.label == "qualifies":
        status = "qualified"
    elif proposal.classifier_decision.label == "contradicts":
        status = "contradicted"
    else:
        status = "supported"
    return claim_review_markdown(workspace, proposal), proposal.statement, status


def claim_review_markdown(workspace: ClaimReviewWorkspace, proposal: ClaimReviewProposal) -> str:
    """Render one proposal with classifier limits and exact source evidence."""
    prior_reviews = workspace.reviews_for(proposal.proposal_id)
    review_state = "Pending human action" if not prior_reviews else _review_summary(prior_reviews[0])
    suggested = proposal.suggested_status or "revision required"
    source_lines = [f"- `{_escape_inline(uri)}`" for span in proposal.evidence_spans for uri in span.source_uris]
    evidence = "\n\n".join(_indented_text(span.text) for span in proposal.evidence_spans)
    return "\n".join(
        (
            f"### {_escape_inline(proposal.claim_id)}",
            "",
            f"**Review state:** {review_state}",
            f"**Classifier label:** `{proposal.classifier_decision.label}`",
            f"**Acceptable unchanged status:** `{suggested}`",
            f"**Classifier rationale:** {_escape_inline(proposal.classifier_decision.rationale)}",
            "",
            "**Proposed statement**",
            "",
            _indented_text(proposal.statement),
            "",
            "**Exact evidence**",
            "",
            evidence,
            "",
            "**Sources**",
            "",
            *(source_lines or ["- No source URI recorded."]),
        )
    )


def submit_claim_review(
    workspace: ClaimReviewWorkspace,
    *,
    proposal_id: str | None,
    reviewer: str,
    action: str,
    revised_statement: str,
    revised_status: str,
    attribution: str,
    qualifiers: str,
    notes: str,
) -> ClaimReviewRecord:
    """Validate and persist one explicit review-form submission."""
    if not proposal_id:
        raise ValueError("Select a claim-review proposal.")
    resolved_action = _REVIEW_ACTIONS.get(action)
    if resolved_action is None:
        raise ValueError("Choose accept, revise, or reject.")
    revised_claim: AuditedClaim | None = None
    if resolved_action == "revise":
        status = _REVIEW_STATUSES.get(revised_status)
        if status is None:
            raise ValueError("Choose a valid revised claim status.")
        proposal = workspace.get(proposal_id)
        revised_claim = AuditedClaim(
            claim_id=proposal.claim_id,
            statement=revised_statement,
            status=status,
            evidence_span_ids=tuple(span.span_id for span in proposal.evidence_spans),
            attribution=_optional_text(attribution),
            qualifiers=_lines(qualifiers),
            notes=_optional_text(notes),
        )
    return workspace.record(
        proposal_id,
        action=resolved_action,
        reviewer=reviewer,
        revised_claim=revised_claim,
        notes=notes,
    )


def recent_claim_reviews_markdown(workspace: ClaimReviewWorkspace, *, limit: int = 10) -> str:
    """Render recent durable human actions without implying ledger application."""
    reviews = workspace.store.recent(
        limit=limit,
        corpus_fingerprint=workspace.corpus_fingerprint,
    )
    if not reviews:
        return "No human claim reviews have been recorded."
    lines = ["### Recent durable reviews", ""]
    for review in reviews:
        status = review.audited_claim.status if review.audited_claim is not None else "no claim created"
        lines.append(
            f"- `{review.action}` · `{status}` · `{_escape_inline(review.proposal.claim_id)}` · "
            f"reviewer `{_escape_inline(review.reviewer)}` · {review.reviewed_at.isoformat()}"
        )
    return "\n".join(lines)


def _render_claim_review_tab(workspace: ClaimReviewWorkspace | None) -> None:
    if workspace is None:
        gr.Markdown(
            "No review queue is configured. Set `AUTHOR_CORPUS_CLAIM_EVAL` to a provenance-bearing local evaluation file."
        )
        return
    choices = claim_review_choices(workspace)
    if not choices:
        gr.Markdown("The configured claim evaluation contains no exact source spans, so no review proposals were created.")
        return

    initial_id = choices[0][1]
    initial_preview, initial_statement, initial_status = claim_review_form(workspace, initial_id)
    gr.Markdown(
        "Review actions are append-only. Accept preserves the proposal; revise requires an explicit resolved claim; "
        "reject records no claim. Recording a review does not automatically apply it to an answer ledger."
    )
    proposal = gr.Dropdown(
        choices=choices,
        value=initial_id,
        label="Source-bound proposal",
        interactive=True,
    )
    preview = gr.Markdown(initial_preview)
    reviewer = gr.Textbox(label="Reviewer name", placeholder="Required for every action")
    action = gr.Radio(
        choices=_REVIEW_ACTION_CHOICES,
        value="accept",
        label="Explicit action",
    )
    with gr.Accordion("Revision fields", open=False):
        revised_statement = gr.Textbox(
            value=initial_statement,
            label="Reviewed statement",
            lines=3,
        )
        revised_status = gr.Dropdown(
            choices=_REVIEW_STATUS_CHOICES,
            value=initial_status,
            label="Reviewed status",
        )
        attribution = gr.Textbox(label="Attribution", placeholder="Required when preserving a reported statement")
        qualifiers = gr.Textbox(
            label="Qualifiers",
            placeholder="One explicit qualifier per line",
            lines=3,
        )
    notes = gr.Textbox(label="Review notes", lines=2)
    submit = gr.Button("Record explicit review", variant="primary")
    outcome = gr.Markdown()
    recent = gr.Markdown(recent_claim_reviews_markdown(workspace))

    def select(selected_id: str | None) -> tuple[str, str, str]:
        return claim_review_form(workspace, selected_id)

    def record(
        selected_id: str | None,
        reviewer_name: str,
        selected_action: str,
        statement: str,
        status: str,
        named_attribution: str,
        qualifier_text: str,
        review_notes: str,
    ) -> tuple[str, str, str]:
        try:
            review = submit_claim_review(
                workspace,
                proposal_id=selected_id,
                reviewer=reviewer_name,
                action=selected_action,
                revised_statement=statement,
                revised_status=status,
                attribution=named_attribution,
                qualifiers=qualifier_text,
                notes=review_notes,
            )
        except ValueError as exc:
            current_preview, _, _ = claim_review_form(workspace, selected_id)
            return (
                f"**Review not recorded:** {_escape_inline(str(exc))}",
                current_preview,
                recent_claim_reviews_markdown(workspace),
            )
        current_preview, _, _ = claim_review_form(workspace, selected_id)
        return (
            f"**Recorded:** `{review.action}` as review `{review.review_id}`. No answer ledger was changed.",
            current_preview,
            recent_claim_reviews_markdown(workspace),
        )

    proposal.change(
        fn=select,
        inputs=proposal,
        outputs=(preview, revised_statement, revised_status),
        api_visibility="private",
    )
    submit.click(
        fn=record,
        inputs=(
            proposal,
            reviewer,
            action,
            revised_statement,
            revised_status,
            attribution,
            qualifiers,
            notes,
        ),
        outputs=(outcome, preview, recent),
        api_visibility="private",
    )


def _user_history(history: list[dict[str, object]]) -> tuple[ConversationMessage, ...]:
    messages: list[ConversationMessage] = []
    for raw_message in history:
        if raw_message.get("role") != "user":
            continue
        text = _content_text(raw_message.get("content"))
        if text:
            messages.append(ConversationMessage(role=ConversationRole.USER, content=text))
    return tuple(messages)


def _content_text(content: object) -> str | None:
    if isinstance(content, str):
        return content.strip() or None
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for block in content:
        if not isinstance(block, Mapping) or block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts) or None


def _review_summary(review: ClaimReviewRecord) -> str:
    status = review.audited_claim.status if review.audited_claim is not None else "no claim created"
    return f"Reviewed `{review.action}` by `{_escape_inline(review.reviewer)}` at {review.reviewed_at.isoformat()} (`{status}`)"


def _lines(value: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in value.splitlines() if line.strip())


def _optional_text(value: str) -> str | None:
    normalized = " ".join(value.strip().split())
    return normalized or None


def _truncate(value: str, *, length: int) -> str:
    return value if len(value) <= length else f"{value[: length - 1]}…"


def _escape_inline(value: str) -> str:
    return value.replace("\\", r"\\").replace("`", r"\`").replace("<", r"\<").replace(">", r"\>")


def _indented_text(value: str) -> str:
    return "\n".join(f"    {line}" for line in value.splitlines()) or "    "
