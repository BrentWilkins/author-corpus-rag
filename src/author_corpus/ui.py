"""Thin, local Gradio interface over the safe corpus query service."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol, cast

import gradio as gr

from author_corpus.answer_review import AnswerReviewRecord, AnswerReviewStore, review_trace_answer
from author_corpus.audit import AuditedClaim
from author_corpus.claim_extraction import AnswerClaimExtraction, extract_answer_claims
from author_corpus.conversation import ConversationMessage, ConversationRole, ask_conversational
from author_corpus.models import CorpusDocument
from author_corpus.review import (
    ClaimReviewAction,
    ClaimReviewProposal,
    ClaimReviewRecord,
    ClaimReviewWorkspace,
    ResolvedClaimStatus,
)
from author_corpus.scope import AuthorScope
from author_corpus.service import CorpusQueryService, VerifierMode
from author_corpus.tracing import QueryTrace, QueryTraceStore

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
_CHATBOT_HEIGHT = "72vh"
_VERIFIER_CHOICES: tuple[tuple[str, str], ...] = (
    ("Conservative lexical baseline", "conservative"),
    ("Structured semantic verifier (experimental)", "semantic"),
)
_CORPUS_SCOPE_VALUE = "__entire_corpus__"


class _EventMethod(Protocol):
    """Describe the stable subset of a dynamically exposed Gradio event method."""

    def __call__(
        self,
        *,
        fn: Callable[..., object],
        inputs: object,
        outputs: object,
        api_visibility: str,
    ) -> object:
        """Bind a callback to one component event."""
        ...


def _event_method(component: object, name: str) -> _EventMethod:
    """Return a checked Gradio event method despite platform-dependent typing."""
    method = getattr(component, name, None)
    if not callable(method):
        raise RuntimeError(f"Gradio component does not expose the {name!r} event method.")
    return cast(_EventMethod, method)


def build_chat_interface(
    service: CorpusQueryService,
    *,
    corpus_name: str = "Author Corpus",
) -> gr.ChatInterface:
    """Build a per-browser-session chat interface without shared global history."""
    reasoning = gr.Checkbox(
        value=False,
        label="Use bounded claim-level reasoning",
        info="Slower: retrieves per subquestion, verifies cited claims, and may make one corrective pass.",
    )
    verifier = gr.Dropdown(
        choices=_VERIFIER_CHOICES,
        value="conservative",
        label="Claim verifier",
        info="The semantic alternative adds model calls and is not the default until holdout evaluation supports it.",
    )
    author_choices, default_authors = _author_scope_options(service)
    authors = gr.Dropdown(
        choices=author_choices,
        value=default_authors,
        multiselect=True,
        label="Author scope",
        info="Choose one author, select multiple for a comparison, or choose Entire corpus. Filtering is exact.",
    )

    def respond(
        message: str,
        history: list[dict[str, object]],
        use_reasoning: bool,
        verifier_mode: str,
        selected_authors: list[str],
    ) -> str:
        return chat_response(
            service,
            message,
            history,
            reason=use_reasoning,
            verifier=verifier_mode,
            authors=selected_authors,
        )

    return gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(
            height=_CHATBOT_HEIGHT,
            min_height=480,
            autoscroll=True,
        ),
        title=f"{corpus_name} explorer",
        description=(
            "Exact metadata questions use the exhaustive catalog. Content questions use source-grounded retrieval. "
            "Each answer shows its route, sources, and timing."
        ),
        additional_inputs=[reasoning, verifier, authors],
        additional_inputs_accordion="Reasoning options",
        examples=[
            ["How many articles has the author written here?", False, "conservative", default_authors],
            ["Who are the other authors?", False, "conservative", default_authors],
            ["What themes recur across the corpus?", False, "conservative", default_authors],
            ["How did the author's advice change over time?", True, "conservative", default_authors],
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
    *,
    reason: bool = False,
    verifier: str = "conservative",
    authors: object = None,
) -> str:
    """Convert Gradio history and execute one bounded conversational turn."""
    verifier_mode = _verifier_mode(verifier)
    turn = ask_conversational(
        service,
        message,
        _user_history(history),
        reason=reason,
        verifier=verifier_mode,
        author_scope=_author_scope(authors),
    )
    return turn.to_markdown()


def build_corpus_interface(
    service: CorpusQueryService,
    *,
    corpus_name: str = "Author Corpus",
    review_workspace: ClaimReviewWorkspace | None = None,
    trace_store: QueryTraceStore | None = None,
    answer_review_store: AnswerReviewStore | None = None,
    documents: Mapping[str, CorpusDocument] | None = None,
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
            _render_claim_review_tab(review_workspace, trace_store)
        with gr.Tab("Answer review"):
            _render_answer_review_tab(answer_review_store, trace_store, documents)
    return cast(gr.Blocks, interface)


def launch_chat_interface(
    service: CorpusQueryService,
    *,
    corpus_name: str,
    review_workspace: ClaimReviewWorkspace | None = None,
    trace_store: QueryTraceStore | None = None,
    answer_review_store: AnswerReviewStore | None = None,
    documents: Mapping[str, CorpusDocument] | None = None,
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
        trace_store=trace_store,
        answer_review_store=answer_review_store,
        documents=documents,
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


def trace_claim_choices(store: QueryTraceStore, *, limit: int = 20) -> tuple[tuple[str, str], ...]:
    """Return recent trace labels and stable IDs for read-only inspection."""
    return tuple(
        (
            f"{trace.created_at.isoformat()} · {trace.answer_status} · {_truncate(trace.user_query or trace.query, length=90)}",
            trace.trace_id,
        )
        for trace in store.recent(limit=limit)
    )


def trace_claim_report(store: QueryTraceStore, trace_id: str | None) -> str:
    """Render extracted generated claims without creating review proposals."""
    if not trace_id:
        return "Select a generated-answer trace."
    trace = store.get(trace_id)
    if trace is None:
        raise ValueError(f"Unknown query trace: {trace_id!r}.")
    return answer_claim_extraction_markdown(extract_answer_claims(trace))


def answer_claim_extraction_markdown(extraction: AnswerClaimExtraction) -> str:
    """Render citation-bound candidates, provenance coverage, and uncited prose."""
    covered, total = extraction.exact_span_coverage
    lines = [
        "### Read-only generated-answer claim preview",
        "",
        f"**Question:** {_escape_inline(extraction.query)}",
        f"**Citation-bound candidates:** {total}",
        f"**Exact-span coverage:** {covered}/{total}",
        "",
        "This preview does not create a proposal, review record, or audited claim.",
    ]
    if not extraction.candidates:
        lines.extend(("", "No citation-bound candidate sentences were found."))
    for candidate in extraction.candidates:
        provenance = "source-bound" if candidate.is_source_bound else "unversioned"
        lines.extend(
            (
                "",
                f"#### Candidate {candidate.ordinal} · `{provenance}`",
                "",
                _indented_text(candidate.statement),
                "",
                f"Citations: {', '.join(f'[{number}]' for number in candidate.citation_numbers)}",
            )
        )
        for assessment in candidate.evidence:
            decision = assessment.classifier_decision
            label = decision.label if decision is not None else "not classified: no exact span"
            title = assessment.title or "unknown evidence"
            lines.append(f"- [{assessment.evidence_number}] {_escape_inline(title)} · heuristic `{label}`")
            if assessment.evidence_span is not None:
                excerpt = _truncate(_normalize_display(assessment.evidence_span.text), length=420)
                lines.extend(("", _indented_text(excerpt)))
    if extraction.uncited_segments:
        lines.extend(("", "### Uncited generated segments", ""))
        lines.extend(f"- {_escape_inline(segment)}" for segment in extraction.uncited_segments)
    return "\n".join(lines)


def answer_review_form(
    trace_store: QueryTraceStore,
    review_store: AnswerReviewStore,
    trace_id: str | None,
) -> tuple[str, str]:
    """Render one generated answer and its durable review state."""
    if not trace_id:
        return "Select a generated answer to review.", ""
    trace = trace_store.get(trace_id)
    if trace is None:
        raise ValueError(f"Unknown query trace: {trace_id!r}.")
    covered, total = trace.cited_span_coverage
    prior = review_store.for_trace(trace.trace_id)
    state = "Pending explicit review" if not prior else _answer_review_summary(prior[0])
    sources = "\n".join(
        f"- [{item.evidence_number}] {_escape_inline(item.title)}"
        + (f" — {_escape_inline(item.canonical_source_uri)}" if item.canonical_source_uri else "")
        for item in trace.evidence
    )
    attempts = _generation_attempts_markdown(trace)
    preview = "\n".join(
        (
            f"### {_escape_inline(trace.user_query or trace.query)}",
            "",
            f"**Review state:** {state}",
            f"**Exact cited-span coverage:** {covered}/{total}",
            f"**Author scope:** `{_escape_inline(trace.author_scope.cache_key)}`",
            f"**Generation outcome:** `{trace.answer_status}`",
            f"**Generation attempts:** {len(trace.generation_attempts)}",
            "",
            trace.answer,
            attempts,
            "",
            "### Frozen evidence",
            "",
            sources or "No evidence was recorded.",
        )
    )
    return preview, trace.answer


def _generation_attempts_markdown(trace: QueryTrace) -> str:
    if not trace.generation_attempts:
        return ""
    lines = ["", "### Private generation attempts", ""]
    for attempt in trace.generation_attempts:
        citations = ", ".join(f"[{number}]" for number in attempt.valid_citation_numbers) or "none"
        lines.extend(
            (
                f"**Attempt {attempt.ordinal}; valid citations: {citations}**",
                "",
                _indented_text(attempt.output or "<empty model output>"),
                "",
            )
        )
    return "\n".join(lines).rstrip()


def submit_answer_review(
    trace_store: QueryTraceStore,
    review_store: AnswerReviewStore,
    documents: Mapping[str, CorpusDocument],
    *,
    trace_id: str | None,
    reviewer: str,
    action: str,
    revised_answer: str,
    notes: str,
) -> AnswerReviewRecord:
    """Validate and persist one explicit whole-answer decision."""
    if not trace_id:
        raise ValueError("Select a generated answer to review.")
    trace = trace_store.get(trace_id)
    if trace is None:
        raise ValueError(f"Unknown query trace: {trace_id!r}.")
    resolved_action = _REVIEW_ACTIONS.get(action)
    if resolved_action is None:
        raise ValueError("Choose accept, revise, or reject.")
    if review_store.for_trace(trace_id):
        raise ValueError("This generated answer already has a durable review.")
    review = review_trace_answer(
        trace,
        documents=documents,
        action=resolved_action,
        reviewer=reviewer,
        revised_answer=revised_answer if resolved_action == "revise" else None,
        notes=notes,
    )
    review_store.put(review)
    return review


def recent_answer_reviews_markdown(
    store: AnswerReviewStore,
    *,
    limit: int = 10,
) -> str:
    """Render recent whole-answer review decisions."""
    reviews = store.recent(limit=limit)
    if not reviews:
        return "No whole-answer reviews have been recorded."
    lines = ["### Recent answer reviews", ""]
    for review in reviews:
        lines.append(
            f"- `{review.action}` · trace `{review.trace_id}` · "
            f"reviewer `{_escape_inline(review.reviewer)}` · {review.reviewed_at.isoformat()}"
        )
    return "\n".join(lines)


def _render_claim_review_tab(
    workspace: ClaimReviewWorkspace | None,
    trace_store: QueryTraceStore | None,
) -> None:
    with gr.Tab("Configured proposals"):
        _render_configured_review_queue(workspace)
    with gr.Tab("Generated claims — read only"):
        _render_generated_claim_preview(trace_store)


def _render_configured_review_queue(workspace: ClaimReviewWorkspace | None) -> None:
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

    _event_method(proposal, "change")(
        fn=select,
        inputs=proposal,
        outputs=(preview, revised_statement, revised_status),
        api_visibility="private",
    )
    _event_method(submit, "click")(
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


def _render_generated_claim_preview(store: QueryTraceStore | None) -> None:
    if store is None:
        gr.Markdown("No generated-answer trace store is configured.")
        return
    choices = trace_claim_choices(store)
    if not choices:
        gr.Markdown("No generated-answer traces have been recorded.")
        return
    initial_id = choices[0][1]
    trace = gr.Dropdown(
        choices=choices,
        value=initial_id,
        label="Generated-answer trace",
        interactive=True,
    )
    report = gr.Markdown(trace_claim_report(store, initial_id))

    def select(trace_id: str | None) -> str:
        try:
            return trace_claim_report(store, trace_id)
        except ValueError as exc:
            return f"**Trace not available:** {_escape_inline(str(exc))}"

    _event_method(trace, "change")(
        fn=select,
        inputs=trace,
        outputs=report,
        api_visibility="private",
    )


def _render_answer_review_tab(
    review_store: AnswerReviewStore | None,
    trace_store: QueryTraceStore | None,
    documents: Mapping[str, CorpusDocument] | None,
) -> None:
    if review_store is None or trace_store is None or documents is None:
        gr.Markdown("Answer review requires the trace store, review store, and current corpus documents.")
        return
    choices = trace_claim_choices(trace_store)
    if not choices:
        gr.Markdown("No generated-answer traces have been recorded.")
        return
    initial_id = choices[0][1]
    trace = gr.Dropdown(choices=choices, value=initial_id, label="Generated answer", interactive=True)
    initial_preview, initial_answer = answer_review_form(trace_store, review_store, initial_id)
    preview = gr.Markdown(initial_preview)
    gr.Markdown(
        "Accept preserves the answer, revise requires a cited replacement, and reject creates no training example. "
        "Accepted and revised answers require current exact spans for every citation."
    )
    reviewer = gr.Textbox(label="Reviewer identity", placeholder="Required")
    action = gr.Radio(choices=_REVIEW_ACTION_CHOICES, value="accept", label="Explicit action")
    revised_answer = gr.Textbox(value=initial_answer, label="Reviewed answer", lines=10)
    notes = gr.Textbox(label="Review notes", lines=3)
    submit = gr.Button("Record durable answer review", variant="primary")
    outcome = gr.Markdown()
    recent = gr.Markdown(recent_answer_reviews_markdown(review_store))

    def select(trace_id: str | None) -> tuple[str, str]:
        try:
            return answer_review_form(trace_store, review_store, trace_id)
        except ValueError as exc:
            return f"**Trace not available:** {_escape_inline(str(exc))}", ""

    _event_method(trace, "change")(
        fn=select,
        inputs=trace,
        outputs=(preview, revised_answer),
        api_visibility="private",
    )

    def record(
        trace_id: str | None,
        reviewer_value: str,
        action_value: str,
        revised_value: str,
        notes_value: str,
    ) -> tuple[str, str, str]:
        try:
            review = submit_answer_review(
                trace_store,
                review_store,
                documents,
                trace_id=trace_id,
                reviewer=reviewer_value,
                action=action_value,
                revised_answer=revised_value,
                notes=notes_value,
            )
        except ValueError as exc:
            current_preview, _ = answer_review_form(trace_store, review_store, trace_id)
            return (
                f"**Review not recorded:** {_escape_inline(str(exc))}",
                current_preview,
                recent_answer_reviews_markdown(review_store),
            )
        current_preview, _ = answer_review_form(trace_store, review_store, trace_id)
        return (
            f"Recorded immutable `{review.action}` review `{review.review_id}`.",
            current_preview,
            recent_answer_reviews_markdown(review_store),
        )

    _event_method(submit, "click")(
        fn=record,
        inputs=(trace, reviewer, action, revised_answer, notes),
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


def _verifier_mode(value: str) -> VerifierMode:
    normalized = value.strip().lower()
    if normalized == "conservative":
        return "conservative"
    if normalized == "semantic":
        return "semantic"
    raise ValueError(f"Unknown verifier mode: {value!r}.")


def _author_scope_options(service: CorpusQueryService) -> tuple[tuple[tuple[str, str], ...], list[str]]:
    if not isinstance(service, CorpusQueryService):
        return ((("Entire corpus", _CORPUS_SCOPE_VALUE),), [_CORPUS_SCOPE_VALUE])
    choices = (("Entire corpus", _CORPUS_SCOPE_VALUE),) + tuple((author, author) for author in service.available_authors)
    default = list(service.default_scope.authors) or [_CORPUS_SCOPE_VALUE]
    return choices, default


def _author_scope(value: object) -> AuthorScope | None:
    if value is None:
        return None
    if isinstance(value, str):
        authors = (value,)
    elif isinstance(value, list) and all(isinstance(author, str) for author in value):
        authors = tuple(value)
    else:
        raise ValueError("Author selection must be a list of configured author names.")
    normalized = tuple(author.strip() for author in authors if author.strip())
    if not normalized:
        return None
    if _CORPUS_SCOPE_VALUE in normalized:
        if len(normalized) != 1:
            raise ValueError("Entire corpus cannot be combined with individual authors.")
        return AuthorScope()
    if len(normalized) == 1:
        return AuthorScope.for_author(normalized[0])
    return AuthorScope(kind="comparison", authors=normalized)


def _review_summary(review: ClaimReviewRecord) -> str:
    status = review.audited_claim.status if review.audited_claim is not None else "no claim created"
    return f"Reviewed `{review.action}` by `{_escape_inline(review.reviewer)}` at {review.reviewed_at.isoformat()} (`{status}`)"


def _answer_review_summary(review: AnswerReviewRecord) -> str:
    return f"Reviewed `{review.action}` by `{_escape_inline(review.reviewer)}` at {review.reviewed_at.isoformat()}"


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


def _normalize_display(value: str) -> str:
    return " ".join(value.strip().split())
