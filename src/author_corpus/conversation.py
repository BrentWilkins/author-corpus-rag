"""Bounded conversational context that cannot override query routing."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from author_corpus.routing import QueryRoute, route_query
from author_corpus.service import CorpusQueryResult, CorpusQueryService


class ConversationRole(StrEnum):
    """Roles retained from an interactive chat history."""

    USER = "user"
    ASSISTANT = "assistant"


class ConversationMessage(BaseModel):
    """One text-only message supplied by a conversation interface."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: ConversationRole
    content: str = Field(min_length=1)


class ConversationResolution(BaseModel):
    """Inspectable relationship between a user question and retrieval query."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str
    retrieval_query: str
    used_previous_user_query: bool
    previous_user_query: str | None = None
    rationale: str


class ConversationTurn(BaseModel):
    """A context resolution paired with its routed execution result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    resolution: ConversationResolution
    result: CorpusQueryResult

    def to_markdown(self) -> str:
        """Render the result and disclose when bounded context was added."""
        body = self.result.to_markdown()
        if not self.resolution.used_previous_user_query:
            return body
        return f"{body}\n\n_Context used: previous user question only._"


_FOLLOW_UP_SIGNAL = re.compile(
    r"^(what about|how about|and\b|also\b|then\b|why\b|how so\b|does that\b|do they\b|"
    r"is that\b|are those\b|which of those\b|tell me more\b|expand on\b|can you elaborate\b)|"
    r"\b(this|that|these|those|it|same)\b",
    re.IGNORECASE,
)


def resolve_conversation_query(
    query: str,
    history: tuple[ConversationMessage, ...],
) -> ConversationResolution:
    """Add at most the previous user question to a semantic follow-up.

    Exact questions are never rewritten. Assistant messages are never fed back
    into retrieval because a previous generated answer is not source evidence.
    """
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("Query must not be empty.")
    decision = route_query(normalized_query)
    if decision.route is QueryRoute.EXACT_CATALOG:
        return ConversationResolution(
            query=normalized_query,
            retrieval_query=normalized_query,
            used_previous_user_query=False,
            rationale="Exact catalog questions are executed from their literal wording.",
        )

    previous = next(
        (message.content.strip() for message in reversed(history) if message.role is ConversationRole.USER),
        None,
    )
    if previous is None or _FOLLOW_UP_SIGNAL.search(normalized_query) is None:
        return ConversationResolution(
            query=normalized_query,
            retrieval_query=normalized_query,
            used_previous_user_query=False,
            rationale="The question is self-contained or no previous user question is available.",
        )

    retrieval_query = f"{normalized_query}\n\nPrevious user question for context: {previous}"
    return ConversationResolution(
        query=normalized_query,
        retrieval_query=retrieval_query,
        used_previous_user_query=True,
        previous_user_query=previous,
        rationale="A bounded semantic follow-up uses only the immediately preceding user question.",
    )


def ask_conversational(
    service: CorpusQueryService,
    query: str,
    history: tuple[ConversationMessage, ...] = (),
    *,
    generate: bool = True,
    reason: bool = False,
) -> ConversationTurn:
    """Resolve bounded context and execute through the normal safe query service."""
    resolution = resolve_conversation_query(query, history)
    result = service.ask(
        resolution.query,
        retrieval_query=resolution.retrieval_query,
        generate=generate,
        reason=reason,
    )
    return ConversationTurn(resolution=resolution, result=result)
