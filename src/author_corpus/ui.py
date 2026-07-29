"""Thin, local Gradio interface over the safe corpus query service."""

from __future__ import annotations

from collections.abc import Mapping

import gradio as gr

from author_corpus.conversation import ConversationMessage, ConversationRole, ask_conversational
from author_corpus.service import CorpusQueryService


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


def launch_chat_interface(
    service: CorpusQueryService,
    *,
    corpus_name: str,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
    inbrowser: bool = False,
) -> None:
    """Launch the local chat interface with sharing disabled by default."""
    interface = build_chat_interface(service, corpus_name=corpus_name)
    interface.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        inbrowser=inbrowser,
        show_error=True,
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
