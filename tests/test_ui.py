"""Tests for the local Gradio interface boundary."""

from typing import cast

import gradio as gr

from author_corpus.service import CorpusQueryService
from author_corpus.ui import _user_history, build_chat_interface


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
