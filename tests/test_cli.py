"""Tests for command-line entry-point defaults."""

from author_corpus.cli import _parser


def test_chat_command_is_local_and_private_by_default() -> None:
    """Require explicit flags before binding broadly or requesting a share URL."""
    arguments = _parser().parse_args(["chat"])

    assert arguments.server_name == "127.0.0.1"
    assert arguments.server_port == 7860
    assert arguments.share is False
    assert arguments.inbrowser is False
