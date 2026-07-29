"""Tests for command-line entry-point defaults."""

from author_corpus.cli import _parser


def test_chat_command_is_local_and_private_by_default() -> None:
    """Require explicit flags before binding broadly or requesting a share URL."""
    arguments = _parser().parse_args(["chat"])

    assert arguments.server_name == "127.0.0.1"
    assert arguments.server_port == 7860
    assert arguments.share is False
    assert arguments.inbrowser is False


def test_reviewed_export_defaults_to_ignored_private_data() -> None:
    """Keep future training examples out of version control by default."""
    arguments = _parser().parse_args(["export-reviewed"])

    assert str(arguments.output) == "data/private/reviewed-answers.jsonl"
    assert arguments.corpus_fingerprint is None


def test_generated_claim_evaluation_uses_private_cases_by_default() -> None:
    """Allow an ignored environment path without committing corpus labels."""
    arguments = _parser().parse_args(["evaluate-generated-claims"])

    assert arguments.cases is None
