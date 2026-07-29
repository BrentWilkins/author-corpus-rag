"""Tests for conservative, explicitly configured author identity resolution."""

import pytest

from author_corpus.identity import AuthorIdentity, parse_author_aliases, resolve_author_query


def _identity(*aliases: str) -> AuthorIdentity:
    return AuthorIdentity.from_catalog(
        default_author="Avery Stone",
        aliases=aliases,
        catalog_authors=("Avery Stone", "Jamie River"),
    )


def test_resolves_trusted_alias_and_adds_profile_context() -> None:
    """Use an explicit nickname as corpus identity without retaining it as a search keyword."""
    resolution = resolve_author_query("What are Av's three favorite things?", _identity("Av"))

    assert resolution.retrieval_query == "corpus author personal facts biography interests preferences"
    assert resolution.grounding_question == "What are Avery Stone's three favorite things?"
    assert resolution.canonical_author == "Avery Stone"
    assert resolution.matched_references == ("Av's",)
    assert resolution.added_profile_context is True


def test_does_not_guess_an_unconfigured_nickname() -> None:
    """Prefer a missed link over fuzzy nickname inference."""
    query = "Does Aves have any siblings?"

    resolution = resolve_author_query(query, _identity("Av"))

    assert resolution.retrieval_query == query
    assert resolution.grounding_question == query
    assert resolution.matched_references == ()
    assert resolution.canonical_author is None


def test_resolves_generic_default_author_reference() -> None:
    """Connect generic author wording to the configured corpus identity."""
    resolution = resolve_author_query("What did the main author write about?", _identity())

    assert resolution.retrieval_query == "What did the corpus author write about?"
    assert resolution.grounding_question == "What did Avery Stone write about?"
    assert resolution.canonical_author == "Avery Stone"


def test_canonical_name_is_not_expanded_again_by_a_first_name_alias() -> None:
    """Canonicalize a name once when a trusted alias is also its first name."""
    resolution = resolve_author_query("What does Avery Stone recommend?", _identity("Avery"))

    assert resolution.retrieval_query == "What does the corpus author recommend?"
    assert resolution.grounding_question == "What does Avery Stone recommend?"


def test_rejects_alias_colliding_with_another_catalog_author() -> None:
    """Fail closed when an alias could identify a different credited writer."""
    with pytest.raises(ValueError, match="collides"):
        _identity("Jamie")


@pytest.mark.parametrize("value", ['"Av"', '["Av", 3]', "Av"])
def test_alias_environment_requires_json_string_array(value: str) -> None:
    """Reject ambiguous environment formats instead of guessing delimiters."""
    with pytest.raises(ValueError, match="JSON array"):
        parse_author_aliases(value)


def test_alias_environment_parses_explicit_array() -> None:
    """Accept a private JSON list while stripping redundant whitespace."""
    assert parse_author_aliases('[" Av ", "A. Stone"]') == ("Av", "A. Stone")
