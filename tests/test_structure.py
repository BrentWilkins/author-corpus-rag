"""Tests for exact Markdown structure segmentation."""

from author_corpus.structure import split_markdown_sections


def test_preserves_nested_heading_paths_and_exact_ranges() -> None:
    """Split on real headings without rewriting the underlying evidence."""
    content = "Opening.\n\n# Main\n\nLead paragraph.\n\n## Detail\n\nSpecific evidence.\n"

    sections = split_markdown_sections(content)

    assert [section.heading_path for section in sections] == [(), ("Main",), ("Main", "Detail")]
    assert "".join(section.text for section in sections) == content
    assert all(content[section.start : section.end] == section.text for section in sections)
    assert sections[1].lead == "Lead paragraph."


def test_ignores_heading_syntax_inside_code_fences() -> None:
    """Do not invent document structure from fenced examples."""
    content = "# Real\n\n```markdown\n# Example only\n```\n\nAfter.\n"

    sections = split_markdown_sections(content)

    assert len(sections) == 1
    assert sections[0].heading_path == ("Real",)


def test_tracks_sibling_paths_when_heading_levels_begin_below_one() -> None:
    """Use actual heading levels rather than positions in a compact stack."""
    content = "## Parent\n\nIntro.\n\n### First\n\nOne.\n\n### Second\n\nTwo.\n"

    sections = split_markdown_sections(content)

    assert [section.heading_path for section in sections] == [
        ("Parent",),
        ("Parent", "First"),
        ("Parent", "Second"),
    ]


def test_identifies_heading_only_sections() -> None:
    """Allow indexing to omit structural labels that contain no evidence."""
    sections = split_markdown_sections("## Container\n\n### Evidence\n\nActual prose.\n")

    assert sections[0].has_body is False
    assert sections[1].has_body is True
