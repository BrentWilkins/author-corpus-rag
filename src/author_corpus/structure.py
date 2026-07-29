"""Structure-preserving segmentation for Markdown corpus documents."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, model_validator

MARKDOWN_STRUCTURE_VERSION = "markdown-sections-v2"

_HEADING_PATTERN = re.compile(r"^(?P<marks>#{1,6})[ \t]+(?P<title>.+?)[ \t]*#*[ \t]*(?:\n|$)")
_FENCE_PATTERN = re.compile(r"^[ \t]*(```|~~~)")


class MarkdownSection(BaseModel):
    """One exact contiguous source range under a Markdown heading path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ordinal: int = Field(ge=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    heading_path: tuple[str, ...] = ()
    text: str

    @model_validator(mode="after")
    def validate_offsets(self) -> MarkdownSection:
        """Require the text length to match its half-open source range."""
        if self.end <= self.start:
            raise ValueError("Markdown section end must be greater than its start.")
        if len(self.text) != self.end - self.start:
            raise ValueError("Markdown section text must exactly match its source range.")
        return self

    @property
    def heading(self) -> str | None:
        """Return the nearest section heading."""
        return self.heading_path[-1] if self.heading_path else None

    @property
    def heading_context(self) -> str:
        """Return a readable hierarchical heading path."""
        return " > ".join(self.heading_path)

    @property
    def has_body(self) -> bool:
        """Return whether the section contains evidence beyond its heading."""
        lines = self.text.splitlines()
        if lines and _HEADING_PATTERN.match(f"{lines[0]}\n"):
            lines = lines[1:]
        return bool("\n".join(lines).strip())

    @property
    def lead(self) -> str:
        """Return a compact first prose block for contextual embedding."""
        blocks = re.split(r"\n[ \t]*\n", self.text.strip(), maxsplit=2)
        for block in blocks:
            lines = [line for line in block.splitlines() if not _HEADING_PATTERN.match(f"{line}\n")]
            prose = " ".join(line.strip() for line in lines if line.strip())
            if prose:
                return prose[:500]
        return ""


def split_markdown_sections(content: str) -> tuple[MarkdownSection, ...]:
    """Split Markdown at ATX headings while preserving exact source ranges."""
    if not content:
        return ()

    boundaries: list[tuple[int, tuple[str, ...]]] = [(0, ())]
    heading_stack: list[tuple[int, str]] = []
    offset = 0
    fence_marker: str | None = None
    for line in content.splitlines(keepends=True):
        fence_match = _FENCE_PATTERN.match(line)
        if fence_match:
            marker = fence_match.group(1)
            fence_marker = None if fence_marker == marker else marker
        elif fence_marker is None:
            heading_match = _HEADING_PATTERN.match(line)
            if heading_match:
                level = len(heading_match.group("marks"))
                title = heading_match.group("title").strip()
                heading_stack = [(existing_level, value) for existing_level, value in heading_stack if existing_level < level]
                heading_stack.append((level, title))
                heading_path = tuple(value for _, value in heading_stack)
                if offset == 0:
                    boundaries[0] = (0, heading_path)
                else:
                    boundaries.append((offset, heading_path))
        offset += len(line)

    sections: list[MarkdownSection] = []
    for boundary_index, (start, heading_path) in enumerate(boundaries):
        end = boundaries[boundary_index + 1][0] if boundary_index + 1 < len(boundaries) else len(content)
        text = content[start:end]
        if not text.strip():
            continue
        sections.append(
            MarkdownSection(
                ordinal=len(sections) + 1,
                start=start,
                end=end,
                heading_path=heading_path,
                text=text,
            )
        )
    return tuple(sections)
