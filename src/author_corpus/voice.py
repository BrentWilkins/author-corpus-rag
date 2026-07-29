"""Conservative voice provenance for authored prose and direct quotations."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

VOICE_ANALYSIS_VERSION = "voice-v3"

VoiceKind = Literal["document_author", "quoted_speech", "uncertain"]
PassageVoice = Literal["document_author", "quoted_speech", "uncertain", "mixed"]

_STRAIGHT_QUOTE_PATTERN = re.compile(r'(?<![\w])"[^"\n]+"(?![\w])')
_BLOCKQUOTE_PATTERN = re.compile(r"(?m)^(?:[ \t]*>[^\n]*(?:\n|$))+")
_CURLY_MARK_PATTERN = re.compile(r"[“”]")
_PARAGRAPH_BREAK_PATTERN = re.compile(r"\n[ \t]*\n")
_ATTRIBUTION_VERBS = (
    "adds",
    "asks",
    "continues",
    "explains",
    "notes",
    "recalls",
    "recounts",
    "replies",
    "says",
    "said",
    "states",
    "tells",
    "writes",
)
_NAME = r"[A-Z][\w’'-]+(?:\s+[A-Z][\w’'-]+){0,3}"
_BEFORE_ATTRIBUTION = re.compile(
    rf"(?P<speaker>{_NAME})\s+(?:{'|'.join(_ATTRIBUTION_VERBS)})"
    r"(?:\s+that)?[\s,:—-]*$"
)
_AFTER_ATTRIBUTION = re.compile(
    rf"^[\s,;:—-]*(?:(?P<speaker_first>{_NAME})\s+(?:{'|'.join(_ATTRIBUTION_VERBS)})"
    rf"|(?:{'|'.join(_ATTRIBUTION_VERBS)})\s+(?P<speaker_last>{_NAME}))"
)


class VoiceSpan(BaseModel):
    """One exact source span assigned to a conservative voice category."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str
    kind: VoiceKind
    speaker: str | None = None

    @model_validator(mode="after")
    def validate_offsets(self) -> VoiceSpan:
        """Require a non-empty half-open source range."""
        if self.end <= self.start:
            raise ValueError("Voice span end must be greater than its start.")
        return self


class VoiceAnalysis(BaseModel):
    """Exact voice spans and aggregate proportions for one passage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = VOICE_ANALYSIS_VERSION
    text_length: int = Field(ge=0)
    spans: tuple[VoiceSpan, ...]
    passage_voice: PassageVoice
    document_author_characters: int = Field(ge=0)
    quoted_speech_characters: int = Field(ge=0)
    uncertain_characters: int = Field(ge=0)
    attributed_speakers: tuple[str, ...] = ()

    @property
    def document_author_fraction(self) -> float:
        """Return the proportion conservatively treated as author narration."""
        return 0.0 if self.text_length == 0 else self.document_author_characters / self.text_length

    @property
    def quoted_speech_fraction(self) -> float:
        """Return the proportion enclosed as direct quotation."""
        return 0.0 if self.text_length == 0 else self.quoted_speech_characters / self.text_length

    @property
    def uncertain_fraction(self) -> float:
        """Return the proportion whose voice is unsafe to infer."""
        return 0.0 if self.text_length == 0 else self.uncertain_characters / self.text_length


def analyze_voice(text: str) -> VoiceAnalysis:
    """Separate exact direct-quotation spans from document-author narration.

    This deterministic analysis deliberately does not infer that unattributed
    quotation belongs to a named person. Text outside explicit quotation marks,
    Markdown blockquotes, or malformed quote ranges is labeled as document-author
    narration, not as the voice of any one coauthor.
    """
    classifications: list[VoiceKind] = ["document_author"] * len(text)
    quoted_ranges, uncertain_ranges = _classified_quote_ranges(text)
    for start, end in uncertain_ranges:
        classifications[start:end] = ["uncertain"] * (end - start)
    for start, end in quoted_ranges:
        classifications[start:end] = ["quoted_speech"] * (end - start)

    spans: list[VoiceSpan] = []
    start = 0
    while start < len(text):
        kind = classifications[start]
        end = start + 1
        while end < len(text) and classifications[end] == kind:
            end += 1
        speaker = _attributed_speaker(text, start=start, end=end) if kind == "quoted_speech" else None
        spans.append(
            VoiceSpan(
                start=start,
                end=end,
                text=text[start:end],
                kind=kind,
                speaker=speaker,
            )
        )
        start = end

    return _build_analysis(text, spans)


def slice_voice_analysis(
    analysis: VoiceAnalysis,
    source_text: str,
    *,
    start: int,
    end: int,
) -> VoiceAnalysis:
    """Project complete-source voice spans onto one exact overlapping chunk."""
    if analysis.text_length != len(source_text):
        raise ValueError("Voice analysis length does not match the supplied source text.")
    if not 0 <= start < end <= len(source_text):
        raise ValueError("Voice slice must be a non-empty range within the source text.")

    excerpt = source_text[start:end]
    spans: list[VoiceSpan] = []
    for span in analysis.spans:
        overlap_start = max(start, span.start)
        overlap_end = min(end, span.end)
        if overlap_start >= overlap_end:
            continue
        relative_start = overlap_start - start
        relative_end = overlap_end - start
        spans.append(
            VoiceSpan(
                start=relative_start,
                end=relative_end,
                text=excerpt[relative_start:relative_end],
                kind=span.kind,
                speaker=span.speaker,
            )
        )
    return _build_analysis(excerpt, spans)


def _build_analysis(text: str, spans: list[VoiceSpan]) -> VoiceAnalysis:
    author_characters = sum(span.end - span.start for span in spans if span.kind == "document_author")
    quoted_characters = sum(span.end - span.start for span in spans if span.kind == "quoted_speech")
    uncertain_characters = sum(span.end - span.start for span in spans if span.kind == "uncertain")
    speakers = tuple(dict.fromkeys(span.speaker for span in spans if span.kind == "quoted_speech" and span.speaker is not None))
    return VoiceAnalysis(
        text_length=len(text),
        spans=tuple(spans),
        passage_voice=_passage_voice(author_characters, quoted_characters, uncertain_characters),
        document_author_characters=author_characters,
        quoted_speech_characters=quoted_characters,
        uncertain_characters=uncertain_characters,
        attributed_speakers=speakers,
    )


def _classified_quote_ranges(
    text: str,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    quoted = [
        (match.start(), match.end())
        for pattern in (_BLOCKQUOTE_PATTERN, _STRAIGHT_QUOTE_PATTERN)
        for match in pattern.finditer(text)
    ]
    uncertain: list[tuple[int, int]] = []
    for paragraph_start, paragraph_end in _paragraph_ranges(text):
        markers = [paragraph_start + match.start() for match in _CURLY_MARK_PATTERN.finditer(text[paragraph_start:paragraph_end])]
        paired_count = len(markers) - len(markers) % 2
        quoted.extend((markers[index], markers[index + 1] + 1) for index in range(0, paired_count, 2))
        if len(markers) % 2:
            marker = markers[-1]
            if text[marker] == "“":
                uncertain.append((marker, paragraph_end))
            else:
                uncertain.append((paragraph_start, marker + 1))
    return tuple(quoted), tuple(uncertain)


def _paragraph_ranges(text: str) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for match in _PARAGRAPH_BREAK_PATTERN.finditer(text):
        if start < match.start():
            ranges.append((start, match.start()))
        start = match.end()
    if start < len(text):
        ranges.append((start, len(text)))
    return tuple(ranges)


def _passage_voice(
    author_characters: int,
    quoted_characters: int,
    uncertain_characters: int,
) -> PassageVoice:
    populated = sum(value > 0 for value in (author_characters, quoted_characters, uncertain_characters))
    if populated > 1:
        return "mixed"
    if quoted_characters:
        return "quoted_speech"
    if uncertain_characters:
        return "uncertain"
    return "document_author"


def _attributed_speaker(text: str, *, start: int, end: int) -> str | None:
    before = text[max(0, start - 160) : start]
    before_match = _BEFORE_ATTRIBUTION.search(before)
    if before_match:
        return _named_speaker(before_match.group("speaker"))
    after = text[end : min(len(text), end + 160)]
    after_match = _AFTER_ATTRIBUTION.search(after)
    if not after_match:
        return None
    return _named_speaker(after_match.group("speaker_first") or after_match.group("speaker_last"))


def _named_speaker(value: str) -> str | None:
    non_names = {
        "dr",
        "he",
        "i",
        "it",
        "miss",
        "mr",
        "mrs",
        "ms",
        "professor",
        "she",
        "that",
        "the",
        "they",
        "this",
        "we",
        "who",
    }
    return None if value.casefold().rstrip(".") in non_names else value
