"""Tests for conservative author-narration and quotation provenance."""

from author_corpus.voice import analyze_voice, slice_voice_analysis


def test_separates_exact_quoted_speech_and_explicit_speaker() -> None:
    """Preserve source offsets and accept only a named attribution."""
    text = "The reporter introduces the subject. Rowan Vale says, “Carry water and turn around early.” Then narration resumes."

    analysis = analyze_voice(text)

    quote = next(span for span in analysis.spans if span.kind == "quoted_speech")
    assert text[quote.start : quote.end] == quote.text
    assert quote.text == "“Carry water and turn around early.”"
    assert quote.speaker == "Rowan Vale"
    assert analysis.passage_voice == "mixed"
    assert analysis.attributed_speakers == ("Rowan Vale",)


def test_does_not_guess_a_name_from_pronoun_attribution() -> None:
    """Keep a speaker unknown when the local attribution only says she."""
    analysis = analyze_voice("She explains, “The trail taught me patience.”")

    quote = next(span for span in analysis.spans if span.kind == "quoted_speech")

    assert quote.speaker is None
    assert analysis.attributed_speakers == ()


def test_does_not_treat_an_honorific_as_a_speaker_name() -> None:
    """Prefer unknown over an incomplete title-only attribution."""
    analysis = analyze_voice("Dr says, “The finding needs more study.”")

    quote = next(span for span in analysis.spans if span.kind == "quoted_speech")

    assert quote.speaker is None


def test_projects_quote_provenance_across_a_chunk_boundary() -> None:
    """Retain quoted status when a sliced chunk contains no quote marks."""
    source = "An introduction. “" + ("quoted material " * 20) + "” Closing narration."
    analysis = analyze_voice(source)
    quote = next(span for span in analysis.spans if span.kind == "quoted_speech")
    start = quote.start + 20
    end = quote.end - 20

    sliced = slice_voice_analysis(analysis, source, start=start, end=end)

    assert "“" not in source[start:end]
    assert "”" not in source[start:end]
    assert sliced.passage_voice == "quoted_speech"
    assert sliced.quoted_speech_fraction == 1.0


def test_marks_unmatched_curly_quote_ranges_uncertain() -> None:
    """Do not credit malformed scrape text to the document author."""
    text = "Confirmed narration. “A quotation whose closing mark was lost"

    analysis = analyze_voice(text)

    assert analysis.passage_voice == "mixed"
    assert analysis.uncertain_fraction > 0.5
    assert any(span.kind == "uncertain" and span.text.startswith("“") for span in analysis.spans)


def test_pairs_swapped_curly_delimiters_conservatively() -> None:
    """Recognize a common scrape defect that uses two closing quote marks."""
    analysis = analyze_voice("She explains, ”Quoted speech survives the scrape.” Narration.")

    quote = next(span for span in analysis.spans if span.kind == "quoted_speech")

    assert quote.text == "”Quoted speech survives the scrape.”"
    assert analysis.uncertain_fraction == 0.0


def test_recognizes_markdown_blockquotes() -> None:
    """Treat Markdown blockquotes as quoted rather than author narration."""
    analysis = analyze_voice("Narration.\n\n> A quoted paragraph.\n> Still quoted.\n\nNarration resumes.")

    assert analysis.passage_voice == "mixed"
    assert any(span.kind == "quoted_speech" and span.text.startswith(">") for span in analysis.spans)
