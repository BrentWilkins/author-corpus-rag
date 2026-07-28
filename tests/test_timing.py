"""Tests for reusable notebook timing records."""

from author_corpus.timing import TimingLog


def test_timing_log_records_and_formats_elapsed_steps() -> None:
    """Collect deterministic step timings and a report total."""
    values = iter((10.0, 11.25, 20.0, 22.5))
    timings = TimingLog(clock=lambda: next(values))

    first = timings.start()
    first_record = timings.finish("First step", first)
    second = timings.start()
    second_record = timings.finish("Second step", second)

    assert first_record.elapsed_seconds == 1.25
    assert second_record.elapsed_seconds == 2.5
    assert timings.records == (first_record, second_record)
    assert timings.format_report() == ("Timing report\n- First step: 1.250s\n- Second step: 2.500s\n- Recorded total: 3.750s")
