"""Reusable wall-clock timing for notebook and command workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True, slots=True)
class TimingRecord:
    """Elapsed wall-clock time for one named operation."""

    label: str
    elapsed_seconds: float

    def __str__(self) -> str:
        """Return a compact human-readable timing line."""
        return f"{self.label}: {self.elapsed_seconds:.3f}s"


class TimingLog:
    """Collect named wall-clock timings in execution order."""

    def __init__(self, *, clock: Callable[[], float] = perf_counter) -> None:
        """Initialize an empty timing log with an injectable monotonic clock."""
        self._clock = clock
        self._records: list[TimingRecord] = []

    @property
    def records(self) -> tuple[TimingRecord, ...]:
        """Return immutable timing records in execution order."""
        return tuple(self._records)

    def start(self) -> float:
        """Return a monotonic start token for a later ``finish`` call."""
        return self._clock()

    def finish(self, label: str, started_at: float) -> TimingRecord:
        """Record and return elapsed time since a start token."""
        record = TimingRecord(
            label=label,
            elapsed_seconds=self._clock() - started_at,
        )
        self._records.append(record)
        return record

    def format_report(self) -> str:
        """Render all collected timings and their total."""
        lines = ["Timing report"]
        lines.extend(f"- {record}" for record in self._records)
        total = sum(record.elapsed_seconds for record in self._records)
        lines.append(f"- Recorded total: {total:.3f}s")
        return "\n".join(lines)
