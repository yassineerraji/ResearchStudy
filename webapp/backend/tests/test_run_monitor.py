"""Tests for parsing the CLI's stdout progress lines.

Inside `tests`, this module checks `app.services.run_monitor`'s two
regexes against the exact strings `cli.py` actually prints
(`_print_progress` and `_run_experiment`'s "Experiment ID:" line), plus
lines that should NOT match (arbitrary log noise, partial matches).
"""

from __future__ import annotations

from app.services.run_monitor import (
    ReplicationProgress,
    parse_experiment_id,
    parse_replication_progress,
)


def test_parses_replication_progress_line() -> None:
    line = "Replication 3/100: delta=-5240.000000 winner=LLM"
    assert parse_replication_progress(line) == ReplicationProgress(completed=3, total=100)


def test_parses_replication_progress_with_surrounding_whitespace() -> None:
    assert parse_replication_progress("  Replication 1/1: delta=0.000000 winner=TIE  \n") == (
        ReplicationProgress(completed=1, total=1)
    )


def test_non_progress_lines_return_none() -> None:
    assert parse_replication_progress("Experiment ID: baseline_port_closure_comparison") is None
    assert parse_replication_progress("Output written to: /some/path") is None
    assert parse_replication_progress("") is None
    assert parse_replication_progress("Replication abc/def: delta=") is None


def test_parses_experiment_id_line() -> None:
    assert parse_experiment_id("Experiment ID: baseline_port_closure_comparison") == (
        "baseline_port_closure_comparison"
    )


def test_non_experiment_id_lines_return_none() -> None:
    assert parse_experiment_id("Replication 1/5: delta=0.0 winner=TIE") is None
    assert parse_experiment_id("") is None
