"""Parses `supply_chain_simulator.cli`'s stdout lines into progress updates.

Inside `app.services`, this module is the seam between a subprocess's raw
text output and structured progress `app.services.run_registry` can store.
It reuses the CLI's own printed contract — `_print_progress` in `cli.py`
prints one `Replication N/Total: delta=... winner=...` line per replication,
and `_run_experiment` prints one `Experiment ID: ...` line up front — rather
than reaching into `runner.py`'s in-process `on_replication_complete`
callback, so `run_launcher` stays a pure subprocess wrapper (see the M3
design notes on why that's the more change-resistant boundary). Pure string
parsing, no I/O, so it's cheap to test without a subprocess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_REPLICATION_LINE = re.compile(r"^Replication (\d+)/(\d+): delta=")
_EXPERIMENT_ID_LINE = re.compile(r"^Experiment ID: (.+)$")


@dataclass(frozen=True)
class ReplicationProgress:
    completed: int
    total: int


def parse_replication_progress(line: str) -> ReplicationProgress | None:
    match = _REPLICATION_LINE.match(line.strip())
    if not match:
        return None
    return ReplicationProgress(completed=int(match.group(1)), total=int(match.group(2)))


def parse_experiment_id(line: str) -> str | None:
    match = _EXPERIMENT_ID_LINE.match(line.strip())
    if not match:
        return None
    return match.group(1).strip()
