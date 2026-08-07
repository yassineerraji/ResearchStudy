"""In-memory bookkeeping for sandbox runs: status, progress, and results.

Inside `app.services`, this module is the single source of truth for "what
is this run_id doing right now" — `run_launcher` writes to it as a run
progresses, `routers.runs` reads from it to answer status/detail/replay
requests. It deliberately knows nothing about subprocesses, config files, or
HTTP — just run identity, state, and progress counters — so M5 can swap the
in-memory `dict` here for SQLite-backed storage without any other module
changing. It does not decide when a run transitions state; callers do.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset(
    {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
)


class RunNotFoundError(Exception):
    """Raised when a run_id has no matching registry entry."""


class RunNotReadyError(Exception):
    """Raised when detail/replay is requested before a run has completed."""


@dataclass
class RunRecord:
    run_id: str
    status: RunStatus
    created_at: datetime
    total_replications: int
    experiment_id: str | None = None
    completed_replications: int = 0
    error: str | None = None
    result_directory: Path | None = None


class RunRegistry:
    def __init__(self) -> None:
        self._records: dict[str, RunRecord] = {}
        self._lock = asyncio.Lock()

    async def create(self, run_id: str, total_replications: int) -> RunRecord:
        record = RunRecord(
            run_id=run_id,
            status=RunStatus.QUEUED,
            created_at=datetime.now(UTC),
            total_replications=total_replications,
        )
        async with self._lock:
            self._records[run_id] = record
        return record

    async def get(self, run_id: str) -> RunRecord:
        async with self._lock:
            record = self._records.get(run_id)
        if record is None:
            raise RunNotFoundError(run_id)
        return record

    async def list_all(self) -> list[RunRecord]:
        async with self._lock:
            return sorted(self._records.values(), key=lambda r: r.created_at, reverse=True)

    async def update(self, run_id: str, **fields: object) -> RunRecord:
        async with self._lock:
            record = self._records.get(run_id)
            if record is None:
                raise RunNotFoundError(run_id)
            for key, value in fields.items():
                setattr(record, key, value)
            return record


_registry: RunRegistry | None = None


def get_registry() -> RunRegistry:
    """Process-wide singleton — see M0's `get_settings` for the same pattern.

    A real dependency-injected instance (rather than a module-level global)
    would be preferable, but FastAPI route handlers and the background
    `run_launcher` task both need to reach the same registry without being
    wired through request state, and this is the smallest thing that does
    that correctly for a single-process deployment.
    """
    global _registry
    if _registry is None:
        _registry = RunRegistry()
    return _registry


def reset_registry_for_tests() -> None:
    global _registry
    _registry = RunRegistry()
