"""Builds and caches byte-offset indexes over large per-experiment output files.

Inside `app.services`, this module exists because `decision_traces.jsonl` and
`daily_metrics.csv` can run into tens of megabytes and tens of thousands of
rows for a 100-replication experiment (`ExperimentWriter` in
`data_io/writers.py` appends them per branch, in replication order). A
day-by-day replay only ever needs one (replication, policy, run_kind) slice
out of that. This module does one sequential scan per file, per experiment
directory, recording the contiguous byte range each group occupies (contiguous
because the writer already appends in that grouped order), caches the result
in memory, and lets `app.services.gallery_reader` fetch a slice with a single
`seek()` instead of re-parsing the whole file on every request. It does not
interpret row/record contents beyond the grouping keys it needs.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from supply_chain_simulator.data_io.writers import DAILY_METRICS_COLUMNS

GroupKey = tuple[int, str, str]
LlmInteractionKey = tuple[int, str]

_SENTINEL = object()


@dataclass(frozen=True)
class ByteRange:
    start: int
    end: int


@dataclass(frozen=True)
class ExperimentIndex:
    daily_metrics: dict[GroupKey, ByteRange]
    decision_traces: dict[GroupKey, ByteRange]
    event_tapes: dict[int, ByteRange]
    llm_interactions: dict[LlmInteractionKey, ByteRange]


def _index_jsonl(path: Path, key_fn: Callable[[dict[str, Any]], Any]) -> dict[Any, ByteRange]:
    index: dict[Any, ByteRange] = {}
    if not path.is_file():
        return index
    current_key: Any = _SENTINEL
    current_start = 0
    with path.open("rb") as handle:
        offset = handle.tell()
        for raw_line in handle:
            if raw_line.strip():
                record = json.loads(raw_line)
                key = key_fn(record)
                if key != current_key:
                    if current_key is not _SENTINEL:
                        index[current_key] = ByteRange(current_start, offset)
                    current_key = key
                    current_start = offset
            offset = handle.tell()
        if current_key is not _SENTINEL:
            index[current_key] = ByteRange(current_start, offset)
    return index


def _index_daily_metrics(path: Path) -> dict[GroupKey, ByteRange]:
    index: dict[GroupKey, ByteRange] = {}
    if not path.is_file():
        return index
    replication_idx = DAILY_METRICS_COLUMNS.index("replication")
    policy_idx = DAILY_METRICS_COLUMNS.index("policy")
    run_kind_idx = DAILY_METRICS_COLUMNS.index("run_kind")

    current_key: GroupKey | None = None
    current_start = 0
    with path.open("rb") as handle:
        handle.readline()  # header row, not part of any group's byte range
        offset = handle.tell()
        for raw_line in handle:
            if raw_line.strip():
                fields = raw_line.decode("utf-8").rstrip("\r\n").split(",")
                key = (int(fields[replication_idx]), fields[policy_idx], fields[run_kind_idx])
                if key != current_key:
                    if current_key is not None:
                        index[current_key] = ByteRange(current_start, offset)
                    current_key = key
                    current_start = offset
            offset = handle.tell()
        if current_key is not None:
            index[current_key] = ByteRange(current_start, offset)
    return index


def _build_index(experiment_dir: Path) -> ExperimentIndex:
    return ExperimentIndex(
        daily_metrics=_index_daily_metrics(experiment_dir / "daily_metrics.csv"),
        decision_traces=_index_jsonl(
            experiment_dir / "decision_traces.jsonl",
            key_fn=lambda r: (int(r["replication"]), str(r["policy"]), str(r["run_kind"])),
        ),
        event_tapes=_index_jsonl(
            experiment_dir / "event_tapes.jsonl",
            key_fn=lambda r: int(r["replication"]),
        ),
        # llm_interactions.jsonl only ever contains llm_agent entries (the
        # heuristic never calls out), so its records carry no top-level
        # `policy` field -- replication/run_kind live one level down, inside
        # each entry's own `decision_key` (unlike decision_traces.jsonl,
        # which duplicates those fields at the top level).
        llm_interactions=_index_jsonl(
            experiment_dir / "llm_interactions.jsonl",
            key_fn=lambda r: (
                int(r["decision_key"]["replication"]),
                str(r["decision_key"]["run_kind"]),
            ),
        ),
    )


_cache: dict[Path, ExperimentIndex] = {}
_cache_lock = Lock()


def get_index(experiment_dir: Path) -> ExperimentIndex:
    with _cache_lock:
        cached = _cache.get(experiment_dir)
    if cached is not None:
        return cached
    built = _build_index(experiment_dir)
    with _cache_lock:
        _cache.setdefault(experiment_dir, built)
        return _cache[experiment_dir]


def read_range(path: Path, byte_range: ByteRange) -> bytes:
    with path.open("rb") as handle:
        handle.seek(byte_range.start)
        return handle.read(byte_range.end - byte_range.start)
