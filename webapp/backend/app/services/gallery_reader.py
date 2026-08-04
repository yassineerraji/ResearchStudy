"""Reads completed experiment output directories for the Results Gallery.

Inside `app.services`, this module is the read-only counterpart to
`data_io/writers.py`: it knows the exact on-disk shape of
`manifest.json`/`summary.json`/`replications.csv`/`run_metrics.csv` (small,
loaded whole) and, via `app.services.gallery_index`, how to pull a single
(replication, policy, run_kind) slice out of the much larger
`daily_metrics.csv`/`decision_traces.jsonl` without parsing the whole file.
In the full backend, it is what turns `outputs/<experiment>__<ts>/` on disk
into the JSON the gallery API serves. It does not run simulations or write
anything back to `outputs/`.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from app.core.paths import outputs_dir
from app.services import gallery_index

_INT_DAILY_METRICS_FIELDS = {
    "replication",
    "day",
    "inventory_units",
    "backlog_units",
    "shipments_at_node",
    "shipments_in_transit",
    "shipments_delivered",
    "daily_demand_units",
    "daily_same_day_fulfilled_units",
    "daily_backlog_fulfilled_units",
}
_FLOAT_DAILY_METRICS_FIELDS = {
    "daily_transport_cost",
    "daily_reroute_cost",
    "daily_expedite_cost",
    "daily_holding_cost",
    "daily_backlog_cost",
    "daily_late_cost",
    "cumulative_total_cost",
}


class ExperimentNotFoundError(Exception):
    """Raised when a gallery experiment directory name is unknown or unsafe."""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"expected a JSON object at the top level of {path}")
    return data


def _list_experiment_dirs() -> list[Path]:
    root = outputs_dir()
    if not root.exists():
        return []
    return [
        entry
        for entry in root.iterdir()
        if entry.is_dir()
        and not entry.name.startswith((".", "_"))
        and (entry / "manifest.json").is_file()
    ]


def resolve_experiment_dir(directory: str) -> Path:
    if not directory or "/" in directory or "\\" in directory or directory in {".", ".."}:
        raise ExperimentNotFoundError(directory)
    root = outputs_dir().resolve()
    candidate = (root / directory).resolve()
    if not candidate.is_relative_to(root) or not (candidate / "manifest.json").is_file():
        raise ExperimentNotFoundError(directory)
    return candidate


def list_experiments() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for directory in _list_experiment_dirs():
        manifest = _read_json(directory / "manifest.json")
        summary_path = directory / "summary.json"
        summary = _read_json(summary_path) if summary_path.is_file() else {}
        items.append(
            {
                "directory": directory.name,
                "manifest": manifest,
                "experiment_summary": summary.get("experiment_summary"),
            }
        )
    items.sort(key=lambda item: item["manifest"].get("created_at_utc", ""), reverse=True)
    return items


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def get_experiment_detail(directory: str) -> dict[str, Any]:
    experiment_dir = resolve_experiment_dir(directory)
    manifest = _read_json(experiment_dir / "manifest.json")
    summary_path = experiment_dir / "summary.json"
    summary = _read_json(summary_path) if summary_path.is_file() else None

    resolved_config_path = experiment_dir / "resolved_config.yaml"
    network: dict[str, Any] = {}
    scenario: dict[str, Any] = {}
    if resolved_config_path.is_file():
        with resolved_config_path.open("r", encoding="utf-8") as handle:
            resolved = yaml.safe_load(handle) or {}
        network = resolved.get("network", {})
        scenario = resolved.get("scenario", {})

    return {
        "directory": experiment_dir.name,
        "manifest": manifest,
        "summary": summary,
        "replications": _read_csv_rows(experiment_dir / "replications.csv"),
        "run_metrics": _read_csv_rows(experiment_dir / "run_metrics.csv"),
        "network": network,
        "scenario": scenario,
    }


def _normalize_daily_metrics_row(row: dict[str, str]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in row.items():
        if key in _INT_DAILY_METRICS_FIELDS:
            normalized[key] = int(value)
        elif key in _FLOAT_DAILY_METRICS_FIELDS:
            normalized[key] = float(value)
        elif key == "active_shock_ids":
            normalized[key] = [s for s in value.split(";") if s]
        else:
            normalized[key] = value
    return normalized


def _read_daily_metrics_slice(
    experiment_dir: Path, byte_range: gallery_index.ByteRange
) -> list[dict[str, Any]]:
    from supply_chain_simulator.data_io.writers import DAILY_METRICS_COLUMNS

    raw = gallery_index.read_range(experiment_dir / "daily_metrics.csv", byte_range)
    text = raw.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text), fieldnames=list(DAILY_METRICS_COLUMNS))
    return [_normalize_daily_metrics_row(row) for row in reader]


def _read_jsonl_slice(path: Path, byte_range: gallery_index.ByteRange) -> list[dict[str, Any]]:
    raw = gallery_index.read_range(path, byte_range)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def get_replay_slice(
    directory: str, replication: int, policy: str, run_kind: str
) -> dict[str, Any]:
    experiment_dir = resolve_experiment_dir(directory)
    index = gallery_index.get_index(experiment_dir)
    key = (replication, policy, run_kind)

    daily_range = index.daily_metrics.get(key)
    daily_metrics = (
        _read_daily_metrics_slice(experiment_dir, daily_range) if daily_range else []
    )

    decision_range = index.decision_traces.get(key)
    decisions = (
        _read_jsonl_slice(experiment_dir / "decision_traces.jsonl", decision_range)
        if decision_range
        else []
    )

    return {
        "directory": experiment_dir.name,
        "replication": replication,
        "policy": policy,
        "run_kind": run_kind,
        "daily_metrics": daily_metrics,
        "decisions": decisions,
    }
