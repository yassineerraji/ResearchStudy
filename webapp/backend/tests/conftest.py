"""Shared pytest fixtures for the backend test suite.

Inside `tests`, this module builds an isolated fake repository root per test
(a real copy of `configs/` plus a small synthetic `outputs/` experiment
directory) so tests never depend on the developer's actual, gitignored
`outputs/` contents or mutate the real repository. It does not test any
behavior itself — that belongs to the individual test modules.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from supply_chain_simulator.data_io.writers import (
    DAILY_METRICS_COLUMNS,
    REPLICATIONS_COLUMNS,
    RUN_METRICS_COLUMNS,
)

REAL_REPO_ROOT = Path(__file__).resolve().parents[3]

FAKE_EXPERIMENT_DIR_NAME = "fake_experiment__20260101T000000Z"


def _write_csv(
    path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]
) -> None:
    """Writes rows keyed by column name, in `columns` order.

    Building rows as dicts (rather than positional lists) avoids silently
    miscounting fields when a schema like `RUN_METRICS_COLUMNS` has thirty-odd
    columns — a missing key raises immediately instead of shifting every
    later column by one.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(columns)
    for row in rows:
        writer.writerow([row[column] for column in columns])
    path.write_text(buffer.getvalue(), encoding="utf-8")


def _build_fake_outputs(outputs_dir: Path) -> None:
    experiment_dir = outputs_dir / FAKE_EXPERIMENT_DIR_NAME
    experiment_dir.mkdir(parents=True)

    manifest = {
        "experiment_id": "fake_experiment",
        "created_at_utc": "20260101T000000Z",
        "replications": 1,
        "llm_execution_mode": "LIVE",
        "llm_model": "fake-model",
        "base_seed": 1,
    }
    (experiment_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    summary = {
        "experiment_summary": {
            "replication_count": 1,
            "mean_delta": -1.0,
            "llm_win_rate": 1.0,
        }
    }
    (experiment_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    resolved_config = {
        "network": {"nodes": [{"node_id": "n1"}, {"node_id": "n2"}], "edges": []},
        "scenario": {"shocks": [{"shock_id": "s1"}]},
    }
    (experiment_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config), encoding="utf-8"
    )

    _write_csv(
        experiment_dir / "replications.csv",
        REPLICATIONS_COLUMNS,
        [
            {
                "replication": 1,
                "seed": 7,
                "heuristic_undisrupted_cost": "100.000000",
                "heuristic_disrupted_cost": "110.000000",
                "heuristic_tcd": "10.000000",
                "llm_undisrupted_cost": "100.000000",
                "llm_disrupted_cost": "105.000000",
                "llm_tcd": "5.000000",
                "delta": "-5.000000",
                "winner": "LLM",
            }
        ],
    )

    # Every numeric run/daily-metrics column defaults to a harmless zero;
    # only the fields the gallery grouping/indexing logic actually keys on
    # (replication/policy/run_kind/day) vary per row below.
    run_metrics_defaults: dict[str, object] = dict.fromkeys(RUN_METRICS_COLUMNS, "0.000000")
    run_metrics_defaults.update(
        ending_backlog_units=0,
        backlog_unit_days=0,
        late_delivered_units=0,
        reroute_count=0,
        expedite_count=0,
        expedited_units=0,
        decision_count=0,
        days_to_clear_backlog_after_shock="",
        terminated_with_unresolved_state=False,
    )
    daily_metrics_defaults: dict[str, object] = dict.fromkeys(DAILY_METRICS_COLUMNS, "0.000000")
    daily_metrics_defaults.update(
        inventory_units=0,
        backlog_units=0,
        shipments_at_node=0,
        shipments_in_transit=0,
        shipments_delivered=0,
        daily_demand_units=0,
        daily_same_day_fulfilled_units=0,
        daily_backlog_fulfilled_units=0,
        active_shock_ids="",
    )

    # Four (replication, policy, run_kind) branch groups, in the same
    # contiguous grouped order ExperimentWriter would produce.
    branches = [
        ("heuristic", "UNDISRUPTED"),
        ("heuristic", "DISRUPTED"),
        ("llm_agent", "UNDISRUPTED"),
        ("llm_agent", "DISRUPTED"),
    ]
    run_metrics_rows = []
    daily_rows = []
    decision_lines = []
    for policy, run_kind in branches:
        run_metrics_rows.append(
            {
                **run_metrics_defaults,
                "experiment_id": "fake_experiment",
                "scenario_id": "fake_scenario",
                "replication": 1,
                "seed": 7,
                "policy": policy,
                "run_kind": run_kind,
            }
        )
        for day in (21, 22):
            daily_rows.append(
                {
                    **daily_metrics_defaults,
                    "experiment_id": "fake_experiment",
                    "scenario_id": "fake_scenario",
                    "replication": 1,
                    "policy": policy,
                    "run_kind": run_kind,
                    "day": day,
                }
            )
        decision_lines.append(
            json.dumps(
                {
                    "replication": 1,
                    "policy": policy,
                    "run_kind": run_kind,
                    "day": 21,
                    "shipment_id": f"shipment_{policy}_{run_kind}",
                }
            )
        )

    _write_csv(experiment_dir / "run_metrics.csv", RUN_METRICS_COLUMNS, run_metrics_rows)
    _write_csv(experiment_dir / "daily_metrics.csv", DAILY_METRICS_COLUMNS, daily_rows)

    (experiment_dir / "decision_traces.jsonl").write_text(
        "\n".join(decision_lines) + "\n", encoding="utf-8"
    )
    (experiment_dir / "event_tapes.jsonl").write_text(
        json.dumps({"replication": 1, "shocks": []}) + "\n", encoding="utf-8"
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    fake_root = tmp_path / "repo"
    shutil.copytree(REAL_REPO_ROOT / "configs", fake_root / "configs")
    _build_fake_outputs(fake_root / "outputs")

    monkeypatch.setenv("SCAE_REPO_ROOT", str(fake_root))

    from app.core.paths import repo_root

    repo_root.cache_clear()
    try:
        from app.config import get_settings

        get_settings.cache_clear()
        from app.main import create_app

        with TestClient(create_app()) as test_client:
            yield test_client
    finally:
        repo_root.cache_clear()
