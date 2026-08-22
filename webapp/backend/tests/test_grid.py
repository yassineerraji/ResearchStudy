"""Tests for the topology x severity grid endpoint (app.services.grid).

Inside `tests`, this module exercises `GET /api/v1/gallery/grid` against
extra synthetic experiment directories written directly into the fake
`outputs/` tree the `client` fixture already builds (see `conftest.py`) --
the central risks it guards against are the nine-cell shape staying stable
when a cell is unfilled, an `experiment_id` correctly resolving to its one
grid cell, calibration/smoke variants never filling a real cell, and ties
between two real runs of the same cell resolving to the newest one.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.core.paths import outputs_dir


def _write_minimal_experiment(
    directory: str, experiment_id: str, created_at: str, mean_delta: float
) -> None:
    experiment_dir = outputs_dir() / directory
    experiment_dir.mkdir(parents=True)
    manifest = {
        "experiment_id": experiment_id,
        "created_at_utc": created_at,
        "replications": 10,
        "llm_execution_mode": "LIVE",
        "llm_model": "fake-model",
        "base_seed": 1,
    }
    (experiment_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    summary = {"experiment_summary": {"mean_delta": mean_delta, "replication_count": 10}}
    (experiment_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_grid_has_nine_cells_labeled_blank_when_missing(client: TestClient) -> None:
    response = client.get("/api/v1/gallery/grid")
    assert response.status_code == 200
    body = response.json()
    assert len(body["cells"]) == 9
    assert body["topologies"] == ["Compact", "Standard", "Extended"]
    assert body["severities"] == ["Light", "Medium", "Heavy"]
    assert all(cell["directory"] is None for cell in body["cells"])


def test_grid_fills_a_cell_from_a_matching_experiment_id(client: TestClient) -> None:
    _write_minimal_experiment(
        "compact_medium_comparison__20260101T000000Z",
        "compact_medium_comparison",
        "20260101T000000Z",
        -5.0,
    )
    body = client.get("/api/v1/gallery/grid").json()
    cell = next(c for c in body["cells"] if c["topology"] == "Compact" and c["severity"] == "Medium")
    assert cell["directory"] == "compact_medium_comparison__20260101T000000Z"
    assert cell["experiment_summary"]["mean_delta"] == -5.0


def test_grid_ignores_calibration_variants(client: TestClient) -> None:
    _write_minimal_experiment(
        "compact_medium_comparison_calibration__20260101T000000Z",
        "compact_medium_comparison_calibration",
        "20260101T000000Z",
        -999.0,
    )
    body = client.get("/api/v1/gallery/grid").json()
    cell = next(c for c in body["cells"] if c["topology"] == "Compact" and c["severity"] == "Medium")
    assert cell["directory"] is None


def test_grid_prefers_the_most_recently_created_match(client: TestClient) -> None:
    _write_minimal_experiment(
        "compact_medium_comparison__20260101T000000Z",
        "compact_medium_comparison",
        "20260101T000000Z",
        -1.0,
    )
    _write_minimal_experiment(
        "compact_medium_comparison__20260201T000000Z",
        "compact_medium_comparison",
        "20260201T000000Z",
        -2.0,
    )
    body = client.get("/api/v1/gallery/grid").json()
    cell = next(c for c in body["cells"] if c["topology"] == "Compact" and c["severity"] == "Medium")
    assert cell["directory"] == "compact_medium_comparison__20260201T000000Z"
    assert cell["experiment_summary"]["mean_delta"] == -2.0
