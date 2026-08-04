"""Tests for the read-only Results Gallery endpoints and byte-offset indexing.

Inside `tests`, this module exercises `app.services.gallery_reader` and
`app.services.gallery_index` against the synthetic experiment directory
`conftest.py` builds (four (replication, policy, run_kind) branch groups).
The central risk this guards against: a slice request for one branch
accidentally returning another branch's rows, which byte-offset indexing
would get wrong if the grouping logic were off by one line or one field.
"""

from __future__ import annotations

import csv

from fastapi.testclient import TestClient

from tests.conftest import FAKE_EXPERIMENT_DIR_NAME


def test_list_experiments_returns_the_fake_experiment(client: TestClient) -> None:
    response = client.get("/api/v1/gallery")
    assert response.status_code == 200
    body = response.json()
    assert len(body["experiments"]) == 1
    item = body["experiments"][0]
    assert item["directory"] == FAKE_EXPERIMENT_DIR_NAME
    assert item["manifest"]["experiment_id"] == "fake_experiment"
    assert item["experiment_summary"]["mean_delta"] == -1.0


def test_unknown_experiment_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/gallery/does_not_exist")
    assert response.status_code == 404


def test_experiment_detail(client: TestClient) -> None:
    response = client.get(f"/api/v1/gallery/{FAKE_EXPERIMENT_DIR_NAME}")
    assert response.status_code == 200
    body = response.json()
    assert body["manifest"]["experiment_id"] == "fake_experiment"
    assert len(body["replications"]) == 1
    assert len(body["run_metrics"]) == 4
    assert len(body["network"]["nodes"]) == 2
    assert len(body["scenario"]["shocks"]) == 1


def test_replay_slice_returns_only_its_own_branch(client: TestClient) -> None:
    response = client.get(
        f"/api/v1/gallery/{FAKE_EXPERIMENT_DIR_NAME}/replay",
        params={"replication": 1, "policy": "heuristic", "run_kind": "UNDISRUPTED"},
    )
    assert response.status_code == 200
    body = response.json()

    assert len(body["daily_metrics"]) == 2
    assert [row["day"] for row in body["daily_metrics"]] == [21, 22]
    assert all(row["policy"] == "heuristic" for row in body["daily_metrics"])
    assert all(row["run_kind"] == "UNDISRUPTED" for row in body["daily_metrics"])

    assert len(body["decisions"]) == 1
    assert body["decisions"][0]["shipment_id"] == "shipment_heuristic_UNDISRUPTED"


def test_replay_slice_for_a_different_branch_does_not_leak_rows(client: TestClient) -> None:
    disrupted = client.get(
        f"/api/v1/gallery/{FAKE_EXPERIMENT_DIR_NAME}/replay",
        params={"replication": 1, "policy": "heuristic", "run_kind": "DISRUPTED"},
    ).json()
    llm_undisrupted = client.get(
        f"/api/v1/gallery/{FAKE_EXPERIMENT_DIR_NAME}/replay",
        params={"replication": 1, "policy": "llm_agent", "run_kind": "UNDISRUPTED"},
    ).json()

    assert all(row["run_kind"] == "DISRUPTED" for row in disrupted["daily_metrics"])
    assert all(row["policy"] == "llm_agent" for row in llm_undisrupted["daily_metrics"])
    assert disrupted["decisions"][0]["shipment_id"] == "shipment_heuristic_DISRUPTED"
    assert llm_undisrupted["decisions"][0]["shipment_id"] == "shipment_llm_agent_UNDISRUPTED"


def test_replay_slice_for_unknown_group_is_empty_not_error(client: TestClient) -> None:
    response = client.get(
        f"/api/v1/gallery/{FAKE_EXPERIMENT_DIR_NAME}/replay",
        params={"replication": 999, "policy": "heuristic", "run_kind": "UNDISRUPTED"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["daily_metrics"] == []
    assert body["decisions"] == []


def test_indexed_slice_matches_full_file_parse(client: TestClient) -> None:
    """Guards the exact risk flagged during design review: that seeking into
    a byte range returns identical data to fully parsing the file and
    filtering in Python.
    """
    # Resolve the fake experiment directory the fixture actually wrote to,
    # via the same client/app the assertions above used.
    detail = client.get(f"/api/v1/gallery/{FAKE_EXPERIMENT_DIR_NAME}").json()
    assert detail["manifest"]["experiment_id"] == "fake_experiment"

    from app.core.paths import outputs_dir

    daily_metrics_path = outputs_dir() / FAKE_EXPERIMENT_DIR_NAME / "daily_metrics.csv"
    with daily_metrics_path.open("r", encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    expected = [
        row
        for row in all_rows
        if row["replication"] == "1" and row["policy"] == "llm_agent" and row["run_kind"] == "DISRUPTED"
    ]

    response = client.get(
        f"/api/v1/gallery/{FAKE_EXPERIMENT_DIR_NAME}/replay",
        params={"replication": 1, "policy": "llm_agent", "run_kind": "DISRUPTED"},
    )
    actual = response.json()["daily_metrics"]

    assert len(actual) == len(expected) == 2
    assert [row["day"] for row in actual] == [int(row["day"]) for row in expected]
