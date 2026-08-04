"""Tests for the `/configs` schema, defaults, and validate endpoints.

Inside `tests`, this module checks that `app.services.config_schema` serves
real, current schemas/defaults from `supply_chain_simulator`'s config models
and that `app.services.config_validate` genuinely resolves a submitted
bundle through the research package's own validator rather than a
reimplementation — accepting the real baseline configs and rejecting both a
single-field and a cross-config violation with a 422.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

CONFIG_TYPES = ["network", "scenario", "heuristic_policy", "llm_policy", "experiment"]

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(relative: str) -> dict:
    with (REPO_ROOT / "configs" / relative).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_raw_configs() -> dict[str, dict]:
    """Exact on-disk content — what GET /configs/defaults/{type} should echo."""
    return {
        "network": _load("networks/baseline_network.yaml"),
        "scenario": _load("scenarios/port_closure.yaml"),
        "heuristic_policy": _load("policies/heuristic.yaml"),
        "llm_policy": _load("policies/llm_agent.yaml"),
        "experiment": _load("experiments/baseline_comparison.yaml"),
    }


def _load_validate_bundle() -> dict[str, dict]:
    """What a frontend would submit to POST /configs/validate.

    `experiment`'s path-referencing fields are omitted because
    `config_validate.validate_config_bundle` overwrites them to point at the
    sandboxed files it writes — a real submission never needs to supply them.
    """
    bundle = _load_raw_configs()
    experiment = dict(bundle["experiment"])
    for key in ("network_config", "scenario_config", "policy_configs", "output_root"):
        experiment.pop(key, None)
    bundle["experiment"] = experiment
    return bundle


@pytest.mark.parametrize("config_type", CONFIG_TYPES)
def test_schema_endpoint_returns_json_schema(client: TestClient, config_type: str) -> None:
    response = client.get(f"/api/v1/configs/schema/{config_type}")
    assert response.status_code == 200
    body = response.json()
    assert body["config_type"] == config_type
    assert "properties" in body["json_schema"]
    assert "note" in body


def test_schema_endpoint_rejects_unknown_type(client: TestClient) -> None:
    response = client.get("/api/v1/configs/schema/not_a_real_type")
    assert response.status_code == 422


@pytest.mark.parametrize("config_type", CONFIG_TYPES)
def test_defaults_endpoint_matches_real_file(client: TestClient, config_type: str) -> None:
    response = client.get(f"/api/v1/configs/defaults/{config_type}")
    assert response.status_code == 200
    expected = _load_raw_configs()[config_type]
    assert response.json()["content"] == expected


def test_validate_accepts_real_baseline_bundle(client: TestClient) -> None:
    bundle = _load_validate_bundle()
    response = client.post("/api/v1/configs/validate", json=bundle)
    assert response.status_code == 200
    assert response.json() == {
        "valid": True,
        "experiment_id": "baseline_port_closure_comparison",
    }


def test_validate_rejects_warmup_not_before_horizon(client: TestClient) -> None:
    bundle = _load_validate_bundle()
    bundle["experiment"] = {**bundle["experiment"], "warmup_days": bundle["experiment"]["horizon_days"]}
    response = client.post("/api/v1/configs/validate", json=bundle)
    assert response.status_code == 422
    assert "warmup_days" in response.json()["detail"]


def test_validate_rejects_cross_config_reference_violation(client: TestClient) -> None:
    """A shock targeting a node that doesn't exist — only catchable by
    resolving the full bundle together, not by any single file's own schema.
    """
    bundle = _load_validate_bundle()
    scenario = {**bundle["scenario"]}
    scenario["shocks"] = [{**scenario["shocks"][0], "target_id": "no_such_node"}]
    bundle["scenario"] = scenario
    response = client.post("/api/v1/configs/validate", json=bundle)
    assert response.status_code == 422
    assert "no_such_node" in response.json()["detail"]


def test_validate_rejects_malformed_single_field(client: TestClient) -> None:
    bundle = _load_validate_bundle()
    network = {**bundle["network"]}
    edges = list(network["edges"])
    edges[0] = {**edges[0], "reliability": 1.5}  # out of [0, 1]
    network["edges"] = edges
    bundle["network"] = network
    response = client.post("/api/v1/configs/validate", json=bundle)
    assert response.status_code == 422
