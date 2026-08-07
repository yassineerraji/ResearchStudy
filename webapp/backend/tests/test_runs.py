"""End-to-end tests for the `/runs` sandbox-execution API.

Inside `tests`, this module drives `POST /api/v1/runs` through to a
terminal status the same way a real caller would: submit, poll
`GET /runs/{id}` until it stops being queued/running, then read
`detail`/`replay`. The subprocess `run_launcher` spawns is
`tests/fixtures/fake_cli.py`, not the real CLI — it mimics the real CLI's
stdout/exit-code contract without making any OpenAI call, so these tests are
fast, free, and deterministic while still exercising the real
`asyncio.create_subprocess_exec` orchestration path.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from app.services.run_launcher import RunLauncher, set_launcher_for_tests
from app.services.run_registry import get_registry

FAKE_CLI_PATH = Path(__file__).resolve().parent / "fixtures" / "fake_cli.py"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _configure_launcher(mode: str, timeout_seconds: int = 30) -> None:
    def command_builder(experiment_path: Path) -> list[str]:
        return [sys.executable, str(FAKE_CLI_PATH), mode, str(experiment_path)]

    set_launcher_for_tests(
        RunLauncher(
            registry=get_registry(),
            max_concurrent_runs=2,
            run_timeout_seconds=timeout_seconds,
            command_builder=command_builder,
        )
    )


def _load_valid_bundle() -> dict[str, dict]:
    def load(relative: str) -> dict:
        with (REPO_ROOT / "configs" / relative).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    experiment = load("experiments/baseline_comparison.yaml")
    for key in ("network_config", "scenario_config", "policy_configs", "output_root"):
        experiment.pop(key, None)
    experiment["replications"] = 2

    return {
        "network": load("networks/baseline_network.yaml"),
        "scenario": load("scenarios/port_closure.yaml"),
        "heuristic_policy": load("policies/heuristic.yaml"),
        "llm_policy": load("policies/llm_agent.yaml"),
        "experiment": experiment,
        "api_key": "sk-fake-test-key",
    }


def _poll_until_terminal(client: TestClient, run_id: str, timeout_seconds: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        body = client.get(f"/api/v1/runs/{run_id}").json()
        if body["status"] in {"completed", "failed", "cancelled"}:
            return body
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not reach a terminal status in time")


def test_limits_endpoint_reflects_settings(client: TestClient) -> None:
    response = client.get("/api/v1/runs/limits")
    assert response.status_code == 200
    body = response.json()
    assert body["max_sandbox_replications"] >= 1
    assert body["max_concurrent_runs"] >= 1


def test_submit_run_completes_and_is_readable(client: TestClient) -> None:
    _configure_launcher(mode="success")
    bundle = _load_valid_bundle()

    submitted = client.post("/api/v1/runs", json=bundle)
    assert submitted.status_code == 201
    run_id = submitted.json()["run_id"]
    assert submitted.json()["status"] in {"queued", "running"}

    final = _poll_until_terminal(client, run_id)
    assert final["status"] == "completed"
    assert final["total_replications"] == 2
    assert final["completed_replications"] == 2
    assert final["experiment_id"] == "baseline_port_closure_comparison"

    detail = client.get(f"/api/v1/runs/{run_id}/detail")
    assert detail.status_code == 200
    assert detail.json()["manifest"]["experiment_id"] == "baseline_port_closure_comparison"


def test_run_not_ready_returns_409_before_completion(client: TestClient) -> None:
    _configure_launcher(mode="hang", timeout_seconds=30)
    bundle = _load_valid_bundle()
    run_id = client.post("/api/v1/runs", json=bundle).json()["run_id"]

    response = client.get(f"/api/v1/runs/{run_id}/detail")
    assert response.status_code == 409


def test_replications_above_cap_are_rejected(client: TestClient) -> None:
    _configure_launcher(mode="success")
    bundle = _load_valid_bundle()
    bundle["experiment"]["replications"] = 999

    response = client.post("/api/v1/runs", json=bundle)
    assert response.status_code == 422
    assert "replications" in response.json()["detail"]


def test_missing_deployment_model_env_var_is_rejected(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Server-side check added after a real smoke test caught this: a
    deployment missing its own LLM_MODEL env var would otherwise fail every
    single run identically, deep inside the subprocess, after already
    occupying a concurrency slot — see run_launcher._prepare_sandbox.
    """
    monkeypatch.delenv("LLM_MODEL", raising=False)  # the client fixture sets this
    _configure_launcher(mode="success")
    bundle = _load_valid_bundle()

    response = client.post("/api/v1/runs", json=bundle)
    assert response.status_code == 422
    assert "LLM_MODEL" in response.json()["detail"]


def test_replay_mode_is_rejected_for_sandbox_runs(client: TestClient) -> None:
    _configure_launcher(mode="success")
    bundle = _load_valid_bundle()
    bundle["llm_policy"] = {**bundle["llm_policy"], "execution_mode": "REPLAY"}

    response = client.post("/api/v1/runs", json=bundle)
    assert response.status_code == 422
    assert "LIVE" in response.json()["detail"]


def test_run_failure_is_reported(client: TestClient) -> None:
    _configure_launcher(mode="fail")
    bundle = _load_valid_bundle()
    run_id = client.post("/api/v1/runs", json=bundle).json()["run_id"]

    final = _poll_until_terminal(client, run_id)
    assert final["status"] == "failed"
    assert "exit" in final["error"].lower() or "3" in final["error"]


def test_cancel_run_marks_cancelled(client: TestClient) -> None:
    _configure_launcher(mode="hang", timeout_seconds=30)
    bundle = _load_valid_bundle()
    run_id = client.post("/api/v1/runs", json=bundle).json()["run_id"]

    # Give the background task a moment to actually spawn the subprocess
    # before cancelling, so this exercises the "kill a live process" path.
    time.sleep(0.2)
    cancel_response = client.delete(f"/api/v1/runs/{run_id}")
    assert cancel_response.status_code == 204

    final = _poll_until_terminal(client, run_id)
    assert final["status"] == "cancelled"


def test_run_timeout_marks_failed(client: TestClient) -> None:
    _configure_launcher(mode="hang", timeout_seconds=1)
    bundle = _load_valid_bundle()
    run_id = client.post("/api/v1/runs", json=bundle).json()["run_id"]

    final = _poll_until_terminal(client, run_id, timeout_seconds=10)
    assert final["status"] == "failed"
    assert "timeout" in final["error"].lower()


def test_unknown_run_id_returns_404(client: TestClient) -> None:
    assert client.get("/api/v1/runs/does-not-exist").status_code == 404
    assert client.delete("/api/v1/runs/does-not-exist").status_code == 404


def test_api_key_is_never_echoed_back(client: TestClient) -> None:
    _configure_launcher(mode="success")
    bundle = _load_valid_bundle()
    run_id = client.post("/api/v1/runs", json=bundle).json()["run_id"]

    status_body = client.get(f"/api/v1/runs/{run_id}").text
    assert "sk-fake-test-key" not in status_body

    _poll_until_terminal(client, run_id)
    list_body = client.get("/api/v1/runs").text
    assert "sk-fake-test-key" not in list_body
