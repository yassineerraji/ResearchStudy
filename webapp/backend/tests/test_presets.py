"""Tests for the sandbox preset picker (`/configs/presets`).

Inside `tests`, this module checks that all nine topology x severity presets
are listed, that a preset's content is the real, on-disk `configs/` file
content (not a hand-copied approximation), and that an unknown preset id is
a 404 -- the central risk this guards against is a preset id silently
resolving to the wrong network/scenario pair.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_list_presets_returns_all_nine_grid_cells(client: TestClient) -> None:
    response = client.get("/api/v1/configs/presets")
    assert response.status_code == 200
    presets = response.json()["presets"]
    assert len(presets) == 9
    cells = {(p["topology"], p["severity"]) for p in presets}
    assert cells == {
        (topology, severity)
        for topology in ("Compact", "Standard", "Extended")
        for severity in ("Light", "Medium", "Heavy")
    }


def test_extended_medium_preset_targets_hub_1_not_port_primary(client: TestClient) -> None:
    """Regression guard for the specific finding that motivated this preset:
    Extended's severity scenarios target `hub_1` (its real computed
    chokepoint), not `port_primary` like Compact/Standard do.
    """
    response = client.get("/api/v1/configs/presets/extended_medium")
    assert response.status_code == 200
    body = response.json()
    assert body["scenario"]["scenario_id"] == "hub_1_closure_extended"
    assert body["scenario"]["shocks"][0]["target_id"] == "hub_1"
    assert len(body["network"]["nodes"]) == 10


def test_compact_heavy_preset_uses_its_own_scenario_file(client: TestClient) -> None:
    """Compact has no alternate port, so it can't reuse Standard's Heavy
    scenario (which references an edge Compact doesn't have) -- V2.8.1.
    """
    response = client.get("/api/v1/configs/presets/compact_heavy")
    assert response.status_code == 200
    body = response.json()
    assert body["scenario"]["scenario_id"] == "primary_port_extended_closure_compact"
    assert len(body["network"]["edges"]) == 4


def test_presets_within_one_topology_share_one_base_seed(client: TestClient) -> None:
    seeds = {
        preset_id: client.get(f"/api/v1/configs/presets/{preset_id}").json()["base_seed"]
        for preset_id in ("compact_light", "compact_medium", "compact_heavy")
    }
    assert len(set(seeds.values())) == 1


def test_unknown_preset_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/configs/presets/does_not_exist")
    assert response.status_code == 404
