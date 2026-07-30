"""Unit tests for data_io/loaders.py: config parsing and domain conversion.

Inside tests/unit, this file checks that the five real configuration files
under configs/ and the tiny test fixtures parse into the expected typed
objects, that every malformed configuration this module is meant to reject
actually fails loudly (with ConfigurationError from the file-loading path, or
pydantic.ValidationError from direct model construction) rather than being
silently accepted, and that build_network_definition and build_initial_state
produce the exact domain objects and day-0 state CLAUDE.md section 13.1
describes. It does not test simulation behavior, since no day ever advances
in this file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from supply_chain_simulator.data_io import loaders
from supply_chain_simulator.data_io.loaders import (
    ConfigurationError,
    ExperimentConfig,
    LLMPolicyConfig,
    NetworkConfig,
    ResolvedConfig,
    ScenarioConfig,
    ShockConfig,
    build_initial_state,
    build_network_definition,
    load_network_config,
    resolve_config,
)
from supply_chain_simulator.domain.models import NodeType, TransportMode

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_EXPERIMENT_CONFIG = REPO_ROOT / "configs/experiments/baseline_comparison.yaml"
TINY_NETWORK_CONFIG = REPO_ROOT / "tests/fixtures/tiny_network.yaml"

_MINIMAL_NETWORK_YAML = """
schema_version: 1
network_id: minimal
units: {{quantity: units, distance: km, time: day, currency: EUR}}
products:
  - {{product_id: p1, name: P1, holding_cost_per_unit_day: 0, backlog_cost_per_unit_day: 0, late_penalty_per_unit_day: 0}}
nodes:
  - {{node_id: n1, name: N1, node_type: SUPPLIER, latitude: null, longitude: null, storage_capacity: 10, processing_capacity: 10, source_capacity: 10}}
  - {{node_id: n2, name: N2, node_type: PLANT, latitude: null, longitude: null, storage_capacity: 10, processing_capacity: 10, source_capacity: 0}}
edges:
  - {{edge_id: e1, origin_node_id: n1, destination_node_id: n2, mode: ROAD, distance_km: 1, base_lead_time_days: 1, daily_capacity: 10, unit_transport_cost: 1, reliability: 1.0, emergency: false}}
initial_inventory: []
demand_process: {{destination_node_id: n2, product_id: p1, distribution: TRUNCATED_NORMAL, mean_daily_demand: 1, standard_deviation: 0, minimum_daily_demand: 1, maximum_daily_demand: 1}}
replenishment_plan: {{product_id: p1, origin_node_id: n1, destination_node_id: n2, first_release_day: 1, release_every_days: 1, shipment_quantity: 1, due_offset_days: 1, initial_route_edge_ids: [e1]}}
action_costs: {{reroute_cost_per_unit: 0, expedite_premium_per_unit: 0}}
{extra}
"""

_THREE_NODE_NETWORK_YAML = """
schema_version: 1
network_id: three_node
units: {{quantity: units, distance: km, time: day, currency: EUR}}
products:
  - {{product_id: p1, name: P1, holding_cost_per_unit_day: 0, backlog_cost_per_unit_day: 0, late_penalty_per_unit_day: 0}}
nodes:
  - {{node_id: n1, name: N1, node_type: SUPPLIER, latitude: null, longitude: null, storage_capacity: 10, processing_capacity: 10, source_capacity: 10}}
  - {{node_id: n2, name: N2, node_type: HUB, latitude: null, longitude: null, storage_capacity: 10, processing_capacity: 10, source_capacity: 0}}
  - {{node_id: n3, name: N3, node_type: PLANT, latitude: null, longitude: null, storage_capacity: 10, processing_capacity: 10, source_capacity: 0}}
edges:
  - {{edge_id: e1, origin_node_id: n1, destination_node_id: n2, mode: ROAD, distance_km: 1, base_lead_time_days: 1, daily_capacity: 10, unit_transport_cost: 1, reliability: 1.0, emergency: false}}
  - {{edge_id: e2, origin_node_id: n2, destination_node_id: n3, mode: ROAD, distance_km: 1, base_lead_time_days: 1, daily_capacity: 10, unit_transport_cost: 1, reliability: 1.0, emergency: false}}
  - {{edge_id: e3, origin_node_id: n1, destination_node_id: n3, mode: AIR, distance_km: 1, base_lead_time_days: 1, daily_capacity: 10, unit_transport_cost: 1, reliability: 1.0, emergency: true}}
initial_inventory: []
demand_process: {{destination_node_id: n3, product_id: p1, distribution: TRUNCATED_NORMAL, mean_daily_demand: 1, standard_deviation: 0, minimum_daily_demand: 1, maximum_daily_demand: 1}}
replenishment_plan: {{product_id: p1, origin_node_id: {origin}, destination_node_id: {destination}, first_release_day: 1, release_every_days: 1, shipment_quantity: 1, due_offset_days: 1, initial_route_edge_ids: [{route}]}}
action_costs: {{reroute_cost_per_unit: 0, expedite_premium_per_unit: 0}}
"""


class TestLoadNetworkConfig:
    def test_baseline_network_loads(self) -> None:
        network = load_network_config(REPO_ROOT / "configs/networks/baseline_network.yaml")
        assert network.network_id == "baseline_network"
        assert len(network.nodes) == 5
        assert len(network.edges) == 6

    def test_tiny_network_loads(self) -> None:
        network = load_network_config(TINY_NETWORK_CONFIG)
        assert network.network_id == "tiny_network"
        assert len(network.nodes) == 3
        assert len(network.edges) == 3

    def test_extra_field_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(_MINIMAL_NETWORK_YAML.format(extra="unexpected_field: true"))
        with pytest.raises(ConfigurationError):
            load_network_config(path)

    def test_bad_identifier_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(
            _MINIMAL_NETWORK_YAML.format(extra="").replace("network_id: minimal", "network_id: Bad-ID")
        )
        with pytest.raises(ConfigurationError):
            load_network_config(path)

    def test_malformed_yaml_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("not: valid: yaml: [")
        with pytest.raises(ConfigurationError):
            load_network_config(path)

    def test_directory_instead_of_file_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="cannot read"):
            load_network_config(tmp_path)

    def test_non_mapping_yaml_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n")
        with pytest.raises(ConfigurationError, match="must contain a mapping"):
            load_network_config(path)

    def test_node_source_capacity_for_non_supplier_rejected(self, tmp_path: Path) -> None:
        text = _MINIMAL_NETWORK_YAML.format(extra="").replace(
            "source_capacity: 0}", "source_capacity: 5}"
        )
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(ConfigurationError):
            load_network_config(path)

    def test_edge_same_origin_and_destination_rejected(self, tmp_path: Path) -> None:
        text = _MINIMAL_NETWORK_YAML.format(extra="").replace(
            "destination_node_id: n2, mode: ROAD", "destination_node_id: n1, mode: ROAD"
        )
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(ConfigurationError):
            load_network_config(path)

    def test_demand_min_above_max_rejected(self, tmp_path: Path) -> None:
        text = _MINIMAL_NETWORK_YAML.format(extra="").replace(
            "minimum_daily_demand: 1, maximum_daily_demand: 1",
            "minimum_daily_demand: 5, maximum_daily_demand: 1",
        )
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(ConfigurationError):
            load_network_config(path)

    def test_duplicate_node_id_rejected(self, tmp_path: Path) -> None:
        text = _MINIMAL_NETWORK_YAML.format(extra="").replace(
            "  - {node_id: n2,",
            "  - {node_id: n1, name: N1Dup, node_type: PLANT, latitude: null, longitude: null, "
            "storage_capacity: 10, processing_capacity: 10, source_capacity: 0}\n"
            "  - {node_id: n2,",
        )
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(ConfigurationError):
            load_network_config(path)

    def test_duplicate_edge_id_rejected(self, tmp_path: Path) -> None:
        text = _MINIMAL_NETWORK_YAML.format(extra="").replace(
            "edges:\n  - {edge_id: e1,",
            "edges:\n  - {edge_id: e1, origin_node_id: n1, destination_node_id: n2, mode: RAIL, "
            "distance_km: 2, base_lead_time_days: 1, daily_capacity: 5, unit_transport_cost: 2, "
            "reliability: 1.0, emergency: false}\n  - {edge_id: e1,",
        )
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(ConfigurationError):
            load_network_config(path)

    def test_duplicate_product_id_rejected(self, tmp_path: Path) -> None:
        text = _MINIMAL_NETWORK_YAML.format(extra="").replace(
            "products:\n  - {product_id: p1,",
            "products:\n  - {product_id: p1, name: P1Dup, holding_cost_per_unit_day: 0, "
            "backlog_cost_per_unit_day: 0, late_penalty_per_unit_day: 0}\n  - {product_id: p1,",
        )
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(ConfigurationError):
            load_network_config(path)

    def test_edge_unknown_origin_rejected(self, tmp_path: Path) -> None:
        text = _MINIMAL_NETWORK_YAML.format(extra="").replace(
            "origin_node_id: n1, destination_node_id: n2, mode: ROAD",
            "origin_node_id: missing, destination_node_id: n2, mode: ROAD",
        )
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(ConfigurationError, match="unknown origin_node_id"):
            load_network_config(path)

    def test_edge_unknown_destination_rejected(self, tmp_path: Path) -> None:
        text = _MINIMAL_NETWORK_YAML.format(extra="").replace(
            "destination_node_id: n2, mode: ROAD", "destination_node_id: missing, mode: ROAD"
        )
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(ConfigurationError, match="unknown destination_node_id"):
            load_network_config(path)

    def test_initial_inventory_unknown_node_rejected(self, tmp_path: Path) -> None:
        text = _MINIMAL_NETWORK_YAML.format(extra="").replace(
            "initial_inventory: []",
            "initial_inventory:\n  - {node_id: missing, product_id: p1, quantity: 1}",
        )
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(ConfigurationError, match="unknown node_id"):
            load_network_config(path)

    def test_initial_inventory_unknown_product_rejected(self, tmp_path: Path) -> None:
        text = _MINIMAL_NETWORK_YAML.format(extra="").replace(
            "initial_inventory: []",
            "initial_inventory:\n  - {node_id: n1, product_id: missing, quantity: 1}",
        )
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(ConfigurationError, match="unknown product_id"):
            load_network_config(path)

    def test_demand_process_unknown_destination_rejected(self, tmp_path: Path) -> None:
        text = _MINIMAL_NETWORK_YAML.format(extra="").replace(
            "destination_node_id: n2, product_id: p1, distribution: TRUNCATED_NORMAL",
            "destination_node_id: missing, product_id: p1, distribution: TRUNCATED_NORMAL",
        )
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(ConfigurationError, match="demand_process references unknown destination"):
            load_network_config(path)

    def test_demand_process_unknown_product_rejected(self, tmp_path: Path) -> None:
        text = _MINIMAL_NETWORK_YAML.format(extra="").replace(
            "product_id: p1, distribution: TRUNCATED_NORMAL",
            "product_id: missing, distribution: TRUNCATED_NORMAL",
        )
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(ConfigurationError, match="demand_process references unknown product"):
            load_network_config(path)

    def test_replenishment_plan_unknown_origin_rejected(self, tmp_path: Path) -> None:
        text = _MINIMAL_NETWORK_YAML.format(extra="").replace(
            "origin_node_id: n1, destination_node_id: n2, first_release_day",
            "origin_node_id: missing, destination_node_id: n2, first_release_day",
        )
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(ConfigurationError, match="replenishment_plan references unknown origin"):
            load_network_config(path)

    def test_replenishment_plan_unknown_destination_rejected(self, tmp_path: Path) -> None:
        text = _MINIMAL_NETWORK_YAML.format(extra="").replace(
            "destination_node_id: n2, first_release_day", "destination_node_id: missing, first_release_day"
        )
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(
            ConfigurationError, match="replenishment_plan references unknown destination"
        ):
            load_network_config(path)

    def test_replenishment_plan_unknown_product_rejected(self, tmp_path: Path) -> None:
        text = _MINIMAL_NETWORK_YAML.format(extra="").replace(
            "replenishment_plan: {product_id: p1, origin_node_id",
            "replenishment_plan: {product_id: missing, origin_node_id",
        )
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(ConfigurationError, match="replenishment_plan references unknown product"):
            load_network_config(path)

    def test_empty_route_rejected(self, tmp_path: Path) -> None:
        text = _MINIMAL_NETWORK_YAML.format(extra="").replace(
            "initial_route_edge_ids: [e1]", "initial_route_edge_ids: []"
        )
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(ConfigurationError):
            load_network_config(path)

    def test_unknown_edge_reference_rejected(self, tmp_path: Path) -> None:
        text = _MINIMAL_NETWORK_YAML.format(extra="").replace(
            "initial_route_edge_ids: [e1]", "initial_route_edge_ids: [missing_edge]"
        )
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(ConfigurationError, match="unknown edge_id"):
            load_network_config(path)

    def test_route_wrong_origin_rejected(self, tmp_path: Path) -> None:
        text = _THREE_NODE_NETWORK_YAML.format(origin="n2", destination="n3", route="e1, e2")
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(ConfigurationError, match="does not begin at"):
            load_network_config(path)

    def test_route_wrong_destination_rejected(self, tmp_path: Path) -> None:
        text = _THREE_NODE_NETWORK_YAML.format(origin="n1", destination="n1", route="e1, e2")
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(ConfigurationError, match="does not end at"):
            load_network_config(path)

    def test_route_not_continuous_rejected(self, tmp_path: Path) -> None:
        text = _THREE_NODE_NETWORK_YAML.format(origin="n1", destination="n3", route="e1, e3")
        path = tmp_path / "bad.yaml"
        path.write_text(text)
        with pytest.raises(ConfigurationError, match="not a continuous"):
            load_network_config(path)


class TestShockConfigValidation:
    def _base_shock(self) -> ShockConfig:
        resolved = resolve_config(BASELINE_EXPERIMENT_CONFIG, REPO_ROOT)
        return resolved.scenario.shocks[0]

    def test_end_before_start_rejected(self) -> None:
        data = self._base_shock().model_dump()
        data["physical_end_day"] = data["physical_start_day"] - 1
        with pytest.raises(ValidationError, match="physical_end_day"):
            ShockConfig(**data)


class TestScenarioConfigValidation:
    def test_duplicate_shock_ids_rejected(self) -> None:
        resolved = resolve_config(BASELINE_EXPERIMENT_CONFIG, REPO_ROOT)
        shock_data = resolved.scenario.shocks[0].model_dump()
        data = resolved.scenario.model_dump()
        data["shocks"] = [shock_data, shock_data]
        with pytest.raises(ValidationError, match="duplicate shock_id"):
            ScenarioConfig(**data)


class TestLLMPolicyConfigValidation:
    def test_replay_mode_requires_trace_path_rejected(self) -> None:
        resolved = resolve_config(BASELINE_EXPERIMENT_CONFIG, REPO_ROOT)
        data = resolved.llm_policy.model_dump()
        data["execution_mode"] = "REPLAY"
        data["replay_trace_path"] = None
        with pytest.raises(ValidationError, match="replay_trace_path"):
            LLMPolicyConfig(**data)


class TestExperimentConfigValidation:
    def test_warmup_not_less_than_horizon_rejected(self) -> None:
        resolved = resolve_config(BASELINE_EXPERIMENT_CONFIG, REPO_ROOT)
        data = resolved.experiment.model_dump()
        data["horizon_days"] = data["warmup_days"]
        with pytest.raises(ValidationError, match="warmup_days"):
            ExperimentConfig(**data)


class TestResolvedConfigValidation:
    def _resolved(self) -> ResolvedConfig:
        return resolve_config(BASELINE_EXPERIMENT_CONFIG, REPO_ROOT)

    def _rebuild(self, resolved: ResolvedConfig, scenario: ScenarioConfig) -> ResolvedConfig:
        return ResolvedConfig(
            experiment=resolved.experiment,
            network=resolved.network,
            scenario=scenario,
            heuristic_policy=resolved.heuristic_policy,
            llm_policy=resolved.llm_policy,
            experiment_config_path=resolved.experiment_config_path,
            network_config_path=resolved.network_config_path,
            scenario_config_path=resolved.scenario_config_path,
            heuristic_config_path=resolved.heuristic_config_path,
            llm_config_path=resolved.llm_config_path,
            output_root=resolved.output_root,
        )

    def test_shock_not_after_warmup_rejected(self) -> None:
        resolved = self._resolved()
        shock = resolved.scenario.shocks[0]
        early_shock = ShockConfig(
            **{**shock.model_dump(), "physical_start_day": resolved.experiment.warmup_days}
        )
        bad_scenario = ScenarioConfig(
            **{**resolved.scenario.model_dump(), "shocks": [early_shock.model_dump()]}
        )
        with pytest.raises(ValidationError, match="warmup_days"):
            self._rebuild(resolved, bad_scenario)

    def test_shock_unknown_node_target_rejected(self) -> None:
        resolved = self._resolved()
        shock = resolved.scenario.shocks[0]
        bad_shock = ShockConfig(**{**shock.model_dump(), "target_id": "no_such_node"})
        bad_scenario = ScenarioConfig(
            **{**resolved.scenario.model_dump(), "shocks": [bad_shock.model_dump()]}
        )
        with pytest.raises(ValidationError, match="unknown node"):
            self._rebuild(resolved, bad_scenario)

    def test_shock_unknown_edge_target_rejected(self) -> None:
        resolved = self._resolved()
        shock = resolved.scenario.shocks[0]
        bad_shock = ShockConfig(
            **{**shock.model_dump(), "target_type": "EDGE", "target_id": "no_such_edge"}
        )
        bad_scenario = ScenarioConfig(
            **{**resolved.scenario.model_dump(), "shocks": [bad_shock.model_dump()]}
        )
        with pytest.raises(ValidationError, match="unknown edge"):
            self._rebuild(resolved, bad_scenario)


class TestResolveConfig:
    def test_baseline_experiment_resolves(self) -> None:
        resolved = resolve_config(BASELINE_EXPERIMENT_CONFIG, REPO_ROOT)
        assert resolved.experiment.experiment_id == "baseline_port_closure_comparison"
        assert resolved.network.network_id == "baseline_network"
        assert resolved.scenario.scenario_id == "primary_port_closure"
        assert resolved.heuristic_policy.policy_id == "heuristic"
        assert resolved.llm_policy.policy_id == "llm_agent"
        assert resolved.output_root == (REPO_ROOT / "outputs").resolve()

    def test_missing_experiment_file_rejected(self) -> None:
        missing_path = REPO_ROOT / "configs/experiments/does_not_exist.yaml"
        with pytest.raises(ConfigurationError, match="does not exist"):
            resolve_config(missing_path, REPO_ROOT)

    def test_path_escaping_repo_root_rejected(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        outside_file = tmp_path / "outside.yaml"
        outside_file.write_text("not used")
        with pytest.raises(ConfigurationError, match="escapes the repository root"):
            loaders._resolve_within_repo(outside_file, repo_root)


class TestBuildNetworkDefinition:
    def _tiny_config(self) -> NetworkConfig:
        return load_network_config(TINY_NETWORK_CONFIG)

    def test_produces_expected_nodes_and_types(self) -> None:
        network_definition = build_network_definition(self._tiny_config())
        assert set(network_definition.nodes) == {"supplier_1", "hub_1", "plant_1"}
        assert network_definition.get_node("supplier_1").node_type is NodeType.SUPPLIER
        assert network_definition.get_node("hub_1").node_type is NodeType.HUB
        assert network_definition.get_node("plant_1").node_type is NodeType.PLANT

    def test_produces_expected_edges_and_emergency_flag(self) -> None:
        network_definition = build_network_definition(self._tiny_config())
        assert network_definition.get_edge("supplier_to_hub").mode is TransportMode.ROAD
        assert network_definition.get_edge("supplier_to_hub").emergency is False
        assert network_definition.get_edge("supplier_to_plant_air").mode is TransportMode.AIR
        assert network_definition.get_edge("supplier_to_plant_air").emergency is True

    def test_produces_expected_products(self) -> None:
        network_definition = build_network_definition(self._tiny_config())
        product = network_definition.get_product("widget")
        assert product.holding_cost_per_unit_day == pytest.approx(0.10)


class TestBuildInitialState:
    def test_day_zero_state_matches_contract(self) -> None:
        network_config = load_network_config(TINY_NETWORK_CONFIG)
        network_definition = build_network_definition(network_config)
        state = build_initial_state(network_definition, network_config)

        assert state.day == 0
        assert state.inventory["plant_1"]["widget"] == 10
        assert state.inventory["supplier_1"]["widget"] == 0
        assert all(
            quantity == 0
            for node_backlog in state.backlog.values()
            for quantity in node_backlog.values()
        )
        assert state.shipments == {}
        assert state.active_shock_ids == set()
        assert state.known_shock_ids == set()
        assert state.costs.transport == 0.0
        assert state.costs.holding == 0.0
        assert state.service.total_demand_units == 0
        assert all(used == 0 for used in state.daily_edge_used_capacity.values())
        assert all(used == 0 for used in state.daily_node_used_processing.values())

    def test_operational_states_default_to_available(self) -> None:
        network_config = load_network_config(TINY_NETWORK_CONFIG)
        network_definition = build_network_definition(network_config)
        state = build_initial_state(network_definition, network_config)

        for node_state in state.node_operational_state.values():
            assert node_state.available is True
            assert node_state.processing_capacity_multiplier == 1.0
        for edge_state in state.edge_operational_state.values():
            assert edge_state.available is True
            assert edge_state.capacity_multiplier == 1.0
            assert edge_state.lead_time_multiplier == 1.0
            assert edge_state.cost_multiplier == 1.0
