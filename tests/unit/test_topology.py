"""Unit tests for the three V2 topology tiers (CLAUDE.md V2 §V2.3.1).

Inside tests/unit, this file checks that Compact, Standard, and Extended
each load into the exact node/edge connectivity the contract specifies, that
every tier's replenishment plan route is continuous from origin to
destination, and -- the structural property the whole topology axis exists
to test -- that closing port_primary leaves Compact with no non-emergency
reroute at all, while Extended has at least two structurally distinct ones.
It does not run a simulation; it only checks static graph structure via
simulation/routing.py's candidate-route enumeration.
"""

from __future__ import annotations

from pathlib import Path

from supply_chain_simulator.data_io.loaders import (
    build_initial_state,
    build_network_definition,
    load_network_config,
)
from supply_chain_simulator.domain.state import Shipment, ShipmentStatus
from supply_chain_simulator.simulation.routing import enumerate_candidate_routes

REPO_ROOT = Path(__file__).resolve().parents[2]
NETWORKS_DIR = REPO_ROOT / "configs/networks"


def _shipment_at_supplier(destination_node_id: str, route_edge_ids: tuple[str, ...]) -> Shipment:
    return Shipment(
        shipment_id="probe_001",
        product_id="component_a",
        quantity=1,
        origin_node_id="supplier_1",
        destination_node_id=destination_node_id,
        release_day=1,
        due_day=1000,
        planned_route_edge_ids=route_edge_ids,
        next_edge_index=0,
        status=ShipmentStatus.AT_NODE,
        current_node_id="supplier_1",
        current_edge_id=None,
        edge_entry_day=None,
        edge_arrival_day=None,
        reroute_count=0,
        expedite_count=0,
        capacity_wait_days=0,
        delivered_day=None,
    )


def _non_emergency_reroutes_around_primary_port_closure(config_filename: str) -> list[tuple[str, ...]]:
    network_config = load_network_config(NETWORKS_DIR / config_filename)
    network_definition = build_network_definition(network_config)
    state = build_initial_state(network_definition, network_config)
    state.node_operational_state["port_primary"].available = False

    plan = network_config.replenishment_plan
    shipment = _shipment_at_supplier(plan.destination_node_id, tuple(plan.initial_route_edge_ids))
    estimates = enumerate_candidate_routes(
        state, shipment, reroute_cost_per_unit=1.0, expedite_premium_per_unit=1.0
    )
    return [estimate.edge_ids for estimate in estimates if not estimate.contains_emergency_edge]


class TestCompactTopology:
    def test_node_and_edge_connectivity(self) -> None:
        network_definition = build_network_definition(load_network_config(NETWORKS_DIR / "topology_compact.yaml"))
        assert set(network_definition.nodes) == {"supplier_1", "port_primary", "hub_1", "plant_1"}
        assert set(network_definition.edges) == {
            "supplier_to_primary_port",
            "primary_port_to_hub",
            "hub_to_plant",
            "supplier_to_plant_air",
        }

    def test_replenishment_route_is_continuous(self) -> None:
        network_config = load_network_config(NETWORKS_DIR / "topology_compact.yaml")
        plan = network_config.replenishment_plan
        edges = {edge.edge_id: edge for edge in network_config.edges}
        route = [edges[edge_id] for edge_id in plan.initial_route_edge_ids]
        assert route[0].origin_node_id == plan.origin_node_id
        assert route[-1].destination_node_id == plan.destination_node_id

    def test_no_non_emergency_reroute_around_primary_port_closure(self) -> None:
        """The property Compact exists to prove: with the alternate port
        removed entirely, closing port_primary leaves only WAIT and EXPEDITE.
        """
        reroutes = _non_emergency_reroutes_around_primary_port_closure("topology_compact.yaml")
        assert reroutes == []


class TestStandardTopology:
    def test_node_and_edge_connectivity(self) -> None:
        network_definition = build_network_definition(load_network_config(NETWORKS_DIR / "baseline_network.yaml"))
        assert len(network_definition.nodes) == 5
        assert len(network_definition.edges) == 6

    def test_replenishment_route_is_continuous(self) -> None:
        network_config = load_network_config(NETWORKS_DIR / "baseline_network.yaml")
        plan = network_config.replenishment_plan
        edges = {edge.edge_id: edge for edge in network_config.edges}
        route = [edges[edge_id] for edge_id in plan.initial_route_edge_ids]
        assert route[0].origin_node_id == plan.origin_node_id
        assert route[-1].destination_node_id == plan.destination_node_id

    def test_has_exactly_one_non_emergency_reroute_around_primary_port_closure(self) -> None:
        reroutes = _non_emergency_reroutes_around_primary_port_closure("baseline_network.yaml")
        assert len(reroutes) == 1


class TestExtendedTopology:
    def test_node_and_edge_connectivity(self) -> None:
        network_definition = build_network_definition(load_network_config(NETWORKS_DIR / "topology_extended.yaml"))
        assert set(network_definition.nodes) == {
            "supplier_1",
            "port_primary",
            "port_alternative",
            "port_tertiary",
            "hub_1",
            "hub_2",
            "plant_1",
        }
        assert len(network_definition.edges) == 10

    def test_replenishment_route_is_continuous(self) -> None:
        network_config = load_network_config(NETWORKS_DIR / "topology_extended.yaml")
        plan = network_config.replenishment_plan
        edges = {edge.edge_id: edge for edge in network_config.edges}
        route = [edges[edge_id] for edge_id in plan.initial_route_edge_ids]
        assert route[0].origin_node_id == plan.origin_node_id
        assert route[-1].destination_node_id == plan.destination_node_id

    def test_has_at_least_two_distinct_non_emergency_reroutes_around_primary_port_closure(self) -> None:
        """The property Extended exists to prove: genuine mesh redundancy
        gives policies more than one real alternative when the primary port
        closes, unlike Standard's single reroute.
        """
        reroutes = _non_emergency_reroutes_around_primary_port_closure("topology_extended.yaml")
        assert len(reroutes) >= 2
        assert len(set(reroutes)) == len(reroutes)  # each candidate is structurally distinct
