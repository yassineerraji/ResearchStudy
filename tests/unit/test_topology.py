"""Unit tests for the three V2 topology tiers: connectivity, route continuity, and the reroute-availability property each tier exists to demonstrate."""

from __future__ import annotations

from pathlib import Path

import networkx as nx  # type: ignore[import-untyped]

from supply_chain_simulator.data_io.loaders import (
    build_initial_state,
    build_network_definition,
    load_network_config,
)
from supply_chain_simulator.domain.state import Shipment, ShipmentStatus
from supply_chain_simulator.simulation.routing import enumerate_candidate_routes

REPO_ROOT = Path(__file__).resolve().parents[2]
NETWORKS_DIR = REPO_ROOT / "configs/networks"


def _raw_non_emergency_path_count(config_filename: str, closed_node_id: str) -> int:
    """Counts structurally distinct supplier->plant paths at the pure graph
    level, bypassing simulation/routing.py's MAX_ROUTE_OPTIONS=5 candidate
    cap -- used only where the true (uncapped) structural fact matters, e.g.
    proving a node barely affects connectivity when several more than 5
    genuine alternatives exist. Route-visible reroute counts (what a policy
    actually gets to choose from) use enumerate_candidate_routes instead.
    """
    network_config = load_network_config(NETWORKS_DIR / config_filename)
    network_definition = build_network_definition(network_config)
    graph = nx.DiGraph()
    for edge in network_definition.edges.values():
        graph.add_edge(edge.origin_node_id, edge.destination_node_id, emergency=edge.emergency)
    graph.remove_node(closed_node_id)

    plan = network_config.replenishment_plan
    paths = nx.all_simple_paths(graph, plan.origin_node_id, plan.destination_node_id)
    return sum(1 for path in paths if len(path) > 2)  # excludes the 1-edge emergency air lane


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


def _non_emergency_reroutes_around_node_closure(
    config_filename: str, closed_node_id: str
) -> list[tuple[str, ...]]:
    network_config = load_network_config(NETWORKS_DIR / config_filename)
    network_definition = build_network_definition(network_config)
    state = build_initial_state(network_definition, network_config)
    state.node_operational_state[closed_node_id].available = False

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
        reroutes = _non_emergency_reroutes_around_node_closure("topology_compact.yaml", "port_primary")
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
        reroutes = _non_emergency_reroutes_around_node_closure("baseline_network.yaml", "port_primary")
        assert len(reroutes) == 1


class TestExtendedTopology:
    def test_node_and_edge_connectivity(self) -> None:
        network_definition = build_network_definition(load_network_config(NETWORKS_DIR / "topology_extended.yaml"))
        assert set(network_definition.nodes) == {
            "supplier_1",
            "port_primary",
            "port_alternative",
            "port_tertiary",
            "port_quaternary",
            "hub_1",
            "hub_2",
            "hub_3",
            "hub_4",
            "plant_1",
        }
        assert len(network_definition.edges) == 16

    def test_replenishment_route_is_continuous(self) -> None:
        network_config = load_network_config(NETWORKS_DIR / "topology_extended.yaml")
        plan = network_config.replenishment_plan
        edges = {edge.edge_id: edge for edge in network_config.edges}
        route = [edges[edge_id] for edge_id in plan.initial_route_edge_ids]
        assert route[0].origin_node_id == plan.origin_node_id
        assert route[-1].destination_node_id == plan.destination_node_id

    def test_port_primary_is_no_longer_the_critical_node(self) -> None:
        """The exact finding that drove retargeting this tier's severity
        scenario to hub_1 (CLAUDE.md V2.3.1): in this mesh, port_primary is
        one of the least central nodes -- closing it barely dents route
        availability (6 of 7 non-emergency paths survive at the graph level),
        unlike Compact/Standard where it's the only chokepoint. This checks
        the raw graph fact, not the policy-visible candidate count, since
        Extended has more genuine paths than MAX_ROUTE_OPTIONS=5 can surface
        at once (routing.py's own candidate cap -- a separate, expected
        constraint, not a structural property of this tier).
        """
        assert _raw_non_emergency_path_count("topology_extended.yaml", "port_primary") == 6

    def test_hub_1_closure_is_more_disruptive_than_port_primary_at_the_graph_level(self) -> None:
        assert _raw_non_emergency_path_count("topology_extended.yaml", "hub_1") == 5
        assert _raw_non_emergency_path_count(
            "topology_extended.yaml", "hub_1"
        ) < _raw_non_emergency_path_count("topology_extended.yaml", "port_primary")

    def test_has_several_distinct_non_emergency_reroutes_around_hub_1_closure(self) -> None:
        """The property this tier's redesign exists to prove: hub_1, not
        port_primary, is the structurally critical node here (verified by
        networkx.betweenness_centrality during design -- see
        configs/networks/topology_extended.yaml's header comment) --  and
        even so, real mesh redundancy leaves most routes intact.
        """
        reroutes = _non_emergency_reroutes_around_node_closure("topology_extended.yaml", "hub_1")
        assert len(reroutes) == 5  # 5 of 7 non-emergency paths survive
        assert len(set(reroutes)) == len(reroutes)  # each candidate is structurally distinct
