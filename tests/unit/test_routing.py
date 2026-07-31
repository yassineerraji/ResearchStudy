"""Unit tests for simulation/routing.py: graph, candidates, and estimates.

Inside tests/unit, this file checks route continuity, that unavailable nodes
and edges are excluded from new candidate routes, that emergency routes are
identified correctly, that candidate routes are sorted and limited to five
per CLAUDE.md section 11.6's tie-break order, that static capacity filtering
rejects routes a shipment cannot physically fit through, that transport-cost
and lead-time estimates match hand-calculated values, and that estimating a
blocked current plan uses a known reopening day (or reports None when none
is known). It does not test simulation/costs.py or any day-to-day state
mutation, since transition.py does not exist yet at this stage of the build.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from supply_chain_simulator.data_io.loaders import (
    build_initial_state,
    build_network_definition,
    load_network_config,
)
from supply_chain_simulator.domain.events import Shock, ShockType, TargetType
from supply_chain_simulator.domain.models import (
    Edge,
    NetworkDefinition,
    Node,
    NodeType,
    Product,
    TransportMode,
)
from supply_chain_simulator.domain.state import (
    OperationalEdgeState,
    OperationalNodeState,
    Shipment,
    ShipmentStatus,
    SimulationState,
)
from supply_chain_simulator.simulation.routing import (
    RoutingError,
    build_operational_graph,
    enumerate_candidate_routes,
    estimate_current_plan,
    estimate_route_option,
    get_effective_edge_capacity,
    get_effective_edge_cost,
    get_effective_edge_lead_time,
    route_from_edge_ids,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_NETWORK_CONFIG = REPO_ROOT / "tests/fixtures/tiny_network.yaml"


def _tiny_state(day: int = 1) -> SimulationState:
    network_config = load_network_config(TINY_NETWORK_CONFIG)
    network_definition = build_network_definition(network_config)
    state = build_initial_state(network_definition, network_config)
    state.day = day
    return state


def _shipment(
    current_node_id: str = "supplier_1",
    destination_node_id: str = "plant_1",
    planned_route_edge_ids: tuple[str, ...] = (),
    next_edge_index: int = 0,
    quantity: int = 5,
    due_day: int = 100,
) -> Shipment:
    return Shipment(
        shipment_id="shipment_001_001",
        product_id="widget",
        quantity=quantity,
        origin_node_id="supplier_1",
        destination_node_id=destination_node_id,
        release_day=1,
        due_day=due_day,
        planned_route_edge_ids=planned_route_edge_ids,
        next_edge_index=next_edge_index,
        status=ShipmentStatus.AT_NODE,
        current_node_id=current_node_id,
        current_edge_id=None,
        edge_entry_day=None,
        edge_arrival_day=None,
        reroute_count=0,
        expedite_count=0,
        capacity_wait_days=0,
        delivered_day=None,
    )


class TestGetEffectiveEdgeValues:
    def _edge(self, **overrides: object) -> Edge:
        defaults: dict[str, object] = {
            "edge_id": "e1",
            "origin_node_id": "a",
            "destination_node_id": "b",
            "mode": TransportMode.ROAD,
            "distance_km": 10.0,
            "base_lead_time_days": 2,
            "daily_capacity": 10,
            "unit_transport_cost": 4.0,
            "reliability": 1.0,
            "emergency": False,
        }
        defaults.update(overrides)
        return Edge(**defaults)  # type: ignore[arg-type]

    def test_capacity_floors_the_product(self) -> None:
        edge = self._edge(daily_capacity=10)
        state = OperationalEdgeState(capacity_multiplier=0.55)
        assert get_effective_edge_capacity(edge, state) == 5

    def test_lead_time_ceils_and_floors_at_one(self) -> None:
        edge = self._edge(base_lead_time_days=2)
        assert get_effective_edge_lead_time(edge, OperationalEdgeState(lead_time_multiplier=1.4)) == 3
        assert get_effective_edge_lead_time(edge, OperationalEdgeState(lead_time_multiplier=0.0)) == 1

    def test_cost_multiplies_directly(self) -> None:
        edge = self._edge(unit_transport_cost=4.0)
        cost = get_effective_edge_cost(edge, OperationalEdgeState(cost_multiplier=1.5))
        assert cost == pytest.approx(6.0)


class TestRouteFromEdgeIds:
    def test_continuous_route_builds(self) -> None:
        state = _tiny_state()
        route = route_from_edge_ids(state, ("supplier_to_hub", "hub_to_plant"))
        assert route.route_id == "supplier_to_hub__hub_to_plant"
        assert route.node_ids == ("supplier_1", "hub_1", "plant_1")
        assert route.contains_emergency_edge is False

    def test_emergency_route_flagged(self) -> None:
        state = _tiny_state()
        route = route_from_edge_ids(state, ("supplier_to_plant_air",))
        assert route.contains_emergency_edge is True

    def test_discontinuous_route_rejected(self) -> None:
        state = _tiny_state()
        with pytest.raises(RoutingError, match="not continuous"):
            route_from_edge_ids(state, ("hub_to_plant", "supplier_to_hub"))

    def test_empty_route_rejected(self) -> None:
        state = _tiny_state()
        with pytest.raises(RoutingError, match="at least one edge"):
            route_from_edge_ids(state, ())


class TestBuildOperationalGraph:
    def test_unavailable_node_and_its_edges_excluded(self) -> None:
        state = _tiny_state()
        state.node_operational_state["hub_1"].available = False
        graph = build_operational_graph(state)
        assert "hub_1" not in graph
        assert not graph.has_edge("supplier_1", "hub_1")
        assert graph.has_edge("supplier_1", "plant_1")  # the direct emergency edge is unaffected

    def test_unavailable_edge_excluded_but_node_kept(self) -> None:
        state = _tiny_state()
        state.edge_operational_state["supplier_to_hub"].available = False
        graph = build_operational_graph(state)
        assert "hub_1" in graph
        assert not graph.has_edge("supplier_1", "hub_1")
        assert graph.has_edge("hub_1", "plant_1")

    def test_edge_attributes_reflect_multipliers(self) -> None:
        state = _tiny_state()
        state.edge_operational_state["supplier_to_hub"].cost_multiplier = 2.0
        graph = build_operational_graph(state)
        assert graph["supplier_1"]["hub_1"]["cost"] == pytest.approx(2.0)


class TestEnumerateCandidateRoutes:
    def test_finds_normal_and_emergency_routes(self) -> None:
        state = _tiny_state()
        shipment = _shipment(quantity=5)
        estimates = enumerate_candidate_routes(
            state, shipment, reroute_cost_per_unit=1.0, expedite_premium_per_unit=2.0
        )
        route_ids = {estimate.route_id for estimate in estimates}
        assert "supplier_to_hub__hub_to_plant" in route_ids
        assert "supplier_to_plant_air" in route_ids
        emergency_flags = {estimate.route_id: estimate.contains_emergency_edge for estimate in estimates}
        assert emergency_flags["supplier_to_hub__hub_to_plant"] is False
        assert emergency_flags["supplier_to_plant_air"] is True

    def test_requires_shipment_at_node(self) -> None:
        # Shipment.__post_init__ forbids constructing this combination directly, since a
        # real AT_NODE shipment always has a current_node_id; mutate after construction to
        # simulate a caller mistakenly passing an in-transit shipment, which the guard
        # exists to catch.
        state = _tiny_state()
        shipment = _shipment()
        shipment.current_node_id = None
        with pytest.raises(RoutingError, match="AT_NODE"):
            enumerate_candidate_routes(state, shipment, reroute_cost_per_unit=1.0, expedite_premium_per_unit=2.0)

    def test_unavailable_edge_removed_from_candidates(self) -> None:
        state = _tiny_state()
        state.edge_operational_state["supplier_to_hub"].available = False
        shipment = _shipment(quantity=5)
        estimates = enumerate_candidate_routes(
            state, shipment, reroute_cost_per_unit=1.0, expedite_premium_per_unit=2.0
        )
        route_ids = {estimate.route_id for estimate in estimates}
        assert "supplier_to_hub__hub_to_plant" not in route_ids
        assert "supplier_to_plant_air" in route_ids

    def test_static_edge_capacity_filters_incompatible_route(self) -> None:
        state = _tiny_state()
        # supplier_to_plant_air has daily_capacity 10; the normal route's edges have 20.
        shipment = _shipment(quantity=15)
        estimates = enumerate_candidate_routes(
            state, shipment, reroute_cost_per_unit=1.0, expedite_premium_per_unit=2.0
        )
        route_ids = {estimate.route_id for estimate in estimates}
        assert "supplier_to_plant_air" not in route_ids
        assert "supplier_to_hub__hub_to_plant" in route_ids

    def test_static_node_processing_capacity_filters_incompatible_route(self) -> None:
        nodes = {
            "a": Node(
                node_id="a",
                name="A",
                node_type=NodeType.SUPPLIER,
                latitude=None,
                longitude=None,
                storage_capacity=1000,
                processing_capacity=3,
                source_capacity=1000,
            ),
            "z": Node(
                node_id="z",
                name="Z",
                node_type=NodeType.PLANT,
                latitude=None,
                longitude=None,
                storage_capacity=1000,
                processing_capacity=1000,
                source_capacity=0,
            ),
        }
        edges = {
            "a_to_z": Edge(
                edge_id="a_to_z",
                origin_node_id="a",
                destination_node_id="z",
                mode=TransportMode.ROAD,
                distance_km=1.0,
                base_lead_time_days=1,
                daily_capacity=1000,
                unit_transport_cost=1.0,
                reliability=1.0,
                emergency=False,
            )
        }
        products = {
            "widget": Product(
                product_id="widget",
                name="Widget",
                holding_cost_per_unit_day=0.0,
                backlog_cost_per_unit_day=0.0,
                late_penalty_per_unit_day=0.0,
            )
        }
        network_definition = NetworkDefinition(nodes=nodes, edges=edges, products=products)
        state = SimulationState(
            day=1,
            network_definition=network_definition,
            node_operational_state={nid: OperationalNodeState() for nid in nodes},
            edge_operational_state={eid: OperationalEdgeState() for eid in edges},
            inventory={nid: {"widget": 0} for nid in nodes},
            backlog={nid: {"widget": 0} for nid in nodes},
            shipments={},
        )
        shipment = _shipment(current_node_id="a", destination_node_id="z", quantity=5)
        estimates = enumerate_candidate_routes(
            state, shipment, reroute_cost_per_unit=1.0, expedite_premium_per_unit=2.0
        )
        assert estimates == ()

    def test_deterministic_across_calls(self) -> None:
        state = _tiny_state()
        shipment = _shipment(quantity=5)
        first = enumerate_candidate_routes(
            state, shipment, reroute_cost_per_unit=1.0, expedite_premium_per_unit=2.0
        )
        second = enumerate_candidate_routes(
            state, shipment, reroute_cost_per_unit=1.0, expedite_premium_per_unit=2.0
        )
        assert first == second



def _linear_hub_network_state() -> tuple[SimulationState, Shipment]:
    """A -> b1..b7 -> Z, with route costs 1,1,2,3,4,5,6 (b1 and b7 tie at 1)."""
    nodes: dict[str, Node] = {
        "a": Node(
            node_id="a",
            name="A",
            node_type=NodeType.SUPPLIER,
            latitude=None,
            longitude=None,
            storage_capacity=1000,
            processing_capacity=1000,
            source_capacity=1000,
        ),
        "z": Node(
            node_id="z",
            name="Z",
            node_type=NodeType.PLANT,
            latitude=None,
            longitude=None,
            storage_capacity=1000,
            processing_capacity=1000,
            source_capacity=0,
        ),
    }
    edges: dict[str, Edge] = {}
    hub_costs = {1: (1.0, 0.0), 7: (0.0, 1.0), 2: (2.0, 0.0), 3: (3.0, 0.0), 4: (4.0, 0.0), 5: (5.0, 0.0), 6: (6.0, 0.0)}
    for index, (first_leg_cost, second_leg_cost) in hub_costs.items():
        hub_id = f"b{index}"
        nodes[hub_id] = Node(
            node_id=hub_id,
            name=hub_id,
            node_type=NodeType.HUB,
            latitude=None,
            longitude=None,
            storage_capacity=1000,
            processing_capacity=1000,
            source_capacity=0,
        )
        edges[f"a_to_{hub_id}"] = Edge(
            edge_id=f"a_to_{hub_id}",
            origin_node_id="a",
            destination_node_id=hub_id,
            mode=TransportMode.ROAD,
            distance_km=1.0,
            base_lead_time_days=1,
            daily_capacity=1000,
            unit_transport_cost=first_leg_cost,
            reliability=1.0,
            emergency=False,
        )
        edges[f"{hub_id}_to_z"] = Edge(
            edge_id=f"{hub_id}_to_z",
            origin_node_id=hub_id,
            destination_node_id="z",
            mode=TransportMode.ROAD,
            distance_km=1.0,
            base_lead_time_days=1,
            daily_capacity=1000,
            unit_transport_cost=second_leg_cost,
            reliability=1.0,
            emergency=False,
        )
    products = {
        "widget": Product(
            product_id="widget",
            name="Widget",
            holding_cost_per_unit_day=0.0,
            backlog_cost_per_unit_day=0.0,
            late_penalty_per_unit_day=0.0,
        )
    }
    network_definition = NetworkDefinition(nodes=nodes, edges=edges, products=products)
    state = SimulationState(
        day=1,
        network_definition=network_definition,
        node_operational_state={nid: OperationalNodeState() for nid in nodes},
        edge_operational_state={eid: OperationalEdgeState() for eid in edges},
        inventory={nid: {"widget": 0} for nid in nodes},
        backlog={nid: {"widget": 0} for nid in nodes},
        shipments={},
    )
    shipment = _shipment(current_node_id="a", destination_node_id="z", quantity=1, due_day=1000)
    return state, shipment


class TestRouteSortingAndLimit:
    def test_sorted_by_cost_then_route_id_and_limited_to_five(self) -> None:
        state, shipment = _linear_hub_network_state()
        estimates = enumerate_candidate_routes(
            state, shipment, reroute_cost_per_unit=0.0, expedite_premium_per_unit=0.0
        )
        assert len(estimates) == 5
        assert [estimate.route_id for estimate in estimates] == [
            "a_to_b1__b1_to_z",
            "a_to_b7__b7_to_z",
            "a_to_b2__b2_to_z",
            "a_to_b3__b3_to_z",
            "a_to_b4__b4_to_z",
        ]
        costs = [estimate.estimated_total_cost for estimate in estimates]
        assert costs == sorted(costs)


class TestEstimateRouteOption:
    def test_transport_cost_and_on_time_arrival(self) -> None:
        state = _tiny_state(day=1)
        shipment = _shipment(due_day=10, quantity=5)
        route = route_from_edge_ids(state, ("supplier_to_hub", "hub_to_plant"))
        estimate = estimate_route_option(state, shipment, route, action_cost=0.0)

        assert estimate.estimated_lead_time_days == 2
        assert estimate.estimated_transport_cost == pytest.approx(10.0)
        assert estimate.estimated_arrival_day == 3
        assert estimate.estimated_lateness_days == 0
        assert estimate.estimated_late_penalty == pytest.approx(0.0)
        assert estimate.estimated_total_cost == pytest.approx(10.0)
        assert estimate.first_edge_remaining_capacity == 20
        assert estimate.currently_dispatchable is True

    def test_late_arrival_charges_late_penalty(self) -> None:
        state = _tiny_state(day=1)
        shipment = _shipment(due_day=2, quantity=5)
        route = route_from_edge_ids(state, ("supplier_to_hub", "hub_to_plant"))
        estimate = estimate_route_option(state, shipment, route, action_cost=0.0)

        assert estimate.estimated_arrival_day == 3
        assert estimate.estimated_lateness_days == 1
        assert estimate.estimated_late_penalty == pytest.approx(5 * 1 * 0.5)
        assert estimate.estimated_total_cost == pytest.approx(10.0 + 2.5)

    def test_action_cost_is_included_in_total(self) -> None:
        state = _tiny_state(day=1)
        shipment = _shipment(due_day=100, quantity=5)
        route = route_from_edge_ids(state, ("supplier_to_plant_air",))
        estimate = estimate_route_option(state, shipment, route, action_cost=10.0)
        assert estimate.estimated_action_cost == 10.0
        assert estimate.estimated_total_cost == pytest.approx(estimate.estimated_transport_cost + 10.0)


class TestEstimateCurrentPlan:
    def test_unblocked_plan_matches_route_option(self) -> None:
        state = _tiny_state(day=1)
        shipment = _shipment(
            due_day=10,
            quantity=5,
            planned_route_edge_ids=("supplier_to_hub", "hub_to_plant"),
            next_edge_index=0,
        )
        plan_estimate = estimate_current_plan(state, shipment, known_shocks=())
        route = route_from_edge_ids(state, ("supplier_to_hub", "hub_to_plant"))
        direct_estimate = estimate_route_option(state, shipment, route, action_cost=0.0)
        assert plan_estimate == direct_estimate

    def test_blocked_plan_with_known_reopening(self) -> None:
        state = _tiny_state(day=2)
        state.edge_operational_state["supplier_to_hub"].available = False
        shock = Shock(
            shock_id="close_supplier_to_hub",
            shock_type=ShockType.EDGE_CLOSURE,
            target_type=TargetType.EDGE,
            target_id="supplier_to_hub",
            physical_start_day=1,
            physical_end_day=3,
            information_day=1,
        )
        shipment = _shipment(
            due_day=100,
            quantity=5,
            planned_route_edge_ids=("supplier_to_hub", "hub_to_plant"),
            next_edge_index=0,
        )
        estimate = estimate_current_plan(state, shipment, known_shocks=(shock,))

        assert estimate.estimated_arrival_day == 4 + 2  # reopens day 4, then 2 days of lead time
        assert estimate.estimated_transport_cost == pytest.approx(10.0)
        assert estimate.currently_dispatchable is False

    def test_blocked_plan_without_known_shock_is_all_none(self) -> None:
        state = _tiny_state(day=2)
        state.edge_operational_state["supplier_to_hub"].available = False
        shipment = _shipment(
            due_day=100,
            quantity=5,
            planned_route_edge_ids=("supplier_to_hub", "hub_to_plant"),
            next_edge_index=0,
        )
        estimate = estimate_current_plan(state, shipment, known_shocks=())

        assert estimate.estimated_arrival_day is None
        assert estimate.estimated_lead_time_days is None
        assert estimate.estimated_transport_cost is None
        assert estimate.estimated_total_cost is None
        assert estimate.currently_dispatchable is False

    def test_requires_shipment_at_node(self) -> None:
        # See the equivalent test on TestEnumerateCandidateRoutes for why this mutates
        # after construction rather than constructing the contradiction directly.
        state = _tiny_state()
        shipment = _shipment(planned_route_edge_ids=("supplier_to_hub", "hub_to_plant"))
        shipment.current_node_id = None
        with pytest.raises(RoutingError, match="AT_NODE"):
            estimate_current_plan(state, shipment, known_shocks=())

    def test_requires_remaining_route(self) -> None:
        state = _tiny_state()
        shipment = _shipment(
            planned_route_edge_ids=("supplier_to_hub", "hub_to_plant"),
            next_edge_index=2,
        )
        with pytest.raises(RoutingError, match="no remaining route"):
            estimate_current_plan(state, shipment, known_shocks=())


class TestBuildOperationalGraphParallelEdges:
    def test_duplicate_origin_destination_keeps_smallest_edge_id(self) -> None:
        nodes = {
            "n1": Node(
                node_id="n1",
                name="N1",
                node_type=NodeType.SUPPLIER,
                latitude=None,
                longitude=None,
                storage_capacity=100,
                processing_capacity=100,
                source_capacity=100,
            ),
            "n2": Node(
                node_id="n2",
                name="N2",
                node_type=NodeType.PLANT,
                latitude=None,
                longitude=None,
                storage_capacity=100,
                processing_capacity=100,
                source_capacity=0,
            ),
        }
        edges = {
            "z_edge": Edge(
                edge_id="z_edge",
                origin_node_id="n1",
                destination_node_id="n2",
                mode=TransportMode.ROAD,
                distance_km=1.0,
                base_lead_time_days=1,
                daily_capacity=10,
                unit_transport_cost=99.0,
                reliability=1.0,
                emergency=False,
            ),
            "a_edge": Edge(
                edge_id="a_edge",
                origin_node_id="n1",
                destination_node_id="n2",
                mode=TransportMode.ROAD,
                distance_km=1.0,
                base_lead_time_days=1,
                daily_capacity=10,
                unit_transport_cost=1.0,
                reliability=1.0,
                emergency=False,
            ),
        }
        products = {
            "widget": Product(
                product_id="widget",
                name="Widget",
                holding_cost_per_unit_day=0.0,
                backlog_cost_per_unit_day=0.0,
                late_penalty_per_unit_day=0.0,
            )
        }
        network_definition = NetworkDefinition(nodes=nodes, edges=edges, products=products)
        state = SimulationState(
            day=1,
            network_definition=network_definition,
            node_operational_state={nid: OperationalNodeState() for nid in nodes},
            edge_operational_state={eid: OperationalEdgeState() for eid in edges},
            inventory={nid: {"widget": 0} for nid in nodes},
            backlog={nid: {"widget": 0} for nid in nodes},
            shipments={},
        )
        graph = build_operational_graph(state)
        assert graph["n1"]["n2"]["edge_id"] == "a_edge"


class TestEnumerateCandidateRoutesEdgeCases:
    def test_returns_empty_when_destination_unavailable(self) -> None:
        state = _tiny_state()
        state.node_operational_state["plant_1"].available = False
        shipment = _shipment(quantity=5)
        estimates = enumerate_candidate_routes(
            state, shipment, reroute_cost_per_unit=1.0, expedite_premium_per_unit=2.0
        )
        assert estimates == ()

    def test_returns_empty_when_no_path_exists(self) -> None:
        nodes = {
            "n1": Node(
                node_id="n1",
                name="N1",
                node_type=NodeType.SUPPLIER,
                latitude=None,
                longitude=None,
                storage_capacity=100,
                processing_capacity=100,
                source_capacity=100,
            ),
            "n2": Node(
                node_id="n2",
                name="N2",
                node_type=NodeType.PLANT,
                latitude=None,
                longitude=None,
                storage_capacity=100,
                processing_capacity=100,
                source_capacity=0,
            ),
        }
        products = {
            "widget": Product(
                product_id="widget",
                name="Widget",
                holding_cost_per_unit_day=0.0,
                backlog_cost_per_unit_day=0.0,
                late_penalty_per_unit_day=0.0,
            )
        }
        network_definition = NetworkDefinition(nodes=nodes, edges={}, products=products)
        state = SimulationState(
            day=1,
            network_definition=network_definition,
            node_operational_state={nid: OperationalNodeState() for nid in nodes},
            edge_operational_state={},
            inventory={nid: {"widget": 0} for nid in nodes},
            backlog={nid: {"widget": 0} for nid in nodes},
            shipments={},
        )
        shipment = _shipment(current_node_id="n1", destination_node_id="n2", quantity=5)
        estimates = enumerate_candidate_routes(
            state, shipment, reroute_cost_per_unit=1.0, expedite_premium_per_unit=2.0
        )
        assert estimates == ()

    def test_path_longer_than_six_edges_excluded(self) -> None:
        chain_length = 7
        node_ids = [f"n{i}" for i in range(chain_length + 1)]
        nodes = {
            node_id: Node(
                node_id=node_id,
                name=node_id,
                node_type=NodeType.SUPPLIER if node_id == "n0" else NodeType.HUB,
                latitude=None,
                longitude=None,
                storage_capacity=100,
                processing_capacity=100,
                source_capacity=100 if node_id == "n0" else 0,
            )
            for node_id in node_ids
        }
        nodes[node_ids[-1]] = Node(
            node_id=node_ids[-1],
            name=node_ids[-1],
            node_type=NodeType.PLANT,
            latitude=None,
            longitude=None,
            storage_capacity=100,
            processing_capacity=100,
            source_capacity=0,
        )
        edges = {
            f"e{i}": Edge(
                edge_id=f"e{i}",
                origin_node_id=node_ids[i],
                destination_node_id=node_ids[i + 1],
                mode=TransportMode.ROAD,
                distance_km=1.0,
                base_lead_time_days=1,
                daily_capacity=100,
                unit_transport_cost=1.0,
                reliability=1.0,
                emergency=False,
            )
            for i in range(chain_length)
        }
        products = {
            "widget": Product(
                product_id="widget",
                name="Widget",
                holding_cost_per_unit_day=0.0,
                backlog_cost_per_unit_day=0.0,
                late_penalty_per_unit_day=0.0,
            )
        }
        network_definition = NetworkDefinition(nodes=nodes, edges=edges, products=products)
        state = SimulationState(
            day=1,
            network_definition=network_definition,
            node_operational_state={nid: OperationalNodeState() for nid in nodes},
            edge_operational_state={eid: OperationalEdgeState() for eid in edges},
            inventory={nid: {"widget": 0} for nid in nodes},
            backlog={nid: {"widget": 0} for nid in nodes},
            shipments={},
        )
        shipment = _shipment(current_node_id="n0", destination_node_id=node_ids[-1], quantity=5)
        estimates = enumerate_candidate_routes(
            state, shipment, reroute_cost_per_unit=1.0, expedite_premium_per_unit=2.0
        )
        assert estimates == ()
