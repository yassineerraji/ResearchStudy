"""Unit tests for every cost formula in simulation/costs.py against independently hand-calculated values."""

from __future__ import annotations

import pytest

from supply_chain_simulator.domain.models import (
    NetworkDefinition,
    Node,
    NodeType,
    Product,
)
from supply_chain_simulator.domain.state import (
    OperationalNodeState,
    Shipment,
    ShipmentStatus,
    SimulationState,
)
from supply_chain_simulator.simulation.costs import (
    charge_delivery_late_penalty,
    charge_edge_entry_transport_cost,
    charge_end_of_day_backlog_cost,
    charge_end_of_day_holding_cost,
    charge_expedite_cost,
    charge_reroute_cost,
    charge_terminal_cost,
    total_cost,
)

P1 = Product(
    product_id="p1",
    name="P1",
    holding_cost_per_unit_day=0.10,
    backlog_cost_per_unit_day=1.00,
    late_penalty_per_unit_day=0.50,
)
P2 = Product(
    product_id="p2",
    name="P2",
    holding_cost_per_unit_day=0.20,
    backlog_cost_per_unit_day=2.00,
    late_penalty_per_unit_day=1.00,
)


def _state(day: int = 1) -> SimulationState:
    node_ids = ["n1", "n2"]
    nodes = {
        node_id: Node(
            node_id=node_id,
            name=node_id,
            node_type=NodeType.PLANT,
            latitude=None,
            longitude=None,
            storage_capacity=1000,
            processing_capacity=1000,
            source_capacity=0,
        )
        for node_id in node_ids
    }
    network_definition = NetworkDefinition(
        nodes=nodes, edges={}, products={"p1": P1, "p2": P2}
    )
    return SimulationState(
        day=day,
        network_definition=network_definition,
        node_operational_state={node_id: OperationalNodeState() for node_id in node_ids},
        edge_operational_state={},
        inventory={"n1": {"p1": 0, "p2": 0}, "n2": {"p1": 0, "p2": 0}},
        backlog={"n1": {"p1": 0, "p2": 0}, "n2": {"p1": 0, "p2": 0}},
        shipments={},
    )


def _shipment(
    status: ShipmentStatus,
    product_id: str = "p1",
    quantity: int = 5,
    due_day: int = 10,
    delivered_day: int | None = None,
    destination_node_id: str = "n2",
    current_node_id: str | None = "n1",
) -> Shipment:
    return Shipment(
        shipment_id="shipment_001_001",
        product_id=product_id,
        quantity=quantity,
        origin_node_id="n1",
        destination_node_id=destination_node_id,
        release_day=1,
        due_day=due_day,
        planned_route_edge_ids=(),
        next_edge_index=0,
        status=status,
        current_node_id=current_node_id,
        current_edge_id=None,
        edge_entry_day=None,
        edge_arrival_day=None,
        reroute_count=0,
        expedite_count=0,
        capacity_wait_days=0,
        delivered_day=delivered_day,
    )


class TestChargeEdgeEntryTransportCost:
    def test_matches_hand_calculation_and_accumulates(self) -> None:
        state = _state()
        first = charge_edge_entry_transport_cost(state, quantity=5, effective_unit_transport_cost=4.0)
        assert first == pytest.approx(20.0)
        assert state.costs.transport == pytest.approx(20.0)

        second = charge_edge_entry_transport_cost(state, quantity=3, effective_unit_transport_cost=2.0)
        assert second == pytest.approx(6.0)
        assert state.costs.transport == pytest.approx(26.0)


class TestChargeRerouteCost:
    def test_matches_hand_calculation(self) -> None:
        state = _state()
        cost = charge_reroute_cost(state, quantity=5, reroute_cost_per_unit=2.0)
        assert cost == pytest.approx(10.0)
        assert state.costs.reroute == pytest.approx(10.0)


class TestChargeExpediteCost:
    def test_matches_hand_calculation(self) -> None:
        state = _state()
        cost = charge_expedite_cost(state, quantity=5, expedite_premium_per_unit=10.0)
        assert cost == pytest.approx(50.0)
        assert state.costs.expedite == pytest.approx(50.0)


class TestChargeEndOfDayHoldingCost:
    def test_matches_hand_calculation_across_nodes_and_products(self) -> None:
        state = _state()
        state.inventory = {"n1": {"p1": 10, "p2": 5}, "n2": {"p1": 3, "p2": 0}}
        # 10*0.10 + 5*0.20 + 3*0.10 + 0*0.20 = 1.0 + 1.0 + 0.3 + 0.0 = 2.3
        cost = charge_end_of_day_holding_cost(state)
        assert cost == pytest.approx(2.3)
        assert state.costs.holding == pytest.approx(2.3)

        # Charging again (e.g. a second day with the same inventory) accumulates.
        cost_again = charge_end_of_day_holding_cost(state)
        assert cost_again == pytest.approx(2.3)
        assert state.costs.holding == pytest.approx(4.6)


class TestChargeEndOfDayBacklogCost:
    def test_matches_hand_calculation_across_nodes_and_products(self) -> None:
        state = _state()
        state.backlog = {"n1": {"p1": 4, "p2": 0}, "n2": {"p1": 0, "p2": 2}}
        # 4*1.00 + 0*2.00 + 0*1.00 + 2*2.00 = 4.0 + 0.0 + 0.0 + 4.0 = 8.0
        cost = charge_end_of_day_backlog_cost(state)
        assert cost == pytest.approx(8.0)
        assert state.costs.backlog == pytest.approx(8.0)


class TestChargeDeliveryLatePenalty:
    def test_late_delivery_matches_hand_calculation(self) -> None:
        state = _state()
        shipment = _shipment(
            status=ShipmentStatus.DELIVERED,
            product_id="p1",
            quantity=5,
            due_day=10,
            delivered_day=12,
            destination_node_id="n1",
            current_node_id="n1",
        )
        # lateness = max(0, 12 - 10) = 2; cost = 5 * 2 * 0.50 = 5.0
        cost = charge_delivery_late_penalty(state, shipment)
        assert cost == pytest.approx(5.0)
        assert state.costs.late == pytest.approx(5.0)

    def test_on_time_delivery_charges_nothing(self) -> None:
        state = _state()
        shipment = _shipment(
            status=ShipmentStatus.DELIVERED,
            due_day=10,
            delivered_day=8,
            destination_node_id="n1",
            current_node_id="n1",
        )
        cost = charge_delivery_late_penalty(state, shipment)
        assert cost == pytest.approx(0.0)

    def test_missing_delivered_day_rejected(self) -> None:
        state = _state()
        shipment = _shipment(status=ShipmentStatus.AT_NODE, delivered_day=None)
        with pytest.raises(ValueError, match="delivered_day"):
            charge_delivery_late_penalty(state, shipment)


class TestChargeTerminalCost:
    def test_matches_hand_calculation(self) -> None:
        state = _state(day=90)
        state.backlog = {"n1": {"p1": 4, "p2": 0}, "n2": {"p1": 0, "p2": 0}}
        # terminal_backlog = 4 * 1.00 * 30 = 120.0
        undelivered = _shipment(
            status=ShipmentStatus.AT_NODE,
            product_id="p1",
            quantity=5,
            due_day=50,
            current_node_id="n1",
        )
        # terminal_shipment = 5 * 0.50 * max(1, 90 - 50) = 5 * 0.50 * 40 = 100.0
        delivered = _shipment(
            status=ShipmentStatus.DELIVERED,
            product_id="p1",
            quantity=99,
            due_day=1,
            delivered_day=2,
            destination_node_id="n1",
            current_node_id="n1",
        )
        state.shipments = {"undelivered": undelivered, "delivered": delivered}

        cost = charge_terminal_cost(state, terminal_penalty_days=30)
        assert cost == pytest.approx(220.0)
        assert state.costs.terminal == pytest.approx(220.0)

    def test_undelivered_shipment_due_tomorrow_is_at_least_one_day_late(self) -> None:
        state = _state(day=10)
        undelivered = _shipment(
            status=ShipmentStatus.AT_NODE,
            product_id="p1",
            quantity=2,
            due_day=10,
            current_node_id="n1",
        )
        state.shipments = {"undelivered": undelivered}
        # terminal_shipment = 2 * 0.50 * max(1, 10 - 10) = 2 * 0.50 * 1 = 1.0
        cost = charge_terminal_cost(state, terminal_penalty_days=30)
        assert cost == pytest.approx(1.0)


class TestTotalCost:
    def test_sums_every_component(self) -> None:
        state = _state()
        state.costs.transport = 1.0
        state.costs.reroute = 2.0
        state.costs.expedite = 3.0
        state.costs.holding = 4.0
        state.costs.backlog = 5.0
        state.costs.late = 6.0
        state.costs.terminal = 7.0
        assert total_cost(state) == pytest.approx(28.0)
