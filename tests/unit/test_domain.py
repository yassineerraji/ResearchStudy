"""Unit tests that the domain package's dataclasses reject every malformed construction the contract forbids."""

from __future__ import annotations

import pytest

from supply_chain_simulator.domain.actions import (
    MAX_RATIONALE_LENGTH,
    ActionType,
    DecisionAction,
    ReasonCode,
    ValidationCode,
    ValidationResult,
)
from supply_chain_simulator.domain.events import (
    DemandEvent,
    ShipmentReleaseEvent,
    Shock,
    ShockType,
    TargetType,
)
from supply_chain_simulator.domain.models import (
    Edge,
    NetworkDefinition,
    Node,
    NodeType,
    Product,
    Route,
    TransportMode,
)
from supply_chain_simulator.domain.state import Shipment, ShipmentStatus


def _make_node(**overrides: object) -> Node:
    defaults: dict[str, object] = {
        "node_id": "n1",
        "name": "Node One",
        "node_type": NodeType.PLANT,
        "latitude": None,
        "longitude": None,
        "storage_capacity": 10,
        "processing_capacity": 10,
        "source_capacity": 0,
    }
    defaults.update(overrides)
    return Node(**defaults)  # type: ignore[arg-type]


def _make_edge(**overrides: object) -> Edge:
    defaults: dict[str, object] = {
        "edge_id": "e1",
        "origin_node_id": "n1",
        "destination_node_id": "n2",
        "mode": TransportMode.ROAD,
        "distance_km": 10.0,
        "base_lead_time_days": 1,
        "daily_capacity": 10,
        "unit_transport_cost": 1.0,
        "reliability": 0.9,
        "emergency": False,
    }
    defaults.update(overrides)
    return Edge(**defaults)  # type: ignore[arg-type]


class TestNode:
    def test_valid_node_constructs(self) -> None:
        node = _make_node()
        assert node.node_id == "n1"

    def test_negative_storage_capacity_rejected(self) -> None:
        with pytest.raises(ValueError, match="storage_capacity"):
            _make_node(storage_capacity=-1)

    def test_negative_processing_capacity_rejected(self) -> None:
        with pytest.raises(ValueError, match="processing_capacity"):
            _make_node(processing_capacity=-1)

    def test_negative_source_capacity_rejected(self) -> None:
        with pytest.raises(ValueError, match="source_capacity"):
            _make_node(source_capacity=-1)

    def test_non_supplier_with_source_capacity_rejected(self) -> None:
        with pytest.raises(ValueError, match="source_capacity must be 0"):
            _make_node(node_type=NodeType.PLANT, source_capacity=5)

    def test_supplier_with_source_capacity_allowed(self) -> None:
        node = _make_node(node_type=NodeType.SUPPLIER, source_capacity=5)
        assert node.source_capacity == 5


class TestEdge:
    def test_valid_edge_constructs(self) -> None:
        edge = _make_edge()
        assert edge.edge_id == "e1"

    def test_same_origin_and_destination_rejected(self) -> None:
        with pytest.raises(ValueError, match="origin and destination must differ"):
            _make_edge(origin_node_id="n1", destination_node_id="n1")

    def test_lead_time_below_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="base_lead_time_days"):
            _make_edge(base_lead_time_days=0)

    def test_negative_capacity_rejected(self) -> None:
        with pytest.raises(ValueError, match="daily_capacity"):
            _make_edge(daily_capacity=-1)

    def test_negative_cost_rejected(self) -> None:
        with pytest.raises(ValueError, match="unit_transport_cost"):
            _make_edge(unit_transport_cost=-1.0)

    def test_negative_distance_rejected(self) -> None:
        with pytest.raises(ValueError, match="distance_km"):
            _make_edge(distance_km=-1.0)

    @pytest.mark.parametrize("reliability", [-0.1, 1.1])
    def test_reliability_out_of_range_rejected(self, reliability: float) -> None:
        with pytest.raises(ValueError, match="reliability"):
            _make_edge(reliability=reliability)

    @pytest.mark.parametrize("reliability", [0.0, 1.0])
    def test_reliability_boundary_allowed(self, reliability: float) -> None:
        edge = _make_edge(reliability=reliability)
        assert edge.reliability == reliability


class TestProduct:
    def test_negative_holding_cost_rejected(self) -> None:
        with pytest.raises(ValueError, match="holding_cost_per_unit_day"):
            Product(
                product_id="p1",
                name="Widget",
                holding_cost_per_unit_day=-1.0,
                backlog_cost_per_unit_day=1.0,
                late_penalty_per_unit_day=1.0,
            )

    def test_negative_backlog_cost_rejected(self) -> None:
        with pytest.raises(ValueError, match="backlog_cost_per_unit_day"):
            Product(
                product_id="p1",
                name="Widget",
                holding_cost_per_unit_day=1.0,
                backlog_cost_per_unit_day=-1.0,
                late_penalty_per_unit_day=1.0,
            )

    def test_negative_late_penalty_rejected(self) -> None:
        with pytest.raises(ValueError, match="late_penalty_per_unit_day"):
            Product(
                product_id="p1",
                name="Widget",
                holding_cost_per_unit_day=1.0,
                backlog_cost_per_unit_day=1.0,
                late_penalty_per_unit_day=-1.0,
            )


class TestNetworkDefinition:
    def _valid_network(self) -> NetworkDefinition:
        node1 = _make_node(node_id="n1", node_type=NodeType.SUPPLIER, source_capacity=5)
        node2 = _make_node(node_id="n2")
        edge = _make_edge(edge_id="e1", origin_node_id="n1", destination_node_id="n2")
        product = Product(
            product_id="p1",
            name="Widget",
            holding_cost_per_unit_day=0.1,
            backlog_cost_per_unit_day=1.0,
            late_penalty_per_unit_day=1.0,
        )
        return NetworkDefinition(
            nodes={"n1": node1, "n2": node2}, edges={"e1": edge}, products={"p1": product}
        )

    def test_valid_network_constructs_and_looks_up(self) -> None:
        network = self._valid_network()
        assert network.get_node("n1").node_id == "n1"
        assert network.get_edge("e1").edge_id == "e1"
        assert network.get_product("p1").product_id == "p1"

    def test_unknown_node_lookup_raises_key_error(self) -> None:
        network = self._valid_network()
        with pytest.raises(KeyError):
            network.get_node("missing")

    def test_unknown_edge_lookup_raises_key_error(self) -> None:
        network = self._valid_network()
        with pytest.raises(KeyError):
            network.get_edge("missing")

    def test_unknown_product_lookup_raises_key_error(self) -> None:
        network = self._valid_network()
        with pytest.raises(KeyError):
            network.get_product("missing")

    def test_mismatched_node_key_rejected(self) -> None:
        node = _make_node(node_id="n1")
        with pytest.raises(ValueError, match="does not match node_id"):
            NetworkDefinition(nodes={"wrong_key": node}, edges={}, products={})

    def test_mismatched_edge_key_rejected(self) -> None:
        node1 = _make_node(node_id="n1", node_type=NodeType.SUPPLIER, source_capacity=5)
        node2 = _make_node(node_id="n2")
        edge = _make_edge(edge_id="e1", origin_node_id="n1", destination_node_id="n2")
        with pytest.raises(ValueError, match="does not match edge_id"):
            NetworkDefinition(
                nodes={"n1": node1, "n2": node2}, edges={"wrong_key": edge}, products={}
            )

    def test_mismatched_product_key_rejected(self) -> None:
        product = Product(
            product_id="p1",
            name="Widget",
            holding_cost_per_unit_day=0.1,
            backlog_cost_per_unit_day=1.0,
            late_penalty_per_unit_day=1.0,
        )
        with pytest.raises(ValueError, match="does not match product_id"):
            NetworkDefinition(nodes={}, edges={}, products={"wrong_key": product})

    def test_edge_referencing_unknown_origin_rejected(self) -> None:
        node = _make_node(node_id="n1")
        edge = _make_edge(edge_id="e1", origin_node_id="missing", destination_node_id="n1")
        with pytest.raises(ValueError, match="unknown origin_node_id"):
            NetworkDefinition(nodes={"n1": node}, edges={"e1": edge}, products={})

    def test_edge_referencing_unknown_destination_rejected(self) -> None:
        node = _make_node(node_id="n1")
        edge = _make_edge(edge_id="e1", origin_node_id="n1", destination_node_id="missing")
        with pytest.raises(ValueError, match="unknown destination_node_id"):
            NetworkDefinition(nodes={"n1": node}, edges={"e1": edge}, products={})


class TestRoute:
    def test_route_constructs(self) -> None:
        route = Route(
            route_id="e1__e2",
            edge_ids=("e1", "e2"),
            node_ids=("n1", "n2", "n3"),
            contains_emergency_edge=False,
        )
        assert route.route_id == "e1__e2"


class TestShipment:
    def _base_kwargs(self) -> dict[str, object]:
        return {
            "shipment_id": "shipment_001_001",
            "product_id": "p1",
            "quantity": 5,
            "origin_node_id": "n1",
            "destination_node_id": "n3",
            "release_day": 1,
            "due_day": 10,
            "planned_route_edge_ids": ("e1", "e2"),
            "next_edge_index": 0,
            "reroute_count": 0,
            "expedite_count": 0,
            "capacity_wait_days": 0,
            "delivered_day": None,
        }

    def test_at_node_requires_current_node_and_no_edge(self) -> None:
        kwargs = self._base_kwargs()
        shipment = Shipment(
            status=ShipmentStatus.AT_NODE,
            current_node_id="n1",
            current_edge_id=None,
            edge_entry_day=None,
            edge_arrival_day=None,
            **kwargs,  # type: ignore[arg-type]
        )
        assert shipment.status is ShipmentStatus.AT_NODE

    def test_at_node_without_current_node_rejected(self) -> None:
        kwargs = self._base_kwargs()
        with pytest.raises(ValueError, match="AT_NODE"):
            Shipment(
                status=ShipmentStatus.AT_NODE,
                current_node_id=None,
                current_edge_id=None,
                edge_entry_day=None,
                edge_arrival_day=None,
                **kwargs,  # type: ignore[arg-type]
            )

    def test_in_transit_requires_edge_and_arrival_day(self) -> None:
        kwargs = self._base_kwargs()
        shipment = Shipment(
            status=ShipmentStatus.IN_TRANSIT,
            current_node_id=None,
            current_edge_id="e1",
            edge_entry_day=1,
            edge_arrival_day=2,
            **kwargs,  # type: ignore[arg-type]
        )
        assert shipment.status is ShipmentStatus.IN_TRANSIT

    def test_in_transit_without_arrival_day_rejected(self) -> None:
        kwargs = self._base_kwargs()
        with pytest.raises(ValueError, match="IN_TRANSIT"):
            Shipment(
                status=ShipmentStatus.IN_TRANSIT,
                current_node_id=None,
                current_edge_id="e1",
                edge_entry_day=1,
                edge_arrival_day=None,
                **kwargs,  # type: ignore[arg-type]
            )

    def test_delivered_requires_current_node_equal_destination(self) -> None:
        kwargs = self._base_kwargs()
        shipment = Shipment(
            status=ShipmentStatus.DELIVERED,
            current_node_id="n3",
            current_edge_id=None,
            edge_entry_day=None,
            edge_arrival_day=None,
            delivered_day=5,
            **{k: v for k, v in kwargs.items() if k != "delivered_day"},  # type: ignore[arg-type]
        )
        assert shipment.status is ShipmentStatus.DELIVERED

    def test_delivered_at_wrong_node_rejected(self) -> None:
        kwargs = self._base_kwargs()
        with pytest.raises(ValueError, match="DELIVERED"):
            Shipment(
                status=ShipmentStatus.DELIVERED,
                current_node_id="n1",
                current_edge_id=None,
                edge_entry_day=None,
                edge_arrival_day=None,
                delivered_day=5,
                **{k: v for k, v in kwargs.items() if k != "delivered_day"},  # type: ignore[arg-type]
            )

    def test_non_positive_quantity_rejected(self) -> None:
        kwargs = self._base_kwargs()
        kwargs["quantity"] = 0
        with pytest.raises(ValueError, match="quantity"):
            Shipment(
                status=ShipmentStatus.AT_NODE,
                current_node_id="n1",
                current_edge_id=None,
                edge_entry_day=None,
                edge_arrival_day=None,
                **kwargs,  # type: ignore[arg-type]
            )


class TestDemandEvent:
    def test_negative_quantity_rejected(self) -> None:
        with pytest.raises(ValueError, match="quantity must be non-negative"):
            DemandEvent(day=1, destination_node_id="n1", product_id="p1", quantity=-1)

    def test_zero_quantity_allowed(self) -> None:
        event = DemandEvent(day=1, destination_node_id="n1", product_id="p1", quantity=0)
        assert event.quantity == 0


class TestShipmentReleaseEvent:
    def test_non_positive_quantity_rejected(self) -> None:
        with pytest.raises(ValueError, match="quantity must be positive"):
            ShipmentReleaseEvent(
                day=1,
                shipment_id="shipment_001_001",
                product_id="p1",
                quantity=0,
                origin_node_id="n1",
                destination_node_id="n2",
                due_day=10,
                initial_route_edge_ids=("e1",),
            )

    def test_empty_route_rejected(self) -> None:
        with pytest.raises(ValueError, match="initial_route_edge_ids must not be empty"):
            ShipmentReleaseEvent(
                day=1,
                shipment_id="shipment_001_001",
                product_id="p1",
                quantity=5,
                origin_node_id="n1",
                destination_node_id="n2",
                due_day=10,
                initial_route_edge_ids=(),
            )


class TestDecisionAction:
    def test_wait_with_route_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="requires route_id=None"):
            DecisionAction(
                shipment_id="s1",
                action_type=ActionType.WAIT,
                route_id="r1",
                reason_code=ReasonCode.REDUCE_LATENESS,
                rationale="",
            )

    def test_reroute_without_route_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="requires a route_id"):
            DecisionAction(
                shipment_id="s1",
                action_type=ActionType.REROUTE,
                route_id=None,
                reason_code=ReasonCode.LOWER_ESTIMATED_COST,
                rationale="",
            )

    def test_rationale_too_long_rejected(self) -> None:
        with pytest.raises(ValueError, match="rationale exceeds"):
            DecisionAction(
                shipment_id="s1",
                action_type=ActionType.WAIT,
                route_id=None,
                reason_code=ReasonCode.REDUCE_LATENESS,
                rationale="x" * (MAX_RATIONALE_LENGTH + 1),
            )

    def test_valid_wait_action_constructs(self) -> None:
        action = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.WAIT,
            route_id=None,
            reason_code=ReasonCode.REDUCE_LATENESS,
            rationale="waiting for reopening",
        )
        assert action.action_type is ActionType.WAIT


class TestValidationResult:
    def test_valid_code_is_valid(self) -> None:
        result = ValidationResult(code=ValidationCode.VALID, detail="")
        assert result.is_valid is True

    def test_other_code_is_not_valid(self) -> None:
        result = ValidationResult(code=ValidationCode.ROUTE_NOT_FOUND, detail="no such route")
        assert result.is_valid is False


class TestShock:
    def test_end_before_start_rejected(self) -> None:
        with pytest.raises(ValueError, match="physical_end_day"):
            Shock(
                shock_id="shock1",
                shock_type=ShockType.NODE_CLOSURE,
                target_type=TargetType.NODE,
                target_id="n1",
                physical_start_day=5,
                physical_end_day=4,
                information_day=5,
            )

    def test_valid_shock_constructs(self) -> None:
        shock = Shock(
            shock_id="shock1",
            shock_type=ShockType.NODE_CLOSURE,
            target_type=TargetType.NODE,
            target_id="n1",
            physical_start_day=5,
            physical_end_day=10,
            information_day=5,
        )
        assert shock.capacity_multiplier == 1.0
