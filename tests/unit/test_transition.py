"""Unit tests for simulation/transition.py: each daily step, in isolation.

Inside tests/unit, this file checks shipment release, edge entry and
calculated arrival day, intermediate arrival, final delivery and inventory
addition, node closure blocking arrival, capacity contention incrementing
wait days, backlog-first demand fulfilment, decision-trigger detection, and
applying already-validated actions — each function from CLAUDE.md section
11.8 tested against its own exact rule from section 14. The full thirteen-
step daily order and end-to-end state invariants are covered together by
tests/integration/test_full_simulation.py, which drives these same functions
through simulation/engine.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from supply_chain_simulator.data_io.loaders import (
    build_initial_state,
    build_network_definition,
    load_network_config,
)
from supply_chain_simulator.domain.actions import ActionType, DecisionAction, ReasonCode
from supply_chain_simulator.domain.events import (
    DemandEvent,
    ShipmentReleaseEvent,
    Shock,
    ShockType,
    TargetType,
)
from supply_chain_simulator.domain.models import Route
from supply_chain_simulator.domain.state import (
    Shipment,
    ShipmentStatus,
    SimulationState,
)
from supply_chain_simulator.simulation.transition import (
    SimulationInvariantError,
    allocate_departures,
    apply_shock_operational_state,
    apply_validated_actions,
    charge_end_of_day_costs,
    fulfil_backlog_and_demand,
    identify_shipments_requiring_decision,
    process_due_arrivals,
    record_daily_metrics,
    release_shipments,
    reset_daily_capacity_usage,
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
    shipment_id: str = "shipment_001_001",
    status: ShipmentStatus = ShipmentStatus.AT_NODE,
    current_node_id: str | None = "supplier_1",
    current_edge_id: str | None = None,
    edge_entry_day: int | None = None,
    edge_arrival_day: int | None = None,
    destination_node_id: str = "plant_1",
    planned_route_edge_ids: tuple[str, ...] = ("supplier_to_hub", "hub_to_plant"),
    next_edge_index: int = 0,
    quantity: int = 5,
    due_day: int = 100,
    capacity_wait_days: int = 0,
    delivered_day: int | None = None,
) -> Shipment:
    return Shipment(
        shipment_id=shipment_id,
        product_id="widget",
        quantity=quantity,
        origin_node_id="supplier_1",
        destination_node_id=destination_node_id,
        release_day=1,
        due_day=due_day,
        planned_route_edge_ids=planned_route_edge_ids,
        next_edge_index=next_edge_index,
        status=status,
        current_node_id=current_node_id,
        current_edge_id=current_edge_id,
        edge_entry_day=edge_entry_day,
        edge_arrival_day=edge_arrival_day,
        reroute_count=0,
        expedite_count=0,
        capacity_wait_days=capacity_wait_days,
        delivered_day=delivered_day,
    )


class TestResetDailyCapacityUsage:
    def test_resets_all_counters_to_zero(self) -> None:
        state = _tiny_state()
        state.daily_edge_used_capacity["supplier_to_hub"] = 15
        state.daily_node_used_processing["supplier_1"] = 30
        reset_daily_capacity_usage(state)
        assert state.daily_edge_used_capacity["supplier_to_hub"] == 0
        assert state.daily_node_used_processing["supplier_1"] == 0


class TestApplyShockOperationalState:
    def _shock(self, **overrides: object) -> Shock:
        defaults: dict[str, object] = {
            "shock_id": "s1",
            "shock_type": ShockType.EDGE_CLOSURE,
            "target_type": TargetType.EDGE,
            "target_id": "supplier_to_hub",
            "physical_start_day": 3,
            "physical_end_day": 5,
            "information_day": 3,
        }
        defaults.update(overrides)
        return Shock(**defaults)  # type: ignore[arg-type]

    def test_no_active_shocks_leaves_defaults(self) -> None:
        state = _tiny_state(day=1)
        apply_shock_operational_state(state, (self._shock(),))
        assert state.edge_operational_state["supplier_to_hub"].available is True
        assert state.active_shock_ids == set()

    def test_edge_closure_marks_unavailable_within_range(self) -> None:
        state = _tiny_state(day=4)
        apply_shock_operational_state(state, (self._shock(),))
        assert state.edge_operational_state["supplier_to_hub"].available is False
        assert state.active_shock_ids == {"s1"}

    def test_shock_outside_range_has_no_effect(self) -> None:
        state = _tiny_state(day=6)
        apply_shock_operational_state(state, (self._shock(),))
        assert state.edge_operational_state["supplier_to_hub"].available is True
        assert state.active_shock_ids == set()

    def test_node_closure(self) -> None:
        state = _tiny_state(day=3)
        shock = self._shock(
            shock_id="n1", shock_type=ShockType.NODE_CLOSURE, target_type=TargetType.NODE, target_id="hub_1"
        )
        apply_shock_operational_state(state, (shock,))
        assert state.node_operational_state["hub_1"].available is False

    def test_capacity_reduction_multiplies(self) -> None:
        state = _tiny_state(day=3)
        shock = self._shock(
            shock_id="c1",
            shock_type=ShockType.EDGE_CAPACITY_REDUCTION,
            capacity_multiplier=0.5,
        )
        apply_shock_operational_state(state, (shock,))
        assert state.edge_operational_state["supplier_to_hub"].capacity_multiplier == pytest.approx(0.5)

    def test_overlapping_same_target_multipliers_combine(self) -> None:
        state = _tiny_state(day=3)
        shock_a = self._shock(shock_id="a", shock_type=ShockType.EDGE_CAPACITY_REDUCTION, capacity_multiplier=0.5)
        shock_b = self._shock(shock_id="b", shock_type=ShockType.EDGE_CAPACITY_REDUCTION, capacity_multiplier=0.5)
        apply_shock_operational_state(state, (shock_a, shock_b))
        assert state.edge_operational_state["supplier_to_hub"].capacity_multiplier == pytest.approx(0.25)

    def test_node_capacity_reduction_multiplies(self) -> None:
        state = _tiny_state(day=3)
        shock = self._shock(
            shock_id="n1",
            shock_type=ShockType.NODE_CAPACITY_REDUCTION,
            target_type=TargetType.NODE,
            target_id="supplier_1",
            capacity_multiplier=0.4,
        )
        apply_shock_operational_state(state, (shock,))
        assert state.node_operational_state[
            "supplier_1"
        ].processing_capacity_multiplier == pytest.approx(0.4)

    def test_edge_cost_increase_multiplies(self) -> None:
        state = _tiny_state(day=3)
        shock = self._shock(shock_id="e1", shock_type=ShockType.EDGE_COST_INCREASE, cost_multiplier=3.0)
        apply_shock_operational_state(state, (shock,))
        assert state.edge_operational_state["supplier_to_hub"].cost_multiplier == pytest.approx(3.0)

    def test_rebuilds_from_scratch_each_day(self) -> None:
        state = _tiny_state(day=3)
        apply_shock_operational_state(state, (self._shock(),))
        assert state.edge_operational_state["supplier_to_hub"].available is False
        state.day = 6
        apply_shock_operational_state(state, (self._shock(),))
        assert state.edge_operational_state["supplier_to_hub"].available is True


class TestProcessDueArrivals:
    def test_intermediate_arrival_advances_to_at_node(self) -> None:
        state = _tiny_state(day=2)
        state.shipments["s1"] = _shipment(
            shipment_id="s1",
            status=ShipmentStatus.IN_TRANSIT,
            current_node_id=None,
            current_edge_id="supplier_to_hub",
            edge_entry_day=1,
            edge_arrival_day=2,
            next_edge_index=0,
        )
        process_due_arrivals(state)
        shipment = state.shipments["s1"]
        assert shipment.status is ShipmentStatus.AT_NODE
        assert shipment.current_node_id == "hub_1"
        assert shipment.next_edge_index == 1

    def test_final_arrival_delivers_and_adds_inventory(self) -> None:
        state = _tiny_state(day=2)
        state.shipments["s1"] = _shipment(
            shipment_id="s1",
            status=ShipmentStatus.IN_TRANSIT,
            current_node_id=None,
            current_edge_id="hub_to_plant",
            edge_entry_day=1,
            edge_arrival_day=2,
            planned_route_edge_ids=("supplier_to_hub", "hub_to_plant"),
            next_edge_index=1,
            quantity=5,
            due_day=10,
        )
        before_inventory = state.inventory["plant_1"]["widget"]
        process_due_arrivals(state)
        shipment = state.shipments["s1"]
        assert shipment.status is ShipmentStatus.DELIVERED
        assert shipment.delivered_day == 2
        assert state.inventory["plant_1"]["widget"] == before_inventory + 5
        assert state.service.delivered_shipment_units == 5

    def test_late_final_arrival_charges_penalty_and_counters(self) -> None:
        state = _tiny_state(day=12)
        state.shipments["s1"] = _shipment(
            shipment_id="s1",
            status=ShipmentStatus.IN_TRANSIT,
            current_node_id=None,
            current_edge_id="hub_to_plant",
            edge_entry_day=11,
            edge_arrival_day=12,
            next_edge_index=1,
            quantity=5,
            due_day=10,
        )
        process_due_arrivals(state)
        assert state.service.late_delivered_units == 5
        assert state.service.total_lateness_unit_days == 5 * 2
        assert state.costs.late == pytest.approx(5 * 2 * 0.5)

    def test_node_closure_postpones_arrival(self) -> None:
        state = _tiny_state(day=2)
        state.node_operational_state["hub_1"].available = False
        state.shipments["s1"] = _shipment(
            shipment_id="s1",
            status=ShipmentStatus.IN_TRANSIT,
            current_node_id=None,
            current_edge_id="supplier_to_hub",
            edge_entry_day=1,
            edge_arrival_day=2,
        )
        process_due_arrivals(state)
        shipment = state.shipments["s1"]
        assert shipment.status is ShipmentStatus.IN_TRANSIT
        assert shipment.edge_arrival_day == 3

    def test_storage_overflow_postpones_arrival(self) -> None:
        state = _tiny_state(day=2)
        state.inventory["hub_1"]["widget"] = 99  # hub_1 storage_capacity is 100
        state.shipments["s1"] = _shipment(
            shipment_id="s1",
            status=ShipmentStatus.IN_TRANSIT,
            current_node_id=None,
            current_edge_id="supplier_to_hub",
            edge_entry_day=1,
            edge_arrival_day=2,
            quantity=5,
        )
        process_due_arrivals(state)
        shipment = state.shipments["s1"]
        assert shipment.status is ShipmentStatus.IN_TRANSIT
        assert shipment.edge_arrival_day == 3

    def test_not_due_today_is_untouched(self) -> None:
        state = _tiny_state(day=2)
        state.shipments["s1"] = _shipment(
            shipment_id="s1",
            status=ShipmentStatus.IN_TRANSIT,
            current_node_id=None,
            current_edge_id="supplier_to_hub",
            edge_entry_day=1,
            edge_arrival_day=5,
        )
        process_due_arrivals(state)
        assert state.shipments["s1"].status is ShipmentStatus.IN_TRANSIT
        assert state.shipments["s1"].edge_arrival_day == 5


class TestReleaseShipments:
    def _event(self, **overrides: object) -> ShipmentReleaseEvent:
        defaults: dict[str, object] = {
            "day": 1,
            "shipment_id": "shipment_001_001",
            "product_id": "widget",
            "quantity": 5,
            "origin_node_id": "supplier_1",
            "destination_node_id": "plant_1",
            "due_day": 6,
            "initial_route_edge_ids": ("supplier_to_hub", "hub_to_plant"),
        }
        defaults.update(overrides)
        return ShipmentReleaseEvent(**defaults)  # type: ignore[arg-type]

    def test_valid_release_creates_at_node_shipment(self) -> None:
        state = _tiny_state(day=1)
        release_shipments(state, (self._event(),))
        shipment = state.shipments["shipment_001_001"]
        assert shipment.status is ShipmentStatus.AT_NODE
        assert shipment.current_node_id == "supplier_1"
        assert shipment.quantity == 5

    def test_unavailable_source_rejected(self) -> None:
        state = _tiny_state(day=1)
        state.node_operational_state["supplier_1"].available = False
        with pytest.raises(SimulationInvariantError, match="unavailable"):
            release_shipments(state, (self._event(),))

    def test_quantity_over_source_capacity_rejected(self) -> None:
        state = _tiny_state(day=1)
        with pytest.raises(SimulationInvariantError, match="source_capacity"):
            release_shipments(state, (self._event(quantity=999),))

    def test_storage_overflow_at_source_rejected(self) -> None:
        state = _tiny_state(day=1)
        state.inventory["supplier_1"]["widget"] = 99  # supplier_1 storage_capacity is 100
        with pytest.raises(SimulationInvariantError, match="storage_capacity"):
            release_shipments(state, (self._event(quantity=5),))


class TestFulfilBacklogAndDemand:
    def _event(self, quantity: int) -> DemandEvent:
        return DemandEvent(
            day=1, destination_node_id="plant_1", product_id="widget", quantity=quantity
        )

    def test_clears_old_backlog_before_same_day_demand(self) -> None:
        state = _tiny_state()
        state.inventory["plant_1"]["widget"] = 6
        state.backlog["plant_1"]["widget"] = 4
        result = fulfil_backlog_and_demand(state, (self._event(5),))

        # 4 clears backlog first (6 - 4 = 2 remain); then 2 covers same-day demand,
        # leaving 3 unmet -> new backlog of 3.
        assert result.backlog_fulfilled_units == 4
        assert result.same_day_fulfilled_units == 2
        assert state.inventory["plant_1"]["widget"] == 0
        assert state.backlog["plant_1"]["widget"] == 3
        assert result.demand_units == 5

    def test_full_same_day_fulfilment_with_no_backlog(self) -> None:
        state = _tiny_state()
        state.inventory["plant_1"]["widget"] = 10
        result = fulfil_backlog_and_demand(state, (self._event(5),))
        assert result.same_day_fulfilled_units == 5
        assert result.backlog_fulfilled_units == 0
        assert state.inventory["plant_1"]["widget"] == 5
        assert state.backlog["plant_1"]["widget"] == 0

    def test_unmet_demand_becomes_backlog(self) -> None:
        state = _tiny_state()
        state.inventory["plant_1"]["widget"] = 0
        result = fulfil_backlog_and_demand(state, (self._event(7),))
        assert result.same_day_fulfilled_units == 0
        assert state.backlog["plant_1"]["widget"] == 7
        assert state.service.total_demand_units == 7


class TestIdentifyShipmentsRequiringDecision:
    def test_no_trigger_not_included(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1", due_day=100)
        triggered = identify_shipments_requiring_decision(state, known_shocks=())
        assert triggered == ()

    def test_unavailable_next_edge_triggers(self) -> None:
        state = _tiny_state(day=1)
        state.edge_operational_state["supplier_to_hub"].available = False
        state.shipments["s1"] = _shipment(shipment_id="s1", due_day=100)
        triggered = identify_shipments_requiring_decision(state, known_shocks=())
        assert triggered == ("s1",)

    def test_capacity_wait_days_triggers(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1", due_day=100, capacity_wait_days=2)
        triggered = identify_shipments_requiring_decision(state, known_shocks=())
        assert triggered == ("s1",)

    def test_estimated_late_arrival_triggers(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1", due_day=1)  # lead time alone is 2 days
        triggered = identify_shipments_requiring_decision(state, known_shocks=())
        assert triggered == ("s1",)

    def test_shipment_at_destination_excluded(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(
            shipment_id="s1", current_node_id="plant_1", destination_node_id="plant_1", due_day=1
        )
        triggered = identify_shipments_requiring_decision(state, known_shocks=())
        assert triggered == ()

    def test_in_transit_shipment_excluded(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(
            shipment_id="s1",
            status=ShipmentStatus.IN_TRANSIT,
            current_node_id=None,
            current_edge_id="supplier_to_hub",
            edge_entry_day=1,
            edge_arrival_day=2,
        )
        triggered = identify_shipments_requiring_decision(state, known_shocks=())
        assert triggered == ()

    def test_deterministic_order_by_due_day(self) -> None:
        state = _tiny_state(day=1)
        state.edge_operational_state["supplier_to_hub"].available = False
        state.shipments["late"] = _shipment(shipment_id="late", due_day=50)
        state.shipments["early"] = _shipment(shipment_id="early", due_day=10)
        triggered = identify_shipments_requiring_decision(state, known_shocks=())
        assert triggered == ("early", "late")

    def test_active_known_shock_on_route_triggers(self) -> None:
        state = _tiny_state(day=3)
        shock = Shock(
            shock_id="s1",
            shock_type=ShockType.EDGE_LEAD_TIME_INCREASE,
            target_type=TargetType.EDGE,
            target_id="hub_to_plant",
            physical_start_day=3,
            physical_end_day=5,
            information_day=3,
            lead_time_multiplier=2.0,
        )
        apply_shock_operational_state(state, (shock,))
        state.shipments["s1"] = _shipment(shipment_id="s1", due_day=100)
        triggered = identify_shipments_requiring_decision(state, known_shocks=(shock,))
        assert triggered == ("s1",)

    def test_empty_remaining_route_triggers(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(
            shipment_id="s1",
            due_day=100,
            planned_route_edge_ids=("supplier_to_hub", "hub_to_plant"),
            next_edge_index=2,  # already past the end of its planned route
        )
        triggered = identify_shipments_requiring_decision(state, known_shocks=())
        assert triggered == ("s1",)

    def test_discontinuous_route_triggers(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(
            shipment_id="s1",
            due_day=100,
            # hub_to_plant ends at plant_1, supplier_to_hub starts at supplier_1: not continuous.
            planned_route_edge_ids=("hub_to_plant", "supplier_to_hub"),
            next_edge_index=0,
        )
        triggered = identify_shipments_requiring_decision(state, known_shocks=())
        assert triggered == ("s1",)


class TestApplyValidatedActions:
    def test_wait_leaves_route_unchanged(self) -> None:
        state = _tiny_state()
        state.shipments["s1"] = _shipment(shipment_id="s1")
        action = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.WAIT,
            route_id=None,
            reason_code=ReasonCode.REDUCE_LATENESS,
            rationale="",
        )
        apply_validated_actions(state, {"s1": (action, None)}, reroute_cost_per_unit=1.0, expedite_premium_per_unit=2.0)
        assert state.shipments["s1"].planned_route_edge_ids == ("supplier_to_hub", "hub_to_plant")
        assert state.service.wait_count == 1
        assert state.service.decision_count == 1

    def test_reroute_replaces_route_and_charges_cost(self) -> None:
        state = _tiny_state()
        state.shipments["s1"] = _shipment(shipment_id="s1", quantity=5)
        route = Route(
            route_id="supplier_to_plant_air",
            edge_ids=("supplier_to_plant_air",),
            node_ids=("supplier_1", "plant_1"),
            contains_emergency_edge=False,
        )
        action = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.REROUTE,
            route_id=route.route_id,
            reason_code=ReasonCode.LOWER_ESTIMATED_COST,
            rationale="",
        )
        apply_validated_actions(
            state, {"s1": (action, route)}, reroute_cost_per_unit=2.0, expedite_premium_per_unit=10.0
        )
        shipment = state.shipments["s1"]
        assert shipment.planned_route_edge_ids == ("supplier_to_plant_air",)
        assert shipment.next_edge_index == 0
        assert shipment.reroute_count == 1
        assert state.costs.reroute == pytest.approx(10.0)

    def test_expedite_replaces_route_and_charges_cost(self) -> None:
        state = _tiny_state()
        state.shipments["s1"] = _shipment(shipment_id="s1", quantity=5)
        route = Route(
            route_id="supplier_to_plant_air",
            edge_ids=("supplier_to_plant_air",),
            node_ids=("supplier_1", "plant_1"),
            contains_emergency_edge=True,
        )
        action = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.EXPEDITE,
            route_id=route.route_id,
            reason_code=ReasonCode.REDUCE_LATENESS,
            rationale="",
        )
        apply_validated_actions(
            state, {"s1": (action, route)}, reroute_cost_per_unit=2.0, expedite_premium_per_unit=10.0
        )
        shipment = state.shipments["s1"]
        assert shipment.expedite_count == 1
        assert state.service.expedited_units == 5
        assert state.costs.expedite == pytest.approx(50.0)

    def test_reroute_without_route_rejected(self) -> None:
        state = _tiny_state()
        state.shipments["s1"] = _shipment(shipment_id="s1")
        action = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.REROUTE,
            route_id="x",
            reason_code=ReasonCode.LOWER_ESTIMATED_COST,
            rationale="",
        )
        with pytest.raises(SimulationInvariantError, match="REROUTE"):
            apply_validated_actions(
                state, {"s1": (action, None)}, reroute_cost_per_unit=2.0, expedite_premium_per_unit=10.0
            )

    def test_expedite_without_route_rejected(self) -> None:
        state = _tiny_state()
        state.shipments["s1"] = _shipment(shipment_id="s1")
        action = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.EXPEDITE,
            route_id="x",
            reason_code=ReasonCode.REDUCE_LATENESS,
            rationale="",
        )
        with pytest.raises(SimulationInvariantError, match="EXPEDITE"):
            apply_validated_actions(
                state, {"s1": (action, None)}, reroute_cost_per_unit=2.0, expedite_premium_per_unit=10.0
            )


class TestAllocateDepartures:
    def test_dispatches_when_capacity_sufficient(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1", quantity=5)
        allocate_departures(state, edge_extra_delay_days={"supplier_to_hub": 0})
        shipment = state.shipments["s1"]
        assert shipment.status is ShipmentStatus.IN_TRANSIT
        assert shipment.current_edge_id == "supplier_to_hub"
        assert shipment.edge_arrival_day == 2  # day 1 + lead_time 1 + extra_delay 0
        assert state.daily_edge_used_capacity["supplier_to_hub"] == 5
        assert state.costs.transport == pytest.approx(5.0)

    def test_extra_delay_extends_arrival_day(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1", quantity=5)
        allocate_departures(state, edge_extra_delay_days={"supplier_to_hub": 1})
        assert state.shipments["s1"].edge_arrival_day == 3

    def test_insufficient_edge_capacity_increments_wait_days(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1", quantity=5)
        state.daily_edge_used_capacity["supplier_to_hub"] = 20  # full capacity already used
        allocate_departures(state, edge_extra_delay_days={})
        shipment = state.shipments["s1"]
        assert shipment.status is ShipmentStatus.AT_NODE
        assert shipment.capacity_wait_days == 1

    def test_insufficient_node_capacity_increments_wait_days(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1", quantity=5)
        state.daily_node_used_processing["supplier_1"] = 50  # full processing capacity used
        allocate_departures(state, edge_extra_delay_days={})
        assert state.shipments["s1"].status is ShipmentStatus.AT_NODE
        assert state.shipments["s1"].capacity_wait_days == 1

    def test_unavailable_edge_skips_without_incrementing_wait_days(self) -> None:
        state = _tiny_state(day=1)
        state.edge_operational_state["supplier_to_hub"].available = False
        state.shipments["s1"] = _shipment(shipment_id="s1", quantity=5)
        allocate_departures(state, edge_extra_delay_days={})
        shipment = state.shipments["s1"]
        assert shipment.status is ShipmentStatus.AT_NODE
        assert shipment.capacity_wait_days == 0

    def test_successful_dispatch_resets_wait_days(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1", quantity=5, capacity_wait_days=3)
        allocate_departures(state, edge_extra_delay_days={})
        assert state.shipments["s1"].capacity_wait_days == 0

    def test_shipment_without_current_node_is_skipped(self) -> None:
        # Shipment.__post_init__ forbids constructing this combination directly (an AT_NODE
        # shipment always has a current_node_id); mutate after construction to simulate a
        # caller mistakenly leaving current_node_id unset, which this guard exists to catch.
        state = _tiny_state(day=1)
        shipment = _shipment(shipment_id="s1", quantity=5)
        shipment.current_node_id = None
        state.shipments["s1"] = shipment
        allocate_departures(state, edge_extra_delay_days={})
        assert state.shipments["s1"].status is ShipmentStatus.AT_NODE


class TestChargeEndOfDayCosts:
    def test_charges_holding_and_backlog(self) -> None:
        state = _tiny_state()
        state.inventory["plant_1"]["widget"] = 10
        state.backlog["plant_1"]["widget"] = 4
        charge_end_of_day_costs(state)
        assert state.costs.holding == pytest.approx(10 * 0.10)
        assert state.costs.backlog == pytest.approx(4 * 1.00)


class TestRecordDailyMetrics:
    def test_assembles_expected_fields(self) -> None:
        state = _tiny_state(day=5)
        state.shipments["s1"] = _shipment(shipment_id="s1")
        state.inventory["plant_1"]["widget"] = 10
        state.backlog["plant_1"]["widget"] = 2
        state.active_shock_ids = {"shock_a"}
        demand_result = fulfil_backlog_and_demand(
            state, (DemandEvent(day=5, destination_node_id="plant_1", product_id="widget", quantity=3),)
        )
        metrics = record_daily_metrics(
            state,
            experiment_id="exp1",
            scenario_id="scenario1",
            replication=1,
            policy_name="wait_only",
            run_kind="DISRUPTED",
            demand_result=demand_result,
            daily_transport_cost=1.0,
            daily_reroute_cost=2.0,
            daily_expedite_cost=3.0,
            daily_holding_cost=4.0,
            daily_backlog_cost=5.0,
            daily_late_cost=6.0,
        )
        assert metrics.day == 5
        assert metrics.experiment_id == "exp1"
        assert metrics.shipments_at_node == 1
        assert metrics.active_shock_ids == ("shock_a",)
        assert metrics.daily_demand_units == 3
