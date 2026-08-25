"""The individual daily state-change steps, releases, arrivals, demand fulfilment, applying validated actions, departures, in the exact order engine.py calls them. Chooses no actions itself."""

from __future__ import annotations

import math
from dataclasses import dataclass

from supply_chain_simulator.domain.actions import ActionType, DecisionAction
from supply_chain_simulator.domain.events import (
    DemandEvent,
    ShipmentReleaseEvent,
    Shock,
    ShockType,
    TargetType,
)
from supply_chain_simulator.domain.models import Route
from supply_chain_simulator.domain.state import (
    DailyMetrics,
    OperationalEdgeState,
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
    total_cost,
)
from supply_chain_simulator.simulation.routing import (
    RoutingError,
    estimate_current_plan,
    get_effective_edge_capacity,
    get_effective_edge_cost,
    get_effective_edge_lead_time,
    route_from_edge_ids,
)


class SimulationInvariantError(Exception):
    """Raised when a required physical guarantee or daily invariant is violated."""


def _node_occupancy(state: SimulationState, node_id: str) -> int:
    inventory_total = sum(state.inventory[node_id].values())
    at_node_total = sum(
        shipment.quantity
        for shipment in state.shipments.values()
        if shipment.status is ShipmentStatus.AT_NODE and shipment.current_node_id == node_id
    )
    return inventory_total + at_node_total


# --- step 1: begin day ---------------------------------------------------------


def reset_daily_capacity_usage(state: SimulationState) -> None:
    for edge_id in state.daily_edge_used_capacity:
        state.daily_edge_used_capacity[edge_id] = 0
    for node_id in state.daily_node_used_processing:
        state.daily_node_used_processing[node_id] = 0


# --- step 3: apply physical shocks ---------------------------------------------


def apply_shock_operational_state(state: SimulationState, shocks: tuple[Shock, ...]) -> None:
    """Rebuilds node/edge operational state from the immutable base every day.

    Availability combines by logical AND (any active closure wins) and
    multipliers combine by multiplication, per CLAUDE.md section 18.3. Never
    incrementally mutates a stale multiplier from a previous day.
    """
    for node_id in state.node_operational_state:
        state.node_operational_state[node_id] = OperationalNodeState()
    for edge_id in state.edge_operational_state:
        state.edge_operational_state[edge_id] = OperationalEdgeState()

    active_shock_ids = set()
    for shock in shocks:
        if not (shock.physical_start_day <= state.day <= shock.physical_end_day):
            continue
        active_shock_ids.add(shock.shock_id)

        if shock.target_type is TargetType.DEMAND:
            # DEMAND_SPIKE/DEMAND_DROP are realized into event-tape generation
            # parameters (V2 §V2.3.2/§V2.3.3), never into runtime operational
            # state -- this function's node/edge surface area is unchanged.
            continue

        if shock.target_type is TargetType.NODE:
            node_state = state.node_operational_state.get(shock.target_id)
            if node_state is None:
                continue
            if shock.shock_type is ShockType.NODE_CLOSURE:
                node_state.available = False
            elif shock.shock_type is ShockType.NODE_CAPACITY_REDUCTION:
                node_state.processing_capacity_multiplier *= shock.capacity_multiplier
            elif shock.shock_type is ShockType.SUPPLIER_CAPACITY_REDUCTION:
                node_state.source_capacity_multiplier *= shock.capacity_multiplier
        else:
            edge_state = state.edge_operational_state.get(shock.target_id)
            if edge_state is None:
                continue
            if shock.shock_type is ShockType.EDGE_CLOSURE:
                edge_state.available = False
            elif shock.shock_type is ShockType.EDGE_CAPACITY_REDUCTION:
                edge_state.capacity_multiplier *= shock.capacity_multiplier
            elif shock.shock_type is ShockType.EDGE_LEAD_TIME_INCREASE:
                edge_state.lead_time_multiplier *= shock.lead_time_multiplier
            elif shock.shock_type is ShockType.EDGE_COST_INCREASE:
                edge_state.cost_multiplier *= shock.cost_multiplier

    state.active_shock_ids = active_shock_ids


# --- step 4: process arrivals ---------------------------------------------------


def process_due_arrivals(state: SimulationState) -> None:
    arriving_ids = sorted(
        shipment_id
        for shipment_id, shipment in state.shipments.items()
        if shipment.status is ShipmentStatus.IN_TRANSIT and shipment.edge_arrival_day == state.day
    )

    for shipment_id in arriving_ids:
        shipment = state.shipments[shipment_id]
        edge = state.network_definition.get_edge(shipment.current_edge_id)  # type: ignore[arg-type]
        destination_node_id = edge.destination_node_id
        node_state = state.node_operational_state[destination_node_id]

        if not node_state.available:
            shipment.edge_arrival_day = state.day + 1
            continue

        node = state.network_definition.get_node(destination_node_id)
        if _node_occupancy(state, destination_node_id) + shipment.quantity > node.storage_capacity:
            shipment.edge_arrival_day = state.day + 1
            continue

        if destination_node_id == shipment.destination_node_id:
            shipment.status = ShipmentStatus.DELIVERED
            shipment.current_node_id = destination_node_id
            shipment.current_edge_id = None
            shipment.edge_entry_day = None
            shipment.edge_arrival_day = None
            shipment.delivered_day = state.day

            state.inventory[destination_node_id][shipment.product_id] = (
                state.inventory[destination_node_id].get(shipment.product_id, 0) + shipment.quantity
            )
            state.service.delivered_shipment_units += shipment.quantity

            lateness_days = max(0, shipment.delivered_day - shipment.due_day)
            if lateness_days > 0:
                state.service.late_delivered_units += shipment.quantity
                state.service.total_lateness_unit_days += shipment.quantity * lateness_days
            charge_delivery_late_penalty(state, shipment)
        else:
            shipment.status = ShipmentStatus.AT_NODE
            shipment.current_node_id = destination_node_id
            shipment.current_edge_id = None
            shipment.edge_entry_day = None
            shipment.edge_arrival_day = None
            shipment.next_edge_index += 1


# --- step 5: release scheduled shipments ----------------------------------------


def release_shipments(state: SimulationState, release_events: tuple[ShipmentReleaseEvent, ...]) -> int:
    """Attempts every entry already in state.pending_releases (oldest-scheduled
    first, i.e. ascending shipment_id) before today's newly scheduled events
    (also ascending shipment_id), per CLAUDE.md V2 §V2.3.7. A release whose
    source is unavailable, over its effective source_capacity, or would
    overflow storage no longer raises: it is appended to
    state.pending_releases and retried on a later day. Returns the total
    quantity actually released this call -- the engine accumulates this into
    its running total_released, since a deferred release must not be counted
    until it actually enters the system.
    """
    pending = sorted(state.pending_releases, key=lambda e: e.shipment_id)
    state.pending_releases = []
    todays_events = sorted(release_events, key=lambda e: e.shipment_id)

    total_released = 0
    for event in (*pending, *todays_events):
        origin_node_id = event.origin_node_id
        node = state.network_definition.get_node(origin_node_id)
        node_state = state.node_operational_state[origin_node_id]
        effective_source_capacity = math.floor(node.source_capacity * node_state.source_capacity_multiplier)

        if (
            not node_state.available
            or event.quantity > effective_source_capacity
            or _node_occupancy(state, origin_node_id) + event.quantity > node.storage_capacity
        ):
            state.pending_releases.append(event)
            continue

        state.shipments[event.shipment_id] = Shipment(
            shipment_id=event.shipment_id,
            product_id=event.product_id,
            quantity=event.quantity,
            origin_node_id=event.origin_node_id,
            destination_node_id=event.destination_node_id,
            release_day=event.day,
            due_day=event.due_day,
            planned_route_edge_ids=event.initial_route_edge_ids,
            next_edge_index=0,
            status=ShipmentStatus.AT_NODE,
            current_node_id=origin_node_id,
            current_edge_id=None,
            edge_entry_day=None,
            edge_arrival_day=None,
            reroute_count=0,
            expedite_count=0,
            capacity_wait_days=0,
            delivered_day=None,
        )
        total_released += event.quantity

    return total_released


# --- step 6: realize and fulfil demand ------------------------------------------


@dataclass(frozen=True, slots=True)
class DemandFulfilmentResult:
    demand_units: int
    same_day_fulfilled_units: int
    backlog_fulfilled_units: int


def fulfil_backlog_and_demand(
    state: SimulationState, demand_events: tuple[DemandEvent, ...]
) -> DemandFulfilmentResult:
    total_demand = 0
    total_same_day = 0
    total_backlog_fulfilled = 0

    for event in sorted(demand_events, key=lambda e: (e.destination_node_id, e.product_id)):
        total_demand += event.quantity
        state.service.total_demand_units += event.quantity

        inventory = state.inventory[event.destination_node_id]
        backlog = state.backlog[event.destination_node_id]

        backlog_owed = backlog.get(event.product_id, 0)
        available = inventory.get(event.product_id, 0)
        backlog_cleared = min(available, backlog_owed)
        inventory[event.product_id] = available - backlog_cleared
        backlog[event.product_id] = backlog_owed - backlog_cleared
        total_backlog_fulfilled += backlog_cleared
        state.service.backlog_fulfilled_units += backlog_cleared

        remaining_inventory = inventory[event.product_id]
        same_day_fulfilled = min(remaining_inventory, event.quantity)
        inventory[event.product_id] = remaining_inventory - same_day_fulfilled
        total_same_day += same_day_fulfilled
        state.service.same_day_fulfilled_units += same_day_fulfilled

        unmet = event.quantity - same_day_fulfilled
        backlog[event.product_id] = backlog.get(event.product_id, 0) + unmet

    return DemandFulfilmentResult(
        demand_units=total_demand,
        same_day_fulfilled_units=total_same_day,
        backlog_fulfilled_units=total_backlog_fulfilled,
    )


# --- step 7: build decision set --------------------------------------------------


def identify_shipments_requiring_decision(
    state: SimulationState, known_shocks: tuple[Shock, ...]
) -> tuple[str, ...]:
    """Applies the exact triggers from CLAUDE.md section 19, in deterministic order."""
    triggered = [
        shipment_id
        for shipment_id, shipment in state.shipments.items()
        if shipment.status is ShipmentStatus.AT_NODE
        and shipment.current_node_id != shipment.destination_node_id
        and _shipment_requires_decision(state, shipment, known_shocks)
    ]
    return tuple(
        sorted(
            triggered,
            key=lambda sid: (
                state.shipments[sid].due_day,
                state.shipments[sid].release_day,
                sid,
            ),
        )
    )


def _shipment_requires_decision(
    state: SimulationState, shipment: Shipment, known_shocks: tuple[Shock, ...]
) -> bool:
    remaining_edge_ids = shipment.planned_route_edge_ids[shipment.next_edge_index :]
    if not remaining_edge_ids:
        return True

    try:
        route = route_from_edge_ids(state, remaining_edge_ids)
    except RoutingError:
        return True

    active_known_targets = {
        shock.target_id for shock in known_shocks if shock.shock_id in state.active_shock_ids
    }
    if active_known_targets & (set(route.node_ids) | set(route.edge_ids)):
        return True

    if shipment.capacity_wait_days >= 2:
        return True

    estimate = estimate_current_plan(state, shipment, known_shocks)
    if estimate.estimated_arrival_day is None:
        return True
    return estimate.estimated_arrival_day > shipment.due_day


# --- step 10: apply valid route changes ------------------------------------------


def apply_validated_actions(
    state: SimulationState,
    decisions: dict[str, tuple[DecisionAction, Route | None]],
    reroute_cost_per_unit: float,
    expedite_premium_per_unit: float,
) -> None:
    for shipment_id in sorted(decisions):
        action, route = decisions[shipment_id]
        shipment = state.shipments[shipment_id]
        state.service.decision_count += 1
        state.service.valid_action_count += 1

        if action.action_type is ActionType.WAIT:
            state.service.wait_count += 1
        elif action.action_type is ActionType.REROUTE:
            if route is None:
                raise SimulationInvariantError(f"REROUTE for {shipment_id} requires a route")
            shipment.planned_route_edge_ids = route.edge_ids
            shipment.next_edge_index = 0
            shipment.reroute_count += 1
            state.service.reroute_count += 1
            charge_reroute_cost(state, shipment.quantity, reroute_cost_per_unit)
        elif action.action_type is ActionType.EXPEDITE:
            if route is None:
                raise SimulationInvariantError(f"EXPEDITE for {shipment_id} requires a route")
            shipment.planned_route_edge_ids = route.edge_ids
            shipment.next_edge_index = 0
            shipment.expedite_count += 1
            state.service.expedite_count += 1
            state.service.expedited_units += shipment.quantity
            charge_expedite_cost(state, shipment.quantity, expedite_premium_per_unit)


# --- step 11: allocate departures -------------------------------------------------


def allocate_departures(state: SimulationState, edge_extra_delay_days: dict[str, int]) -> None:
    candidates = [
        shipment_id
        for shipment_id, shipment in state.shipments.items()
        if shipment.status is ShipmentStatus.AT_NODE
        and shipment.next_edge_index < len(shipment.planned_route_edge_ids)
    ]
    ordered = sorted(
        candidates,
        key=lambda sid: (
            state.shipments[sid].due_day,
            state.shipments[sid].release_day,
            sid,
        ),
    )

    for shipment_id in ordered:
        shipment = state.shipments[shipment_id]
        origin_node_id = shipment.current_node_id
        if origin_node_id is None:
            continue
        origin_node_state = state.node_operational_state[origin_node_id]

        edge_id = shipment.planned_route_edge_ids[shipment.next_edge_index]
        edge = state.network_definition.get_edge(edge_id)
        edge_state = state.edge_operational_state[edge_id]

        if not edge_state.available or not origin_node_state.available:
            continue

        edge_remaining = get_effective_edge_capacity(edge, edge_state) - state.daily_edge_used_capacity.get(
            edge_id, 0
        )
        origin_node = state.network_definition.get_node(origin_node_id)
        effective_node_capacity = math.floor(
            origin_node.processing_capacity * origin_node_state.processing_capacity_multiplier
        )
        node_remaining = effective_node_capacity - state.daily_node_used_processing.get(origin_node_id, 0)

        if edge_remaining < shipment.quantity or node_remaining < shipment.quantity:
            shipment.capacity_wait_days += 1
            continue

        charge_edge_entry_transport_cost(state, shipment.quantity, get_effective_edge_cost(edge, edge_state))
        effective_lead_time = get_effective_edge_lead_time(edge, edge_state)
        extra_delay = edge_extra_delay_days.get(edge_id, 0)
        arrival_day = state.day + effective_lead_time + extra_delay

        shipment.capacity_wait_days = 0
        shipment.status = ShipmentStatus.IN_TRANSIT
        shipment.current_node_id = None
        shipment.current_edge_id = edge_id
        shipment.edge_entry_day = state.day
        shipment.edge_arrival_day = arrival_day

        state.daily_edge_used_capacity[edge_id] = (
            state.daily_edge_used_capacity.get(edge_id, 0) + shipment.quantity
        )
        state.daily_node_used_processing[origin_node_id] = (
            state.daily_node_used_processing.get(origin_node_id, 0) + shipment.quantity
        )


# --- step 12: charge end-of-day costs ---------------------------------------------


def charge_end_of_day_costs(state: SimulationState) -> None:
    charge_end_of_day_holding_cost(state)
    charge_end_of_day_backlog_cost(state)


# --- step 13: record daily metrics -------------------------------------------------


def record_daily_metrics(
    state: SimulationState,
    experiment_id: str,
    scenario_id: str,
    replication: int,
    policy_name: str,
    run_kind: str,
    demand_result: DemandFulfilmentResult,
    daily_transport_cost: float,
    daily_reroute_cost: float,
    daily_expedite_cost: float,
    daily_holding_cost: float,
    daily_backlog_cost: float,
    daily_late_cost: float,
) -> DailyMetrics:
    shipments_at_node = sum(1 for s in state.shipments.values() if s.status is ShipmentStatus.AT_NODE)
    shipments_in_transit = sum(
        1 for s in state.shipments.values() if s.status is ShipmentStatus.IN_TRANSIT
    )
    shipments_delivered = sum(
        1 for s in state.shipments.values() if s.status is ShipmentStatus.DELIVERED
    )
    inventory_units = sum(sum(products.values()) for products in state.inventory.values())
    backlog_units = sum(sum(products.values()) for products in state.backlog.values())

    return DailyMetrics(
        experiment_id=experiment_id,
        scenario_id=scenario_id,
        replication=replication,
        policy=policy_name,
        run_kind=run_kind,
        day=state.day,
        inventory_units=inventory_units,
        backlog_units=backlog_units,
        shipments_at_node=shipments_at_node,
        shipments_in_transit=shipments_in_transit,
        shipments_delivered=shipments_delivered,
        daily_demand_units=demand_result.demand_units,
        daily_same_day_fulfilled_units=demand_result.same_day_fulfilled_units,
        daily_backlog_fulfilled_units=demand_result.backlog_fulfilled_units,
        daily_transport_cost=daily_transport_cost,
        daily_reroute_cost=daily_reroute_cost,
        daily_expedite_cost=daily_expedite_cost,
        daily_holding_cost=daily_holding_cost,
        daily_backlog_cost=daily_backlog_cost,
        daily_late_cost=daily_late_cost,
        cumulative_total_cost=total_cost(state),
        active_shock_ids=tuple(sorted(state.active_shock_ids)),
    )
