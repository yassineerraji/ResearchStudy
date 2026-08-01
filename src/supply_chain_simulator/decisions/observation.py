"""Builds the read-only DecisionObservation a policy is allowed to see.

Inside the decisions package, this module turns one shipment's slice of the
current SimulationState into an immutable DecisionObservation: the shipment's
own context, the demand destination's inventory and backlog picture, the
known shocks relevant to its remaining route, an estimate of its current
plan, and up to five candidate routes with their cost and lead-time
estimates. In the full system, this is what makes the heuristic and the LLM
agent's inputs provably equivalent — both are built from this same function,
against the same pre-action state, using the same route and cost
calculations from simulation/routing.py. It does not choose or validate an
action and never exposes or mutates the mutable SimulationState itself.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from supply_chain_simulator.domain.actions import ActionType
from supply_chain_simulator.domain.events import Shock, ShockType
from supply_chain_simulator.domain.state import Shipment, SimulationState
from supply_chain_simulator.simulation.routing import (
    RouteEstimate,
    RoutingError,
    enumerate_candidate_routes,
    estimate_current_plan,
)

# A "route option" here is exactly simulation.routing's RouteEstimate; reusing
# it keeps one shared shape for a route's facts instead of two parallel ones.
RouteOption = RouteEstimate


@dataclass(frozen=True, slots=True)
class ShipmentContext:
    shipment_id: str
    product_id: str
    quantity: int
    current_node_id: str
    destination_node_id: str
    release_day: int
    due_day: int
    days_until_due: int
    capacity_wait_days: int
    remaining_route_edge_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DestinationContext:
    inventory_on_hand: int
    backlog_units: int
    mean_daily_demand: float
    days_of_supply: float | None


@dataclass(frozen=True, slots=True)
class ShockContext:
    shock_id: str
    shock_type: ShockType
    target_id: str
    physical_start_day: int
    physical_end_day: int
    information_day: int
    known_effect: str


@dataclass(frozen=True, slots=True)
class DecisionObservation:
    observation_id: str
    day: int
    shipment: ShipmentContext
    destination: DestinationContext
    relevant_shocks: tuple[ShockContext, ...]
    current_plan: RouteOption
    route_options: tuple[RouteOption, ...]
    allowed_actions: tuple[ActionType, ...]


def build_observation(
    state: SimulationState,
    shipment_id: str,
    known_shocks: tuple[Shock, ...],
    mean_daily_demand: float,
    reroute_cost_per_unit: float,
    expedite_premium_per_unit: float,
) -> DecisionObservation:
    """Builds one shipment's observation from the current pre-action state.

    `known_shocks` must already be restricted to shocks the policy is allowed
    to know about on this day (CLAUDE.md section 20); this function does not
    itself filter by `state.known_shock_ids`, matching the convention already
    used by `simulation.routing.estimate_current_plan` and
    `simulation.transition.identify_shipments_requiring_decision`.
    """
    shipment = state.shipments[shipment_id]
    current_node_id = shipment.current_node_id
    if current_node_id is None:
        raise ValueError(
            f"cannot build an observation for {shipment_id}: shipment is not AT_NODE"
        )

    shipment_context = _build_shipment_context(shipment, current_node_id, state.day)
    destination_context = _build_destination_context(state, shipment, mean_daily_demand)
    relevant_shocks = _build_relevant_shocks(
        state, shipment, current_node_id, known_shocks
    )
    current_plan = _build_current_plan(state, shipment, known_shocks)
    route_options = enumerate_candidate_routes(
        state, shipment, reroute_cost_per_unit, expedite_premium_per_unit
    )
    allowed_actions = _allowed_actions(route_options)

    return DecisionObservation(
        observation_id=f"obs_{state.day:03d}_{shipment_id}",
        day=state.day,
        shipment=shipment_context,
        destination=destination_context,
        relevant_shocks=relevant_shocks,
        current_plan=current_plan,
        route_options=route_options,
        allowed_actions=allowed_actions,
    )


def _build_shipment_context(
    shipment: Shipment, current_node_id: str, day: int
) -> ShipmentContext:
    return ShipmentContext(
        shipment_id=shipment.shipment_id,
        product_id=shipment.product_id,
        quantity=shipment.quantity,
        current_node_id=current_node_id,
        destination_node_id=shipment.destination_node_id,
        release_day=shipment.release_day,
        due_day=shipment.due_day,
        days_until_due=shipment.due_day - day,
        capacity_wait_days=shipment.capacity_wait_days,
        remaining_route_edge_ids=shipment.planned_route_edge_ids[
            shipment.next_edge_index :
        ],
    )


def _build_destination_context(
    state: SimulationState, shipment: Shipment, mean_daily_demand: float
) -> DestinationContext:
    destination_node_id = shipment.destination_node_id
    inventory_on_hand = state.inventory[destination_node_id].get(shipment.product_id, 0)
    backlog_units = state.backlog[destination_node_id].get(shipment.product_id, 0)
    days_of_supply = (
        inventory_on_hand / mean_daily_demand if mean_daily_demand > 0 else None
    )
    return DestinationContext(
        inventory_on_hand=inventory_on_hand,
        backlog_units=backlog_units,
        mean_daily_demand=mean_daily_demand,
        days_of_supply=days_of_supply,
    )


def _build_relevant_shocks(
    state: SimulationState,
    shipment: Shipment,
    current_node_id: str,
    known_shocks: tuple[Shock, ...],
) -> tuple[ShockContext, ...]:
    relevant_ids = {current_node_id}
    for edge_id in shipment.planned_route_edge_ids[shipment.next_edge_index :]:
        edge = state.network_definition.get_edge(edge_id)
        relevant_ids.add(edge_id)
        relevant_ids.add(edge.origin_node_id)
        relevant_ids.add(edge.destination_node_id)

    relevant = sorted(
        (shock for shock in known_shocks if shock.target_id in relevant_ids),
        key=lambda shock: shock.shock_id,
    )
    return tuple(
        ShockContext(
            shock_id=shock.shock_id,
            shock_type=shock.shock_type,
            target_id=shock.target_id,
            physical_start_day=shock.physical_start_day,
            physical_end_day=shock.physical_end_day,
            information_day=shock.information_day,
            known_effect=_describe_known_effect(shock),
        )
        for shock in relevant
    )


def _describe_known_effect(shock: Shock) -> str:
    if shock.shock_type is ShockType.NODE_CLOSURE:
        return "node unavailable"
    if shock.shock_type is ShockType.EDGE_CLOSURE:
        return "edge unavailable"
    if shock.shock_type is ShockType.NODE_CAPACITY_REDUCTION:
        return f"processing capacity multiplier {shock.capacity_multiplier}"
    if shock.shock_type is ShockType.EDGE_CAPACITY_REDUCTION:
        return f"capacity multiplier {shock.capacity_multiplier}"
    if shock.shock_type is ShockType.EDGE_LEAD_TIME_INCREASE:
        return f"lead time multiplier {shock.lead_time_multiplier}"
    return f"cost multiplier {shock.cost_multiplier}"


def _build_current_plan(
    state: SimulationState, shipment: Shipment, known_shocks: tuple[Shock, ...]
) -> RouteOption:
    try:
        return estimate_current_plan(state, shipment, known_shocks)
    except RoutingError:
        # The remaining route is empty or discontinuous (CLAUDE.md section 19
        # trigger 5): report it as a non-dispatchable, un-costed plan rather
        # than letting route estimation fail the whole observation.
        remaining_edge_ids = shipment.planned_route_edge_ids[shipment.next_edge_index :]
        return RouteEstimate(
            route_id="__".join(remaining_edge_ids),
            edge_ids=remaining_edge_ids,
            node_ids=(),
            contains_emergency_edge=False,
            estimated_lead_time_days=None,
            estimated_arrival_day=None,
            estimated_lateness_days=None,
            estimated_transport_cost=None,
            estimated_action_cost=0.0,
            estimated_late_penalty=None,
            estimated_total_cost=None,
            first_edge_remaining_capacity=None,
            currently_dispatchable=False,
        )


def _allowed_actions(route_options: tuple[RouteOption, ...]) -> tuple[ActionType, ...]:
    actions = [ActionType.WAIT]
    if any(not option.contains_emergency_edge for option in route_options):
        actions.append(ActionType.REROUTE)
    if any(option.contains_emergency_edge for option in route_options):
        actions.append(ActionType.EXPEDITE)
    actions.append(ActionType.ABSTAIN)
    return tuple(actions)


def observation_to_canonical_dict(
    observation: DecisionObservation,
) -> dict[str, object]:
    """Converts an observation into a plain, JSON-serializable dict.

    `compute_observation_hash`, below, feeds this through
    `json.dumps(sort_keys=True)` and SHA-256 to compute a decision_key's
    observation_hash (CLAUDE.md section 23.1); building the canonical shape
    here keeps it next to the dataclasses it mirrors.
    """
    return {
        "observation_id": observation.observation_id,
        "day": observation.day,
        "shipment": _shipment_context_to_dict(observation.shipment),
        "destination": _destination_context_to_dict(observation.destination),
        "relevant_shocks": [
            _shock_context_to_dict(shock) for shock in observation.relevant_shocks
        ],
        "current_plan": _route_option_to_dict(observation.current_plan),
        "route_options": [
            _route_option_to_dict(option) for option in observation.route_options
        ],
        "allowed_actions": [action.value for action in observation.allowed_actions],
    }


def compute_observation_hash(observation: DecisionObservation) -> str:
    """SHA-256 of the observation's canonical JSON, per CLAUDE.md section 23.1."""
    canonical_json = json.dumps(
        observation_to_canonical_dict(observation),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _shipment_context_to_dict(context: ShipmentContext) -> dict[str, object]:
    return {
        "shipment_id": context.shipment_id,
        "product_id": context.product_id,
        "quantity": context.quantity,
        "current_node_id": context.current_node_id,
        "destination_node_id": context.destination_node_id,
        "release_day": context.release_day,
        "due_day": context.due_day,
        "days_until_due": context.days_until_due,
        "capacity_wait_days": context.capacity_wait_days,
        "remaining_route_edge_ids": list(context.remaining_route_edge_ids),
    }


def _destination_context_to_dict(context: DestinationContext) -> dict[str, object]:
    return {
        "inventory_on_hand": context.inventory_on_hand,
        "backlog_units": context.backlog_units,
        "mean_daily_demand": context.mean_daily_demand,
        "days_of_supply": context.days_of_supply,
    }


def _shock_context_to_dict(context: ShockContext) -> dict[str, object]:
    return {
        "shock_id": context.shock_id,
        "shock_type": context.shock_type.value,
        "target_id": context.target_id,
        "physical_start_day": context.physical_start_day,
        "physical_end_day": context.physical_end_day,
        "information_day": context.information_day,
        "known_effect": context.known_effect,
    }


def _route_option_to_dict(option: RouteOption) -> dict[str, object]:
    return {
        "route_id": option.route_id,
        "edge_ids": list(option.edge_ids),
        "node_ids": list(option.node_ids),
        "contains_emergency_edge": option.contains_emergency_edge,
        "estimated_lead_time_days": option.estimated_lead_time_days,
        "estimated_arrival_day": option.estimated_arrival_day,
        "estimated_lateness_days": option.estimated_lateness_days,
        "estimated_transport_cost": option.estimated_transport_cost,
        "estimated_action_cost": option.estimated_action_cost,
        "estimated_late_penalty": option.estimated_late_penalty,
        "estimated_total_cost": option.estimated_total_cost,
        "first_edge_remaining_capacity": option.first_edge_remaining_capacity,
        "currently_dispatchable": option.currently_dispatchable,
    }
