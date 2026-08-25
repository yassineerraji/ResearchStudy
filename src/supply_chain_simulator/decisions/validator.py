"""Validates a proposed DecisionAction against its observation (shape, target, route feasibility, emergency semantics) before it can be applied. Classifies an invalid action; never repairs one."""

from __future__ import annotations

from supply_chain_simulator.decisions.observation import DecisionObservation
from supply_chain_simulator.domain.actions import (
    MAX_RATIONALE_LENGTH,
    ActionType,
    DecisionAction,
    ValidationCode,
    ValidationResult,
)
from supply_chain_simulator.domain.models import Route
from supply_chain_simulator.domain.state import ShipmentStatus, SimulationState
from supply_chain_simulator.simulation.routing import (
    RoutingError,
    get_effective_edge_capacity,
    route_from_edge_ids,
)


def validate_action(
    action: DecisionAction,
    observation: DecisionObservation,
    state: SimulationState,
) -> ValidationResult:
    """Runs the twelve validation checks, in order, and stops at
    the first failure.

    A DELIVERED shipment is reported as SHIPMENT_ALREADY_DELIVERED rather than
    the more generic SHIPMENT_NOT_AT_NODE, which is checked second so that the
    more specific code is reachable.
    """
    schema_error = _check_schema(action)
    if schema_error is not None:
        return schema_error

    if action.shipment_id != observation.shipment.shipment_id:
        return ValidationResult(
            ValidationCode.ACTION_SHIPMENT_MISMATCH,
            f"action targets {action.shipment_id!r} but observation is for "
            f"{observation.shipment.shipment_id!r}",
        )

    shipment = state.shipments.get(action.shipment_id)
    if shipment is None:
        return ValidationResult(
            ValidationCode.SHIPMENT_NOT_FOUND,
            f"unknown shipment_id {action.shipment_id!r}",
        )

    if shipment.status is ShipmentStatus.DELIVERED:
        return ValidationResult(
            ValidationCode.SHIPMENT_ALREADY_DELIVERED,
            f"shipment {shipment.shipment_id} is already delivered",
        )
    if shipment.status is not ShipmentStatus.AT_NODE:
        return ValidationResult(
            ValidationCode.SHIPMENT_NOT_AT_NODE,
            f"shipment {shipment.shipment_id} is {shipment.status.value}, not AT_NODE",
        )

    if action.action_type in (ActionType.WAIT, ActionType.ABSTAIN):
        return ValidationResult(ValidationCode.VALID, "no route change requested")

    if action.route_id is None:
        return ValidationResult(
            ValidationCode.ROUTE_REQUIRED,
            f"{action.action_type.value} requires a route_id",
        )

    resolved_route = _resolve_route(state, action.route_id)
    if resolved_route is None:
        return ValidationResult(
            ValidationCode.ROUTE_NOT_FOUND,
            f"route_id {action.route_id!r} does not resolve to a continuous route",
        )

    approved = any(
        option.route_id == action.route_id for option in observation.route_options
    )
    if not approved:
        return ValidationResult(
            ValidationCode.ROUTE_NOT_ALLOWED,
            f"route_id {action.route_id!r} is not one of the observation's approved route options",
        )

    if resolved_route.node_ids[0] != shipment.current_node_id:
        return ValidationResult(
            ValidationCode.ROUTE_WRONG_ORIGIN,
            f"route {action.route_id!r} begins at {resolved_route.node_ids[0]!r}, not "
            f"{shipment.current_node_id!r}",
        )
    if resolved_route.node_ids[-1] != shipment.destination_node_id:
        return ValidationResult(
            ValidationCode.ROUTE_WRONG_DESTINATION,
            f"route {action.route_id!r} ends at {resolved_route.node_ids[-1]!r}, not "
            f"{shipment.destination_node_id!r}",
        )

    unavailable = _unavailable_components(state, resolved_route)
    if unavailable:
        return ValidationResult(
            ValidationCode.ROUTE_USES_UNAVAILABLE_COMPONENT,
            f"route {action.route_id!r} uses unavailable component(s): {', '.join(unavailable)}",
        )

    if not _fits_static_capacity(state, resolved_route, shipment.quantity):
        return ValidationResult(
            ValidationCode.ROUTE_STATIC_CAPACITY_TOO_SMALL,
            f"route {action.route_id!r} has an edge with effective capacity below "
            f"{shipment.quantity} units",
        )

    if (
        action.action_type is ActionType.REROUTE
        and resolved_route.contains_emergency_edge
    ):
        return ValidationResult(
            ValidationCode.REROUTE_USES_EMERGENCY_EDGE,
            f"REROUTE route {action.route_id!r} contains an emergency edge",
        )
    if (
        action.action_type is ActionType.EXPEDITE
        and not resolved_route.contains_emergency_edge
    ):
        return ValidationResult(
            ValidationCode.EXPEDITE_HAS_NO_EMERGENCY_EDGE,
            f"EXPEDITE route {action.route_id!r} contains no emergency edge",
        )

    return ValidationResult(ValidationCode.VALID, "action is valid")


def _check_schema(action: DecisionAction) -> ValidationResult | None:
    if len(action.rationale) > MAX_RATIONALE_LENGTH:
        return ValidationResult(
            ValidationCode.INVALID_ACTION_SCHEMA,
            f"rationale exceeds {MAX_RATIONALE_LENGTH} characters",
        )
    if (
        action.action_type in (ActionType.WAIT, ActionType.ABSTAIN)
        and action.route_id is not None
    ):
        return ValidationResult(
            ValidationCode.INVALID_ACTION_SCHEMA,
            f"{action.action_type.value} must not include a route_id",
        )
    return None


def _resolve_route(state: SimulationState, route_id: str) -> Route | None:
    edge_ids = tuple(route_id.split("__"))
    try:
        return route_from_edge_ids(state, edge_ids)
    except (RoutingError, KeyError):
        return None


def _unavailable_components(state: SimulationState, route: Route) -> list[str]:
    unavailable = [
        node_id
        for node_id in route.node_ids
        if not state.node_operational_state[node_id].available
    ]
    unavailable += [
        edge_id
        for edge_id in route.edge_ids
        if not state.edge_operational_state[edge_id].available
    ]
    return unavailable


def _fits_static_capacity(state: SimulationState, route: Route, quantity: int) -> bool:
    for edge_id in route.edge_ids:
        edge = state.network_definition.get_edge(edge_id)
        edge_state = state.edge_operational_state[edge_id]
        if get_effective_edge_capacity(edge, edge_state) < quantity:
            return False
    return True
