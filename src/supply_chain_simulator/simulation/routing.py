"""Graph construction, candidate-route enumeration, and route cost estimates.

Inside the simulation package, this module builds the day's operational
graph from the immutable network plus current availability and multipliers,
finds candidate routes for a shipment that needs a decision, and estimates
what each candidate — and the shipment's current plan — would cost. In the
full system, this is the single shared source of route options and estimates
for both the heuristic and the LLM agent, which is what keeps the comparison
fair: neither policy can see a route the other could not have seen, and
neither uses a different cost formula. It does not choose an action, does
not validate one, and never mutates the SimulationState it reads from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import networkx as nx  # type: ignore[import-untyped]

from supply_chain_simulator.domain.events import Shock
from supply_chain_simulator.domain.models import Edge, Node, Route
from supply_chain_simulator.domain.state import (
    OperationalEdgeState,
    OperationalNodeState,
    Shipment,
    SimulationState,
)

MAX_ROUTE_EDGES = 6
CANDIDATE_PATHS_PER_WEIGHT = 10
MAX_ROUTE_OPTIONS = 5


class RoutingError(Exception):
    """Raised when a route or route estimation request cannot be satisfied."""


@dataclass(frozen=True, slots=True)
class RouteEstimate:
    route_id: str
    edge_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    contains_emergency_edge: bool
    estimated_lead_time_days: int | None
    estimated_arrival_day: int | None
    estimated_lateness_days: int | None
    estimated_transport_cost: float | None
    estimated_action_cost: float
    estimated_late_penalty: float | None
    estimated_total_cost: float | None
    first_edge_remaining_capacity: int | None
    currently_dispatchable: bool


def get_effective_edge_capacity(edge: Edge, edge_state: OperationalEdgeState) -> int:
    return math.floor(edge.daily_capacity * edge_state.capacity_multiplier)


def get_effective_edge_lead_time(edge: Edge, edge_state: OperationalEdgeState) -> int:
    return max(1, math.ceil(edge.base_lead_time_days * edge_state.lead_time_multiplier))


def get_effective_edge_cost(edge: Edge, edge_state: OperationalEdgeState) -> float:
    return edge.unit_transport_cost * edge_state.cost_multiplier


def _effective_node_processing_capacity(node: Node, node_state: OperationalNodeState) -> int:
    return math.floor(node.processing_capacity * node_state.processing_capacity_multiplier)


def route_from_edge_ids(state: SimulationState, edge_ids: tuple[str, ...]) -> Route:
    if not edge_ids:
        raise RoutingError("a route must contain at least one edge")

    edges = [state.network_definition.get_edge(edge_id) for edge_id in edge_ids]
    for previous_edge, next_edge in pairwise(edges):
        if previous_edge.destination_node_id != next_edge.origin_node_id:
            raise RoutingError(
                f"edges {previous_edge.edge_id} and {next_edge.edge_id} are not continuous"
            )

    node_ids = (edges[0].origin_node_id, *(edge.destination_node_id for edge in edges))
    return Route(
        route_id="__".join(edge_ids),
        edge_ids=edge_ids,
        node_ids=node_ids,
        contains_emergency_edge=any(edge.emergency for edge in edges),
    )


def build_operational_graph(state: SimulationState) -> nx.DiGraph:
    """Builds a DiGraph of currently-available nodes and edges only.

    When two edges happen to share the same origin and destination, the
    lexicographically smallest edge_id is kept as that hop's graph
    representative; this project's configured networks never define
    parallel lanes, so this tie-break is a documented safeguard, not an
    exercised behavior.
    """
    graph = nx.DiGraph()
    for node_id in sorted(state.network_definition.nodes):
        if state.node_operational_state[node_id].available:
            graph.add_node(node_id)

    for edge_id in sorted(state.network_definition.edges):
        edge = state.network_definition.get_edge(edge_id)
        edge_state = state.edge_operational_state[edge_id]
        if not edge_state.available:
            continue
        if edge.origin_node_id not in graph or edge.destination_node_id not in graph:
            continue
        if graph.has_edge(edge.origin_node_id, edge.destination_node_id):
            continue

        graph.add_edge(
            edge.origin_node_id,
            edge.destination_node_id,
            edge_id=edge_id,
            cost=get_effective_edge_cost(edge, edge_state),
            lead_time=get_effective_edge_lead_time(edge, edge_state),
        )
    return graph


def _build_route_estimate(
    state: SimulationState,
    shipment: Shipment,
    route: Route,
    action_cost: float,
    start_day: int,
) -> RouteEstimate:
    edges = [state.network_definition.get_edge(edge_id) for edge_id in route.edge_ids]
    edge_states = [state.edge_operational_state[edge_id] for edge_id in route.edge_ids]

    lead_time_days = sum(
        get_effective_edge_lead_time(edge, edge_state)
        for edge, edge_state in zip(edges, edge_states, strict=True)
    )
    transport_cost = shipment.quantity * sum(
        get_effective_edge_cost(edge, edge_state)
        for edge, edge_state in zip(edges, edge_states, strict=True)
    )
    arrival_day = start_day + lead_time_days
    lateness_days = max(0, arrival_day - shipment.due_day)
    product = state.network_definition.get_product(shipment.product_id)
    late_penalty = shipment.quantity * lateness_days * product.late_penalty_per_unit_day
    total_cost = transport_cost + action_cost + late_penalty

    first_edge_id = route.edge_ids[0]
    first_edge_capacity = get_effective_edge_capacity(edges[0], edge_states[0])
    remaining_capacity = first_edge_capacity - state.daily_edge_used_capacity.get(first_edge_id, 0)
    dispatchable = edge_states[0].available and remaining_capacity >= shipment.quantity

    return RouteEstimate(
        route_id=route.route_id,
        edge_ids=route.edge_ids,
        node_ids=route.node_ids,
        contains_emergency_edge=route.contains_emergency_edge,
        estimated_lead_time_days=lead_time_days,
        estimated_arrival_day=arrival_day,
        estimated_lateness_days=lateness_days,
        estimated_transport_cost=transport_cost,
        estimated_action_cost=action_cost,
        estimated_late_penalty=late_penalty,
        estimated_total_cost=total_cost,
        first_edge_remaining_capacity=remaining_capacity,
        currently_dispatchable=dispatchable,
    )


def estimate_route_option(
    state: SimulationState,
    shipment: Shipment,
    route: Route,
    action_cost: float,
) -> RouteEstimate:
    return _build_route_estimate(state, shipment, route, action_cost, start_day=state.day)


def _blocked_components(state: SimulationState, route: Route) -> tuple[str, ...]:
    blocked = [
        node_id for node_id in route.node_ids if not state.node_operational_state[node_id].available
    ]
    blocked += [
        edge_id for edge_id in route.edge_ids if not state.edge_operational_state[edge_id].available
    ]
    return tuple(blocked)


def _earliest_known_reopening_day(
    state: SimulationState,
    blocked_component_ids: tuple[str, ...],
    known_shocks: tuple[Shock, ...],
) -> int | None:
    blocked = set(blocked_component_ids)
    relevant_end_days = [
        shock.physical_end_day
        for shock in known_shocks
        if shock.target_id in blocked and shock.physical_end_day >= state.day
    ]
    if not relevant_end_days:
        return None
    return max(relevant_end_days) + 1


def estimate_current_plan(
    state: SimulationState,
    shipment: Shipment,
    known_shocks: tuple[Shock, ...],
    action_cost: float = 0.0,
) -> RouteEstimate:
    """Estimates the shipment's existing remaining route.

    If the remaining route is currently unblocked, this is exactly like
    estimating a freshly-found candidate. If it is blocked, the estimate
    assumes the whole remaining journey starts once every blocking shock in
    `known_shocks` has ended (a deliberately conservative simplification:
    a block partway down a multi-edge route is treated the same as a block
    on the very next edge). If no known shock explains the block, arrival
    and cost are None per CLAUDE.md section 21.1, and the plan is reported
    as not currently dispatchable.
    """
    if shipment.current_node_id is None:
        raise RoutingError("estimate_current_plan requires a shipment that is AT_NODE")

    remaining_edge_ids = shipment.planned_route_edge_ids[shipment.next_edge_index :]
    if not remaining_edge_ids:
        raise RoutingError(f"shipment {shipment.shipment_id} has no remaining route to estimate")
    route = route_from_edge_ids(state, remaining_edge_ids)

    blocked = _blocked_components(state, route)
    if not blocked:
        return _build_route_estimate(state, shipment, route, action_cost, start_day=state.day)

    reopening_day = _earliest_known_reopening_day(state, blocked, known_shocks)
    if reopening_day is None:
        return RouteEstimate(
            route_id=route.route_id,
            edge_ids=route.edge_ids,
            node_ids=route.node_ids,
            contains_emergency_edge=route.contains_emergency_edge,
            estimated_lead_time_days=None,
            estimated_arrival_day=None,
            estimated_lateness_days=None,
            estimated_transport_cost=None,
            estimated_action_cost=action_cost,
            estimated_late_penalty=None,
            estimated_total_cost=None,
            first_edge_remaining_capacity=None,
            currently_dispatchable=False,
        )
    return _build_route_estimate(state, shipment, route, action_cost, start_day=reopening_day)


def _sort_and_limit_route_estimates(estimates: list[RouteEstimate]) -> tuple[RouteEstimate, ...]:
    def sort_key(estimate: RouteEstimate) -> tuple[float, float, int, str]:
        return (
            estimate.estimated_total_cost if estimate.estimated_total_cost is not None else math.inf,
            estimate.estimated_arrival_day if estimate.estimated_arrival_day is not None else math.inf,
            len(estimate.edge_ids),
            estimate.route_id,
        )

    return tuple(sorted(estimates, key=sort_key)[:MAX_ROUTE_OPTIONS])


def enumerate_candidate_routes(
    state: SimulationState,
    shipment: Shipment,
    reroute_cost_per_unit: float,
    expedite_premium_per_unit: float,
) -> tuple[RouteEstimate, ...]:
    if shipment.current_node_id is None:
        raise RoutingError("enumerate_candidate_routes requires a shipment that is AT_NODE")

    graph = build_operational_graph(state)
    origin = shipment.current_node_id
    destination = shipment.destination_node_id
    if origin not in graph or destination not in graph:
        return ()

    candidate_edge_id_paths: dict[tuple[str, ...], None] = {}
    for weight in ("cost", "lead_time"):
        try:
            path_iterator = nx.shortest_simple_paths(graph, origin, destination, weight=weight)
            for count, node_path in enumerate(path_iterator):
                if count >= CANDIDATE_PATHS_PER_WEIGHT:
                    break
                if len(node_path) - 1 > MAX_ROUTE_EDGES:
                    continue
                edge_id_path = tuple(graph[u][v]["edge_id"] for u, v in pairwise(node_path))
                candidate_edge_id_paths.setdefault(edge_id_path, None)
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            continue

    quantity = shipment.quantity
    estimates = []
    for edge_id_path in candidate_edge_id_paths:
        route = route_from_edge_ids(state, edge_id_path)
        edges = [state.network_definition.get_edge(edge_id) for edge_id in edge_id_path]
        edge_states = [state.edge_operational_state[edge_id] for edge_id in edge_id_path]

        capacity_ok = all(
            get_effective_edge_capacity(edge, edge_state) >= quantity
            for edge, edge_state in zip(edges, edge_states, strict=True)
        )
        if not capacity_ok:
            continue
        processing_ok = all(
            _effective_node_processing_capacity(
                state.network_definition.get_node(edge.origin_node_id),
                state.node_operational_state[edge.origin_node_id],
            )
            >= quantity
            for edge in edges
        )
        if not processing_ok:
            continue

        action_cost = (
            expedite_premium_per_unit * quantity
            if route.contains_emergency_edge
            else reroute_cost_per_unit * quantity
        )
        estimates.append(estimate_route_option(state, shipment, route, action_cost))

    return _sort_and_limit_route_estimates(estimates)
