"""Drives the daily orchestration loop end to end: transitions, routing, decisions, and costs, for one policy over one event tape. Deep-clones its input, never branches on a concrete policy type, and optionally records decision/LLM traces."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from supply_chain_simulator.decisions.observation import (
    DecisionObservation,
    build_observation,
    compute_observation_hash,
)
from supply_chain_simulator.domain.actions import (
    ActionType,
    DecisionAction,
    ValidationResult,
)
from supply_chain_simulator.domain.events import DayEvents, EventTape, Shock
from supply_chain_simulator.domain.models import Route
from supply_chain_simulator.domain.state import (
    CostCounters,
    DailyMetrics,
    Shipment,
    ShipmentStatus,
    SimulationResult,
    SimulationState,
)
from supply_chain_simulator.integrations.llm_client import LLMInteractionResult
from supply_chain_simulator.policies.base import Policy, make_decision_record
from supply_chain_simulator.policies.fallback import resolve_action
from supply_chain_simulator.simulation import transition
from supply_chain_simulator.simulation.costs import charge_terminal_cost
from supply_chain_simulator.simulation.routing import (
    RoutingError,
    get_effective_edge_capacity,
    route_from_edge_ids,
)
from supply_chain_simulator.simulation.transition import SimulationInvariantError


@dataclass(frozen=True, slots=True)
class RunIdentity:
    experiment_id: str
    scenario_id: str
    replication: int
    policy_name: str
    run_kind: str


@dataclass(frozen=True, slots=True)
class DecisionTraceEntry:
    """One shipment's fully-resolved decision, for decision_traces.jsonl.
    `decision_key` there is the tuple (experiment_id, scenario_id,
    replication, run_kind, day, shipment_id, observation_hash); this entry
    carries `run_identity` and `day` instead of a pre-joined key so
    data_io/writers.py can format it however it needs.
    """

    run_identity: RunIdentity
    day: int
    shipment_id: str
    observation: DecisionObservation
    observation_hash: str
    proposed_action: DecisionAction
    proposal_validation: ValidationResult
    fallback_invoked: bool
    fallback_action: DecisionAction | None
    fallback_validation: ValidationResult | None
    executed_action: DecisionAction
    decision_latency_ms: float


def _all_resolved(state: SimulationState) -> bool:
    if any(
        shipment.status is not ShipmentStatus.DELIVERED
        for shipment in state.shipments.values()
    ):
        return False
    return all(
        quantity == 0
        for product_quantities in state.backlog.values()
        for quantity in product_quantities.values()
    )


def _cost_snapshot(costs: CostCounters) -> CostCounters:
    return CostCounters(
        transport=costs.transport,
        reroute=costs.reroute,
        expedite=costs.expedite,
        holding=costs.holding,
        backlog=costs.backlog,
        late=costs.late,
        terminal=costs.terminal,
    )


class SimulationEngine:
    """Runs a SimulationState through a fixed range of days.

    `policy`, `fallback_policy`, and `mean_daily_demand` are only required
    when `decision_enabled=True`; warm-up-style runs (`decision_enabled=False`)
    never build an observation or consult a policy, so they may be omitted.
    """

    def run(
        self,
        initial_state: SimulationState,
        event_tape: EventTape,
        start_day: int,
        horizon_day: int,
        drain_days: int,
        decision_enabled: bool,
        run_identity: RunIdentity,
        reroute_cost_per_unit: float,
        expedite_premium_per_unit: float,
        terminal_penalty_days: int,
        policy: Policy | None = None,
        fallback_policy: Policy | None = None,
        mean_daily_demand: float = 0.0,
        decision_trace_sink: list[DecisionTraceEntry] | None = None,
        llm_interaction_sink: list[LLMInteractionResult] | None = None,
    ) -> SimulationResult:
        if decision_enabled and (policy is None or fallback_policy is None):
            raise ValueError(
                "policy and fallback_policy are required when decision_enabled is True"
            )
        for maybe_llm_policy in (policy, fallback_policy):
            configure_run_context = getattr(maybe_llm_policy, "configure_run_context", None)
            if configure_run_context is not None:
                configure_run_context(run_identity)

        state = copy.deepcopy(initial_state)
        events_by_day = {day_events.day: day_events for day_events in event_tape.days}

        initial_committed_total = sum(
            sum(products.values()) for products in state.inventory.values()
        ) + sum(
            shipment.quantity
            for shipment in state.shipments.values()
            if shipment.status is not ShipmentStatus.DELIVERED
        )
        total_released = 0
        delivered_count = sum(
            1
            for shipment in state.shipments.values()
            if shipment.status is ShipmentStatus.DELIVERED
        )

        max_day = horizon_day + drain_days
        daily_metrics: list[DailyMetrics] = []
        early_stopped = False

        day = start_day
        while day <= max_day:
            if day not in events_by_day:
                raise SimulationInvariantError(
                    f"event tape has no entry for day {day} (requested range "
                    f"{start_day}..{max_day})"
                )
            day_events = events_by_day[day]

            metrics, delivered_count, total_released = self._process_day(
                state,
                day_events,
                decision_enabled,
                event_tape,
                run_identity,
                reroute_cost_per_unit,
                expedite_premium_per_unit,
                delivered_count,
                initial_committed_total,
                total_released,
                policy,
                fallback_policy,
                mean_daily_demand,
                decision_trace_sink,
                llm_interaction_sink,
            )
            daily_metrics.append(metrics)

            if day > horizon_day and _all_resolved(state):
                early_stopped = True
                break
            day += 1

        terminated_with_unresolved_state = False
        if not early_stopped and not _all_resolved(state):
            charge_terminal_cost(state, terminal_penalty_days)
            terminated_with_unresolved_state = True

        return SimulationResult(
            experiment_id=run_identity.experiment_id,
            scenario_id=run_identity.scenario_id,
            replication=run_identity.replication,
            policy_name=run_identity.policy_name,
            run_kind=run_identity.run_kind,
            final_day=state.day,
            final_state=state,
            daily_metrics=tuple(daily_metrics),
            terminated_with_unresolved_state=terminated_with_unresolved_state,
        )

    def _process_day(
        self,
        state: SimulationState,
        day_events: DayEvents,
        decision_enabled: bool,
        event_tape: EventTape,
        run_identity: RunIdentity,
        reroute_cost_per_unit: float,
        expedite_premium_per_unit: float,
        previous_delivered_count: int,
        initial_committed_total: int,
        total_released_before_today: int,
        policy: Policy | None,
        fallback_policy: Policy | None,
        mean_daily_demand: float,
        decision_trace_sink: list[DecisionTraceEntry] | None,
        llm_interaction_sink: list[LLMInteractionResult] | None,
    ) -> tuple[DailyMetrics, int, int]:
        # 1. Begin day.
        state.day = day_events.day
        transition.reset_daily_capacity_usage(state)

        # 2. Reveal information.
        state.known_shock_ids.update(day_events.newly_known_shock_ids)

        # 3. Apply physical shocks.
        transition.apply_shock_operational_state(state, event_tape.shocks)
        self._assert_active_shocks_match_day(state, event_tape.shocks)

        # 4. Process arrivals.
        transition.process_due_arrivals(state)

        # 5. Release scheduled shipments. A release may defer instead of
        # succeeding (V2 §V2.3.7), so total_released only grows by what
        # actually entered the system today, not by what was merely scheduled.
        released_today = transition.release_shipments(state, day_events.shipment_release_events)
        total_released = total_released_before_today + released_today

        # 6. Realize and fulfil demand.
        demand_result = transition.fulfil_backlog_and_demand(
            state, day_events.demand_events
        )

        # 7-10. Build decision set, consult the policy, resolve fallback, apply it.
        costs_before = _cost_snapshot(state.costs)
        if decision_enabled:
            assert policy is not None
            assert fallback_policy is not None
            known_shocks = tuple(
                shock for shock in event_tape.shocks if shock.shock_id in state.known_shock_ids
            )
            self._resolve_triggered_shipments(
                state,
                known_shocks,
                policy,
                fallback_policy,
                mean_daily_demand,
                reroute_cost_per_unit,
                expedite_premium_per_unit,
                run_identity,
                decision_trace_sink,
                llm_interaction_sink,
            )

        # 11. Allocate departures.
        transition.allocate_departures(state, day_events.edge_extra_delay_days)

        # 12. Charge end-of-day costs.
        transition.charge_end_of_day_costs(state)
        costs_after = state.costs

        # 13. Record metrics and assert invariants.
        metrics = transition.record_daily_metrics(
            state,
            experiment_id=run_identity.experiment_id,
            scenario_id=run_identity.scenario_id,
            replication=run_identity.replication,
            policy_name=run_identity.policy_name,
            run_kind=run_identity.run_kind,
            demand_result=demand_result,
            daily_transport_cost=costs_after.transport - costs_before.transport,
            daily_reroute_cost=costs_after.reroute - costs_before.reroute,
            daily_expedite_cost=costs_after.expedite - costs_before.expedite,
            daily_holding_cost=costs_after.holding - costs_before.holding,
            daily_backlog_cost=costs_after.backlog - costs_before.backlog,
            daily_late_cost=costs_after.late - costs_before.late,
        )
        delivered_count = self._assert_invariants(
            state, previous_delivered_count, initial_committed_total, total_released
        )
        return metrics, delivered_count, total_released

    def _resolve_triggered_shipments(
        self,
        state: SimulationState,
        known_shocks: tuple[Shock, ...],
        policy: Policy,
        fallback_policy: Policy,
        mean_daily_demand: float,
        reroute_cost_per_unit: float,
        expedite_premium_per_unit: float,
        run_identity: RunIdentity,
        decision_trace_sink: list[DecisionTraceEntry] | None,
        llm_interaction_sink: list[LLMInteractionResult] | None,
    ) -> None:
        """Build each triggered shipment's observation from this same
        pre-action state, consult the policy, resolve fallback if it
        abstains or proposes something invalid, and collect the resulting
        executed actions for transition.py to apply.

        `known_shocks` must already be restricted to shocks in
        `state.known_shock_ids` — the caller in `_process_day` does this
        filtering, since this is the one path that
        feeds shock information to policy-facing code. `apply_shock_operational_state`
        and `_assert_active_shocks_match_day` intentionally still use the
        event tape's full, unfiltered shock list: physical effects apply
        regardless of whether a shock has been revealed yet.
        """
        triggered = transition.identify_shipments_requiring_decision(state, known_shocks)
        decisions: dict[str, tuple[DecisionAction, Route | None]] = {}

        for shipment_id in triggered:
            observation = build_observation(
                state,
                shipment_id,
                known_shocks,
                mean_daily_demand,
                reroute_cost_per_unit,
                expedite_premium_per_unit,
            )
            record = make_decision_record(policy, observation)
            if llm_interaction_sink is not None and record.llm_interaction is not None:
                llm_interaction_sink.append(record.llm_interaction)
            resolution = resolve_action(
                record.proposed_action, observation, state, fallback_policy
            )

            if not resolution.proposal_validation.is_valid:
                state.service.invalid_action_count += 1
            if record.proposed_action.action_type is ActionType.ABSTAIN:
                state.service.abstention_count += 1
            if resolution.fallback_invoked:
                state.service.fallback_count += 1

            if decision_trace_sink is not None:
                decision_trace_sink.append(
                    DecisionTraceEntry(
                        run_identity=run_identity,
                        day=state.day,
                        shipment_id=shipment_id,
                        observation=observation,
                        observation_hash=compute_observation_hash(observation),
                        proposed_action=record.proposed_action,
                        proposal_validation=resolution.proposal_validation,
                        fallback_invoked=resolution.fallback_invoked,
                        fallback_action=resolution.fallback_action,
                        fallback_validation=resolution.fallback_validation,
                        executed_action=resolution.executed_action,
                        decision_latency_ms=record.decision_latency_ms,
                    )
                )

            executed_action = resolution.executed_action
            route = (
                self._route_for(observation, executed_action.route_id)
                if executed_action.route_id is not None
                else None
            )
            decisions[shipment_id] = (executed_action, route)

        transition.apply_validated_actions(
            state, decisions, reroute_cost_per_unit, expedite_premium_per_unit
        )

    def _route_for(self, observation: DecisionObservation, route_id: str) -> Route:
        for option in observation.route_options:
            if option.route_id == route_id:
                return Route(
                    route_id=option.route_id,
                    edge_ids=option.edge_ids,
                    node_ids=option.node_ids,
                    contains_emergency_edge=option.contains_emergency_edge,
                )
        raise SimulationInvariantError(
            f"executed action references route {route_id!r}, which is not one of the "
            f"observation's approved route options"
        )

    def _assert_active_shocks_match_day(
        self, state: SimulationState, shocks: tuple[Shock, ...]
    ) -> None:
        expected = {
            shock.shock_id
            for shock in shocks
            if shock.physical_start_day <= state.day <= shock.physical_end_day
        }
        if state.active_shock_ids != expected:
            raise SimulationInvariantError(
                f"day {state.day}: active_shock_ids {state.active_shock_ids} does not match "
                f"expected {expected}"
            )

    def _assert_invariants(
        self,
        state: SimulationState,
        previous_delivered_count: int,
        initial_committed_total: int,
        total_released: int,
    ) -> int:
        for node_id, products in state.inventory.items():
            for product_id, quantity in products.items():
                if quantity < 0:
                    raise SimulationInvariantError(
                        f"day {state.day}: negative inventory at {node_id}/{product_id}: {quantity}"
                    )
        for node_id, products in state.backlog.items():
            for product_id, quantity in products.items():
                if quantity < 0:
                    raise SimulationInvariantError(
                        f"day {state.day}: negative backlog at {node_id}/{product_id}: {quantity}"
                    )
        for field_name in (
            "transport",
            "reroute",
            "expedite",
            "holding",
            "backlog",
            "late",
            "terminal",
        ):
            if getattr(state.costs, field_name) < 0:
                raise SimulationInvariantError(
                    f"day {state.day}: negative cost component {field_name}"
                )

        delivered_count = 0
        for shipment_id, shipment in state.shipments.items():
            if shipment.quantity <= 0:
                raise SimulationInvariantError(
                    f"day {state.day}: shipment {shipment_id} has non-positive quantity"
                )
            if shipment.status is ShipmentStatus.AT_NODE:
                self._assert_at_node_contract(state, shipment_id, shipment)
            elif shipment.status is ShipmentStatus.IN_TRANSIT:
                self._assert_in_transit_contract(state, shipment_id, shipment)
            else:
                delivered_count += 1
                self._assert_delivered_contract(state, shipment_id, shipment)

        if delivered_count < previous_delivered_count:
            raise SimulationInvariantError(
                f"day {state.day}: delivered shipment count decreased"
            )

        for edge_id, used in state.daily_edge_used_capacity.items():
            edge = state.network_definition.get_edge(edge_id)
            edge_state = state.edge_operational_state[edge_id]
            if used > get_effective_edge_capacity(edge, edge_state):
                raise SimulationInvariantError(
                    f"day {state.day}: edge {edge_id} over capacity"
                )

        for node_id, used in state.daily_node_used_processing.items():
            node = state.network_definition.get_node(node_id)
            node_state = state.node_operational_state[node_id]
            effective = int(
                node.processing_capacity * node_state.processing_capacity_multiplier
            )
            if used > effective:
                raise SimulationInvariantError(
                    f"day {state.day}: node {node_id} over processing capacity"
                )

        self._assert_product_balance(state, initial_committed_total, total_released)
        return delivered_count

    def _assert_at_node_contract(
        self, state: SimulationState, shipment_id: str, shipment: Shipment
    ) -> None:
        if shipment.current_node_id is None or shipment.current_edge_id is not None:
            raise SimulationInvariantError(
                f"day {state.day}: shipment {shipment_id} violates AT_NODE field contract"
            )
        remaining = shipment.planned_route_edge_ids[shipment.next_edge_index :]
        if remaining:
            try:
                route_from_edge_ids(state, remaining)
            except RoutingError as exc:
                raise SimulationInvariantError(
                    f"day {state.day}: shipment {shipment_id} has a discontinuous route: {exc}"
                ) from exc

    def _assert_in_transit_contract(
        self, state: SimulationState, shipment_id: str, shipment: Shipment
    ) -> None:
        if (
            shipment.current_node_id is not None
            or shipment.current_edge_id is None
            or shipment.edge_arrival_day is None
        ):
            raise SimulationInvariantError(
                f"day {state.day}: shipment {shipment_id} violates IN_TRANSIT field contract"
            )
        if shipment.edge_entry_day == state.day:
            edge_state = state.edge_operational_state[shipment.current_edge_id]
            if not edge_state.available:
                raise SimulationInvariantError(
                    f"day {state.day}: shipment {shipment_id} entered unavailable edge "
                    f"{shipment.current_edge_id}"
                )

    def _assert_delivered_contract(
        self, state: SimulationState, shipment_id: str, shipment: Shipment
    ) -> None:
        if (
            shipment.current_node_id != shipment.destination_node_id
            or shipment.current_edge_id is not None
            or shipment.delivered_day is None
        ):
            raise SimulationInvariantError(
                f"day {state.day}: shipment {shipment_id} violates DELIVERED field contract"
            )

    def _assert_product_balance(
        self, state: SimulationState, initial_committed_total: int, total_released: int
    ) -> None:
        current_inventory_total = sum(
            sum(products.values()) for products in state.inventory.values()
        )
        non_delivered_total = sum(
            shipment.quantity
            for shipment in state.shipments.values()
            if shipment.status is not ShipmentStatus.DELIVERED
        )
        consumed_total = (
            state.service.same_day_fulfilled_units
            + state.service.backlog_fulfilled_units
        )
        lhs = initial_committed_total + total_released
        rhs = current_inventory_total + non_delivered_total + consumed_total
        if lhs != rhs:
            raise SimulationInvariantError(
                f"day {state.day}: product balance violated: {lhs} (committed+released) != "
                f"{rhs} (inventory+in-system+consumed)"
            )
