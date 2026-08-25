"""Unit tests for the heuristic's candidate construction and tie-break rule; also covers policies/base.py and policies/fallback.py, which have no test file of their own."""

from __future__ import annotations

from pathlib import Path

from supply_chain_simulator.data_io.loaders import (
    build_initial_state,
    build_network_definition,
    load_network_config,
)
from supply_chain_simulator.decisions.observation import (
    DecisionObservation,
    DestinationContext,
    RouteOption,
    ShipmentContext,
    build_observation,
)
from supply_chain_simulator.domain.actions import ActionType, DecisionAction, ReasonCode
from supply_chain_simulator.domain.state import (
    Shipment,
    ShipmentStatus,
    SimulationState,
)
from supply_chain_simulator.policies.base import make_decision_record
from supply_chain_simulator.policies.fallback import (
    HeuristicFallbackPolicy,
    WaitFallbackPolicy,
    resolve_action,
)
from supply_chain_simulator.policies.heuristic import HeuristicPolicy

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_NETWORK_CONFIG = REPO_ROOT / "tests/fixtures/tiny_network.yaml"

EXPEDITE_TRIGGER_LATENESS_DAYS = 2
COST_TOLERANCE = 1e-9


def _policy() -> HeuristicPolicy:
    return HeuristicPolicy(
        expedite_trigger_lateness_days=EXPEDITE_TRIGGER_LATENESS_DAYS,
        cost_tolerance=COST_TOLERANCE,
    )


def _route_option(
    route_id: str,
    estimated_total_cost: float,
    contains_emergency_edge: bool = False,
) -> RouteOption:
    return RouteOption(
        route_id=route_id,
        edge_ids=(route_id,),
        node_ids=("origin", "destination"),
        contains_emergency_edge=contains_emergency_edge,
        estimated_lead_time_days=1,
        estimated_arrival_day=2,
        estimated_lateness_days=0,
        estimated_transport_cost=estimated_total_cost,
        estimated_action_cost=0.0,
        estimated_late_penalty=0.0,
        estimated_total_cost=estimated_total_cost,
        first_edge_remaining_capacity=100,
        currently_dispatchable=True,
    )


def _current_plan(
    estimated_total_cost: float | None, estimated_lateness_days: int | None = 0
) -> RouteOption:
    return RouteOption(
        route_id="current",
        edge_ids=("current_edge",),
        node_ids=("origin", "destination"),
        contains_emergency_edge=False,
        estimated_lead_time_days=1,
        estimated_arrival_day=2,
        estimated_lateness_days=estimated_lateness_days,
        estimated_transport_cost=estimated_total_cost,
        estimated_action_cost=0.0,
        estimated_late_penalty=0.0,
        estimated_total_cost=estimated_total_cost,
        first_edge_remaining_capacity=100,
        currently_dispatchable=estimated_total_cost is not None,
    )


def _observation(
    current_plan: RouteOption, route_options: tuple[RouteOption, ...]
) -> DecisionObservation:
    return DecisionObservation(
        observation_id="obs_test",
        day=1,
        shipment=ShipmentContext(
            shipment_id="s1",
            product_id="widget",
            quantity=5,
            current_node_id="origin",
            destination_node_id="destination",
            release_day=1,
            due_day=100,
            days_until_due=99,
            capacity_wait_days=0,
            remaining_route_edge_ids=(),
        ),
        destination=DestinationContext(
            inventory_on_hand=0,
            backlog_units=0,
            mean_daily_demand=5.0,
            days_of_supply=0.0,
        ),
        relevant_shocks=(),
        current_plan=current_plan,
        route_options=route_options,
        allowed_actions=(),
    )


class TestHeuristicPolicyName:
    def test_name_is_heuristic(self) -> None:
        assert _policy().name == "heuristic"


class TestHeuristicPolicyDecide:
    def test_wait_is_cheapest(self) -> None:
        observation = _observation(
            current_plan=_current_plan(10.0),
            route_options=(
                _route_option("reroute_1", 20.0),
                _route_option("expedite_1", 30.0, contains_emergency_edge=True),
            ),
        )
        action = _policy().decide(observation)
        assert action.action_type is ActionType.WAIT
        assert action.route_id is None
        assert action.reason_code is ReasonCode.LOWER_ESTIMATED_COST

    def test_reroute_is_cheapest(self) -> None:
        observation = _observation(
            current_plan=_current_plan(30.0),
            route_options=(
                _route_option("reroute_1", 10.0),
                _route_option("expedite_1", 20.0, contains_emergency_edge=True),
            ),
        )
        action = _policy().decide(observation)
        assert action.action_type is ActionType.REROUTE
        assert action.route_id == "reroute_1"

    def test_expedite_is_cheapest_and_eligible_via_lateness_trigger(self) -> None:
        observation = _observation(
            current_plan=_current_plan(
                50.0, estimated_lateness_days=3
            ),  # >= trigger (2)
            route_options=(
                _route_option("reroute_1", 20.0),
                _route_option("expedite_1", 10.0, contains_emergency_edge=True),
            ),
        )
        action = _policy().decide(observation)
        assert action.action_type is ActionType.EXPEDITE
        assert action.route_id == "expedite_1"
        assert action.reason_code is ReasonCode.REDUCE_LATENESS

    def test_expedite_cheapest_but_below_trigger_is_excluded(self) -> None:
        observation = _observation(
            current_plan=_current_plan(
                20.0, estimated_lateness_days=1
            ),  # < trigger (2)
            route_options=(
                _route_option("reroute_1", 15.0),
                _route_option("expedite_1", 5.0, contains_emergency_edge=True),
            ),
        )
        action = _policy().decide(observation)
        # expedite would be cheapest (5.0) but is ineligible; reroute (15.0) beats wait (20.0)
        assert action.action_type is ActionType.REROUTE
        assert action.route_id == "reroute_1"

    def test_expedite_eligible_when_no_reroute_alternative_exists(self) -> None:
        observation = _observation(
            current_plan=_current_plan(50.0, estimated_lateness_days=0),
            route_options=(
                _route_option("expedite_1", 10.0, contains_emergency_edge=True),
            ),
        )
        action = _policy().decide(observation)
        assert action.action_type is ActionType.EXPEDITE
        assert action.route_id == "expedite_1"
        assert action.reason_code is ReasonCode.NO_FEASIBLE_ALTERNATIVE

    def test_no_route_options_returns_wait(self) -> None:
        observation = _observation(current_plan=_current_plan(10.0), route_options=())
        action = _policy().decide(observation)
        assert action.action_type is ActionType.WAIT
        assert action.reason_code is ReasonCode.NO_FEASIBLE_ALTERNATIVE

    def test_exact_tie_prefers_wait_over_reroute(self) -> None:
        observation = _observation(
            current_plan=_current_plan(10.0),
            route_options=(_route_option("reroute_1", 10.0),),
        )
        action = _policy().decide(observation)
        assert action.action_type is ActionType.WAIT

    def test_exact_tie_prefers_reroute_over_expedite(self) -> None:
        observation = _observation(
            current_plan=_current_plan(50.0, estimated_lateness_days=3),
            route_options=(
                _route_option("reroute_1", 10.0),
                _route_option("expedite_1", 10.0, contains_emergency_edge=True),
            ),
        )
        action = _policy().decide(observation)
        assert action.action_type is ActionType.REROUTE
        assert action.route_id == "reroute_1"

    def test_tie_among_reroutes_prefers_lexicographically_smallest_route_id(
        self,
    ) -> None:
        observation = _observation(
            current_plan=_current_plan(50.0),
            route_options=(
                _route_option("reroute_b", 10.0),
                _route_option("reroute_a", 10.0),
            ),
        )
        action = _policy().decide(observation)
        assert action.route_id == "reroute_a"

    def test_wait_with_unknown_cost_is_treated_as_infinitely_costly(self) -> None:
        observation = _observation(
            current_plan=_current_plan(None, estimated_lateness_days=None),
            route_options=(_route_option("reroute_1", 15.0),),
        )
        action = _policy().decide(observation)
        assert action.action_type is ActionType.REROUTE

    def test_wait_with_unknown_cost_and_no_alternative_is_still_chosen(self) -> None:
        observation = _observation(
            current_plan=_current_plan(None, estimated_lateness_days=None),
            route_options=(),
        )
        action = _policy().decide(observation)
        assert action.action_type is ActionType.WAIT

    def test_decision_is_deterministic(self) -> None:
        observation = _observation(
            current_plan=_current_plan(10.0),
            route_options=(_route_option("reroute_1", 5.0),),
        )
        first = _policy().decide(observation)
        second = _policy().decide(observation)
        assert first == second


class TestMakeDecisionRecord:
    def test_captures_policy_name_proposed_action_and_nonnegative_latency(self) -> None:
        observation = _observation(current_plan=_current_plan(10.0), route_options=())
        record = make_decision_record(_policy(), observation)
        assert record.policy_name == "heuristic"
        assert record.observation_id == observation.observation_id
        assert record.proposed_action.action_type is ActionType.WAIT
        assert record.decision_latency_ms >= 0.0


def _tiny_state(day: int = 1) -> SimulationState:
    network_config = load_network_config(TINY_NETWORK_CONFIG)
    network_definition = build_network_definition(network_config)
    state = build_initial_state(network_definition, network_config)
    state.day = day
    return state


def _shipment(shipment_id: str = "s1") -> Shipment:
    return Shipment(
        shipment_id=shipment_id,
        product_id="widget",
        quantity=5,
        origin_node_id="supplier_1",
        destination_node_id="plant_1",
        release_day=1,
        due_day=100,
        planned_route_edge_ids=("supplier_to_hub", "hub_to_plant"),
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


class _AlwaysInvalidRerouteFallback:
    """A test double Policy whose fallback proposal is always invalid, used to
    exercise resolve_action's terminal-safe-WAIT path."""

    @property
    def name(self) -> str:
        return "always_invalid_reroute"

    def decide(self, observation: DecisionObservation) -> DecisionAction:
        return DecisionAction(
            shipment_id=observation.shipment.shipment_id,
            action_type=ActionType.REROUTE,
            route_id="no_such_route",
            reason_code=ReasonCode.LOWER_ESTIMATED_COST,
            rationale="always invalid, for testing",
        )


class TestWaitFallbackPolicy:
    def test_always_proposes_wait(self) -> None:
        policy = WaitFallbackPolicy()
        assert policy.name == "wait_fallback"
        state = _tiny_state()
        state.shipments["s1"] = _shipment()
        observation = build_observation(state, "s1", (), 5.0, 1.0, 2.0)
        action = policy.decide(observation)
        assert action.action_type is ActionType.WAIT
        assert action.route_id is None


class TestHeuristicFallbackPolicy:
    def test_delegates_to_wrapped_heuristic(self) -> None:
        policy = HeuristicFallbackPolicy(_policy())
        assert policy.name == "heuristic_fallback"
        state = _tiny_state()
        state.shipments["s1"] = _shipment()
        observation = build_observation(state, "s1", (), 5.0, 1.0, 2.0)
        action = policy.decide(observation)
        assert action == _policy().decide(observation)


class TestResolveAction:
    def test_valid_proposal_executes_directly_without_fallback(self) -> None:
        state = _tiny_state()
        state.shipments["s1"] = _shipment()
        observation = build_observation(state, "s1", (), 5.0, 1.0, 2.0)
        proposed = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.WAIT,
            route_id=None,
            reason_code=ReasonCode.LOWER_ESTIMATED_COST,
            rationale="wait",
        )
        resolution = resolve_action(proposed, observation, state, WaitFallbackPolicy())
        assert resolution.proposal_validation.is_valid is True
        assert resolution.fallback_invoked is False
        assert resolution.fallback_action is None
        assert resolution.executed_action == proposed

    def test_invalid_proposal_falls_back_to_configured_policy(self) -> None:
        state = _tiny_state()
        state.shipments["s1"] = _shipment()
        observation = build_observation(state, "s1", (), 5.0, 1.0, 2.0)
        proposed = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.REROUTE,
            route_id="garbage_route",
            reason_code=ReasonCode.LOWER_ESTIMATED_COST,
            rationale="invalid route",
        )
        resolution = resolve_action(proposed, observation, state, WaitFallbackPolicy())
        assert resolution.proposal_validation.is_valid is False
        assert resolution.fallback_invoked is True
        assert resolution.fallback_action is not None
        assert resolution.fallback_action.action_type is ActionType.WAIT
        assert resolution.fallback_validation is not None
        assert resolution.fallback_validation.is_valid is True
        assert resolution.executed_action.action_type is ActionType.WAIT

    def test_abstain_proposal_falls_back_even_though_it_validates(self) -> None:
        state = _tiny_state()
        state.shipments["s1"] = _shipment()
        observation = build_observation(state, "s1", (), 5.0, 1.0, 2.0)
        proposed = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.ABSTAIN,
            route_id=None,
            reason_code=ReasonCode.INSUFFICIENT_INFORMATION,
            rationale="abstain",
        )
        resolution = resolve_action(proposed, observation, state, WaitFallbackPolicy())
        assert resolution.proposal_validation.is_valid is True
        assert resolution.fallback_invoked is True
        assert resolution.executed_action.action_type is ActionType.WAIT

    def test_fallback_also_invalid_uses_terminal_safe_wait(self) -> None:
        state = _tiny_state()
        state.shipments["s1"] = _shipment()
        observation = build_observation(state, "s1", (), 5.0, 1.0, 2.0)
        proposed = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.ABSTAIN,
            route_id=None,
            reason_code=ReasonCode.INSUFFICIENT_INFORMATION,
            rationale="abstain",
        )
        resolution = resolve_action(
            proposed, observation, state, _AlwaysInvalidRerouteFallback()
        )
        assert resolution.fallback_invoked is True
        assert resolution.fallback_validation is not None
        assert resolution.fallback_validation.is_valid is False
        assert resolution.executed_action.action_type is ActionType.WAIT
        assert (
            resolution.executed_action.reason_code is ReasonCode.POLICY_OUTPUT_INVALID
        )
