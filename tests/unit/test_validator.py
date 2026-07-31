"""Unit tests for decisions/validator.py, and decisions/observation.py.

Inside tests/unit, this file checks every ValidationCode decisions/validator.py
can produce, plus the valid WAIT/REROUTE/EXPEDITE/ABSTAIN cases from CLAUDE.md
section 11.11 and section 30.7. There is no separate observation test file in
the approved repository structure (CLAUDE.md section 7), so the observation
builder's shipment/destination/shock/route-option/allowed-action assembly is
also exercised here, since every validator test needs one anyway. It does not
test simulation/routing.py's own route-estimate math, which tests/unit/
test_routing.py already covers.
"""

from __future__ import annotations

import json
from dataclasses import replace
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
    observation_to_canonical_dict,
)
from supply_chain_simulator.decisions.validator import validate_action
from supply_chain_simulator.domain.actions import (
    ActionType,
    DecisionAction,
    ReasonCode,
    ValidationCode,
)
from supply_chain_simulator.domain.events import Shock, ShockType, TargetType
from supply_chain_simulator.domain.state import (
    Shipment,
    ShipmentStatus,
    SimulationState,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_NETWORK_CONFIG = REPO_ROOT / "tests/fixtures/tiny_network.yaml"

REROUTE_COST_PER_UNIT = 1.00
EXPEDITE_PREMIUM_PER_UNIT = 2.00
MEAN_DAILY_DEMAND = 5.0


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
    edge_arrival_day: int | None = None,
    destination_node_id: str = "plant_1",
    planned_route_edge_ids: tuple[str, ...] = ("supplier_to_hub", "hub_to_plant"),
    next_edge_index: int = 0,
    quantity: int = 5,
    due_day: int = 100,
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
        edge_entry_day=None,
        edge_arrival_day=edge_arrival_day,
        reroute_count=0,
        expedite_count=0,
        capacity_wait_days=0,
        delivered_day=delivered_day,
    )


def _observation(
    state: SimulationState, shipment_id: str, known_shocks: tuple[Shock, ...] = ()
) -> DecisionObservation:
    return build_observation(
        state,
        shipment_id,
        known_shocks,
        MEAN_DAILY_DEMAND,
        REROUTE_COST_PER_UNIT,
        EXPEDITE_PREMIUM_PER_UNIT,
    )


def _route_option(route_id: str, contains_emergency_edge: bool = False) -> RouteOption:
    return RouteOption(
        route_id=route_id,
        edge_ids=tuple(route_id.split("__")),
        node_ids=(),
        contains_emergency_edge=contains_emergency_edge,
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


def _minimal_observation(shipment_id: str) -> DecisionObservation:
    """A mostly-placeholder observation used only to exercise shipment-identity
    and shipment-status validation, which never look past `.shipment.shipment_id`."""
    return DecisionObservation(
        observation_id=f"obs_test_{shipment_id}",
        day=1,
        shipment=ShipmentContext(
            shipment_id=shipment_id,
            product_id="widget",
            quantity=5,
            current_node_id="supplier_1",
            destination_node_id="plant_1",
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
        current_plan=_route_option("placeholder"),
        route_options=(),
        allowed_actions=(ActionType.WAIT, ActionType.ABSTAIN),
    )


def _bypass_action(
    shipment_id: str,
    action_type: ActionType,
    route_id: str | None,
    rationale: str = "test rationale",
) -> DecisionAction:
    """Constructs a DecisionAction while bypassing its own __post_init__.

    DecisionAction's constructor already refuses the exact shapes that
    INVALID_ACTION_SCHEMA and ROUTE_REQUIRED exist to catch, so those codes
    can only be reached by a validator that does not simply trust its caller
    (CLAUDE.md's "never silently repair an invalid LLM route"). This bypass
    is how the tests reach those defense-in-depth branches.
    """
    action = object.__new__(DecisionAction)
    object.__setattr__(action, "shipment_id", shipment_id)
    object.__setattr__(action, "action_type", action_type)
    object.__setattr__(action, "route_id", route_id)
    object.__setattr__(action, "reason_code", ReasonCode.LOWER_ESTIMATED_COST)
    object.__setattr__(action, "rationale", rationale)
    return action


class TestBuildObservation:
    def test_shipment_context_fields(self) -> None:
        state = _tiny_state(day=10)
        state.shipments["s1"] = _shipment(shipment_id="s1", due_day=30)
        observation = _observation(state, "s1")
        assert observation.day == 10
        assert observation.shipment.shipment_id == "s1"
        assert observation.shipment.current_node_id == "supplier_1"
        assert observation.shipment.destination_node_id == "plant_1"
        assert observation.shipment.days_until_due == 20
        assert observation.shipment.remaining_route_edge_ids == (
            "supplier_to_hub",
            "hub_to_plant",
        )

    def test_destination_context_from_inventory_and_backlog(self) -> None:
        state = _tiny_state(day=1)
        state.inventory["plant_1"]["widget"] = 12
        state.backlog["plant_1"]["widget"] = 3
        state.shipments["s1"] = _shipment(shipment_id="s1")
        observation = _observation(state, "s1")
        assert observation.destination.inventory_on_hand == 12
        assert observation.destination.backlog_units == 3
        assert observation.destination.mean_daily_demand == MEAN_DAILY_DEMAND
        assert observation.destination.days_of_supply == 12 / MEAN_DAILY_DEMAND

    def test_days_of_supply_is_none_when_mean_demand_zero(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1")
        observation = build_observation(
            state, "s1", (), 0.0, REROUTE_COST_PER_UNIT, EXPEDITE_PREMIUM_PER_UNIT
        )
        assert observation.destination.days_of_supply is None

    def test_relevant_shocks_filters_by_route_membership(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1")
        on_route = Shock(
            shock_id="on_route",
            shock_type=ShockType.EDGE_LEAD_TIME_INCREASE,
            target_type=TargetType.EDGE,
            target_id="hub_to_plant",
            physical_start_day=1,
            physical_end_day=5,
            information_day=1,
            lead_time_multiplier=2.0,
        )
        off_route = Shock(
            shock_id="off_route",
            shock_type=ShockType.EDGE_LEAD_TIME_INCREASE,
            target_type=TargetType.EDGE,
            target_id="supplier_to_plant_air",
            physical_start_day=1,
            physical_end_day=5,
            information_day=1,
            lead_time_multiplier=2.0,
        )
        observation = _observation(state, "s1", known_shocks=(on_route, off_route))
        assert [shock.shock_id for shock in observation.relevant_shocks] == ["on_route"]
        assert observation.relevant_shocks[0].known_effect == "lead time multiplier 2.0"

    def test_allowed_actions_include_reroute_and_expedite(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1")
        observation = _observation(state, "s1")
        assert observation.allowed_actions == (
            ActionType.WAIT,
            ActionType.REROUTE,
            ActionType.EXPEDITE,
            ActionType.ABSTAIN,
        )
        route_ids = {option.route_id for option in observation.route_options}
        assert "supplier_to_hub__hub_to_plant" in route_ids
        assert "supplier_to_plant_air" in route_ids

    def test_current_plan_reports_non_dispatchable_for_malformed_route(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(
            shipment_id="s1",
            planned_route_edge_ids=("supplier_to_hub", "hub_to_plant"),
            next_edge_index=2,
        )
        observation = _observation(state, "s1")
        assert observation.current_plan.currently_dispatchable is False
        assert observation.current_plan.estimated_total_cost is None

    def test_raises_when_shipment_not_at_node(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(
            shipment_id="s1",
            status=ShipmentStatus.IN_TRANSIT,
            current_node_id=None,
            current_edge_id="supplier_to_hub",
            edge_arrival_day=5,
        )
        try:
            _observation(state, "s1")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    def test_known_effect_describes_every_shock_type(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1")
        cases = [
            (
                ShockType.NODE_CLOSURE,
                TargetType.NODE,
                "supplier_1",
                {},
                "node unavailable",
            ),
            (
                ShockType.EDGE_CLOSURE,
                TargetType.EDGE,
                "supplier_to_hub",
                {},
                "edge unavailable",
            ),
            (
                ShockType.NODE_CAPACITY_REDUCTION,
                TargetType.NODE,
                "supplier_1",
                {"capacity_multiplier": 0.5},
                "processing capacity multiplier 0.5",
            ),
            (
                ShockType.EDGE_CAPACITY_REDUCTION,
                TargetType.EDGE,
                "supplier_to_hub",
                {"capacity_multiplier": 0.5},
                "capacity multiplier 0.5",
            ),
            (
                ShockType.EDGE_COST_INCREASE,
                TargetType.EDGE,
                "supplier_to_hub",
                {"cost_multiplier": 1.5},
                "cost multiplier 1.5",
            ),
        ]
        for shock_type, target_type, target_id, multipliers, expected_effect in cases:
            shock = Shock(
                shock_id="s",
                shock_type=shock_type,
                target_type=target_type,
                target_id=target_id,
                physical_start_day=1,
                physical_end_day=5,
                information_day=1,
                **multipliers,
            )
            observation = _observation(state, "s1", known_shocks=(shock,))
            assert observation.relevant_shocks[0].known_effect == expected_effect

    def test_canonical_dict_is_json_serializable_and_matches_fields(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1")
        shock = Shock(
            shock_id="s",
            shock_type=ShockType.NODE_CLOSURE,
            target_type=TargetType.NODE,
            target_id="supplier_1",
            physical_start_day=1,
            physical_end_day=5,
            information_day=1,
        )
        observation = _observation(state, "s1", known_shocks=(shock,))
        canonical = observation_to_canonical_dict(observation)
        encoded = json.dumps(canonical, sort_keys=True)
        decoded = json.loads(encoded)
        assert decoded["observation_id"] == observation.observation_id
        assert decoded["shipment"]["shipment_id"] == "s1"
        assert decoded["allowed_actions"] == [
            action.value for action in observation.allowed_actions
        ]
        assert len(decoded["route_options"]) == len(observation.route_options)
        assert decoded["relevant_shocks"][0]["shock_id"] == "s"


class TestValidateActionValidCases:
    def test_valid_wait(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1")
        observation = _observation(state, "s1")
        action = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.WAIT,
            route_id=None,
            reason_code=ReasonCode.NO_FEASIBLE_ALTERNATIVE,
            rationale="wait",
        )
        result = validate_action(action, observation, state)
        assert result.code is ValidationCode.VALID
        assert result.is_valid is True

    def test_valid_reroute(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1")
        observation = _observation(state, "s1")
        action = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.REROUTE,
            route_id="supplier_to_hub__hub_to_plant",
            reason_code=ReasonCode.LOWER_ESTIMATED_COST,
            rationale="reroute",
        )
        result = validate_action(action, observation, state)
        assert result.code is ValidationCode.VALID

    def test_valid_expedite(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1")
        observation = _observation(state, "s1")
        action = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.EXPEDITE,
            route_id="supplier_to_plant_air",
            reason_code=ReasonCode.REDUCE_LATENESS,
            rationale="expedite",
        )
        result = validate_action(action, observation, state)
        assert result.code is ValidationCode.VALID

    def test_valid_abstain(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1")
        observation = _observation(state, "s1")
        action = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.ABSTAIN,
            route_id=None,
            reason_code=ReasonCode.INSUFFICIENT_INFORMATION,
            rationale="abstain",
        )
        result = validate_action(action, observation, state)
        assert result.code is ValidationCode.VALID


class TestValidateActionShipmentChecks:
    def test_action_shipment_mismatch(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1")
        observation = _observation(state, "s1")
        action = DecisionAction(
            shipment_id="s2",
            action_type=ActionType.WAIT,
            route_id=None,
            reason_code=ReasonCode.NO_FEASIBLE_ALTERNATIVE,
            rationale="mismatch",
        )
        result = validate_action(action, observation, state)
        assert result.code is ValidationCode.ACTION_SHIPMENT_MISMATCH

    def test_shipment_not_found(self) -> None:
        state = _tiny_state(day=1)
        observation = _minimal_observation("ghost")
        action = DecisionAction(
            shipment_id="ghost",
            action_type=ActionType.WAIT,
            route_id=None,
            reason_code=ReasonCode.NO_FEASIBLE_ALTERNATIVE,
            rationale="missing",
        )
        result = validate_action(action, observation, state)
        assert result.code is ValidationCode.SHIPMENT_NOT_FOUND

    def test_shipment_already_delivered(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(
            shipment_id="s1",
            status=ShipmentStatus.DELIVERED,
            current_node_id="plant_1",
            destination_node_id="plant_1",
            delivered_day=1,
        )
        observation = _minimal_observation("s1")
        action = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.WAIT,
            route_id=None,
            reason_code=ReasonCode.NO_FEASIBLE_ALTERNATIVE,
            rationale="delivered",
        )
        result = validate_action(action, observation, state)
        assert result.code is ValidationCode.SHIPMENT_ALREADY_DELIVERED

    def test_shipment_not_at_node_when_in_transit(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(
            shipment_id="s1",
            status=ShipmentStatus.IN_TRANSIT,
            current_node_id=None,
            current_edge_id="supplier_to_hub",
            edge_arrival_day=5,
        )
        observation = _minimal_observation("s1")
        action = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.WAIT,
            route_id=None,
            reason_code=ReasonCode.NO_FEASIBLE_ALTERNATIVE,
            rationale="in transit",
        )
        result = validate_action(action, observation, state)
        assert result.code is ValidationCode.SHIPMENT_NOT_AT_NODE


class TestValidateActionSchemaAndRouteRequirement:
    def test_invalid_schema_rationale_too_long(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1")
        observation = _observation(state, "s1")
        action = _bypass_action("s1", ActionType.WAIT, None, rationale="x" * 301)
        result = validate_action(action, observation, state)
        assert result.code is ValidationCode.INVALID_ACTION_SCHEMA

    def test_invalid_schema_extraneous_route_for_wait(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1")
        observation = _observation(state, "s1")
        action = _bypass_action("s1", ActionType.WAIT, "supplier_to_hub__hub_to_plant")
        result = validate_action(action, observation, state)
        assert result.code is ValidationCode.INVALID_ACTION_SCHEMA

    def test_route_required_for_reroute(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1")
        observation = _observation(state, "s1")
        action = _bypass_action("s1", ActionType.REROUTE, None)
        result = validate_action(action, observation, state)
        assert result.code is ValidationCode.ROUTE_REQUIRED

    def test_route_required_for_expedite(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1")
        observation = _observation(state, "s1")
        action = _bypass_action("s1", ActionType.EXPEDITE, None)
        result = validate_action(action, observation, state)
        assert result.code is ValidationCode.ROUTE_REQUIRED


class TestValidateActionRouteChecks:
    def test_route_not_found_for_garbage_route_id(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1")
        observation = _observation(state, "s1")
        action = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.REROUTE,
            route_id="no_such_edge",
            reason_code=ReasonCode.LOWER_ESTIMATED_COST,
            rationale="garbage route",
        )
        result = validate_action(action, observation, state)
        assert result.code is ValidationCode.ROUTE_NOT_FOUND

    def test_route_not_allowed_when_not_offered(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1")
        observation = replace(_observation(state, "s1"), route_options=())
        action = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.REROUTE,
            route_id="supplier_to_hub__hub_to_plant",
            reason_code=ReasonCode.LOWER_ESTIMATED_COST,
            rationale="not offered this round",
        )
        result = validate_action(action, observation, state)
        assert result.code is ValidationCode.ROUTE_NOT_ALLOWED

    def test_route_wrong_origin(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1")
        base = _observation(state, "s1")
        observation = replace(
            base, route_options=(*base.route_options, _route_option("hub_to_plant"))
        )
        action = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.REROUTE,
            route_id="hub_to_plant",
            reason_code=ReasonCode.LOWER_ESTIMATED_COST,
            rationale="wrong origin",
        )
        result = validate_action(action, observation, state)
        assert result.code is ValidationCode.ROUTE_WRONG_ORIGIN

    def test_route_wrong_destination(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1")
        base = _observation(state, "s1")
        observation = replace(
            base, route_options=(*base.route_options, _route_option("supplier_to_hub"))
        )
        action = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.REROUTE,
            route_id="supplier_to_hub",
            reason_code=ReasonCode.LOWER_ESTIMATED_COST,
            rationale="wrong destination",
        )
        result = validate_action(action, observation, state)
        assert result.code is ValidationCode.ROUTE_WRONG_DESTINATION

    def test_route_uses_unavailable_component(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1")
        observation = _observation(state, "s1")
        state.edge_operational_state["hub_to_plant"].available = False
        action = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.REROUTE,
            route_id="supplier_to_hub__hub_to_plant",
            reason_code=ReasonCode.LOWER_ESTIMATED_COST,
            rationale="edge now closed",
        )
        result = validate_action(action, observation, state)
        assert result.code is ValidationCode.ROUTE_USES_UNAVAILABLE_COMPONENT

    def test_route_static_capacity_too_small(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1", quantity=25)
        base = _observation(state, "s1")
        observation = replace(
            base,
            route_options=(
                *base.route_options,
                _route_option("supplier_to_hub__hub_to_plant"),
            ),
        )
        action = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.REROUTE,
            route_id="supplier_to_hub__hub_to_plant",
            reason_code=ReasonCode.LOWER_ESTIMATED_COST,
            rationale="too big for the lane",
        )
        result = validate_action(action, observation, state)
        assert result.code is ValidationCode.ROUTE_STATIC_CAPACITY_TOO_SMALL

    def test_reroute_uses_emergency_edge(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1")
        observation = _observation(state, "s1")
        action = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.REROUTE,
            route_id="supplier_to_plant_air",
            reason_code=ReasonCode.LOWER_ESTIMATED_COST,
            rationale="should be expedite",
        )
        result = validate_action(action, observation, state)
        assert result.code is ValidationCode.REROUTE_USES_EMERGENCY_EDGE

    def test_expedite_has_no_emergency_edge(self) -> None:
        state = _tiny_state(day=1)
        state.shipments["s1"] = _shipment(shipment_id="s1")
        observation = _observation(state, "s1")
        action = DecisionAction(
            shipment_id="s1",
            action_type=ActionType.EXPEDITE,
            route_id="supplier_to_hub__hub_to_plant",
            reason_code=ReasonCode.LOWER_ESTIMATED_COST,
            rationale="should be reroute",
        )
        result = validate_action(action, observation, state)
        assert result.code is ValidationCode.EXPEDITE_HAS_NO_EMERGENCY_EDGE
