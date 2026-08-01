"""Unit tests for policies/llm_agent.py.

Inside tests/unit, this file checks `LLMAgentPolicy.decide()`'s interpretation
of an `LLMClient`'s result into a `DecisionAction` (normal submission,
malformed submission, tool-limit, no-submission), `configure_run_context`'s
effect on the `DecisionKey` embedded in `last_interaction`, each of the five
tools' local dispatch via `make_tool_executor`, and the versioned prompt
hash's stability. It drives `LLMAgentPolicy` against `FakeLLMClient` and a
real `build_observation(...)` off `tiny_network.yaml` (the same convention
tests/unit/test_heuristic.py and tests/unit/test_validator.py use), never a
real network call. It does not test integrations/llm_client.py's own
tool-loop mechanics, which tests/unit/test_llm_client.py covers.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from supply_chain_simulator.data_io.loaders import (
    build_initial_state,
    build_network_definition,
    load_network_config,
)
from supply_chain_simulator.decisions.observation import (
    DecisionObservation,
    build_observation,
    compute_observation_hash,
)
from supply_chain_simulator.domain.actions import ActionType, ReasonCode
from supply_chain_simulator.domain.state import (
    Shipment,
    ShipmentStatus,
    SimulationState,
)
from supply_chain_simulator.integrations.llm_client import FakeLLMClient
from supply_chain_simulator.policies.llm_agent import (
    SYSTEM_PROMPT,
    LLMAgentPolicy,
    compute_prompt_hash,
    make_tool_executor,
)
from supply_chain_simulator.simulation.engine import RunIdentity

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_NETWORK_CONFIG = REPO_ROOT / "tests/fixtures/tiny_network.yaml"

MEAN_DAILY_DEMAND = 5.0
REROUTE_COST_PER_UNIT = 1.0
EXPEDITE_PREMIUM_PER_UNIT = 2.0


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


def _observation(shipment_id: str = "s1") -> DecisionObservation:
    state = _tiny_state(day=1)
    state.shipments[shipment_id] = _shipment(shipment_id)
    return build_observation(
        state,
        shipment_id,
        (),
        MEAN_DAILY_DEMAND,
        REROUTE_COST_PER_UNIT,
        EXPEDITE_PREMIUM_PER_UNIT,
    )


def _run_identity() -> RunIdentity:
    return RunIdentity(
        experiment_id="exp",
        scenario_id="scenario",
        replication=1,
        policy_name="llm_agent",
        run_kind="DISRUPTED",
    )


def _policy(client: FakeLLMClient, *, configured: bool = True) -> LLMAgentPolicy:
    policy = LLMAgentPolicy(
        client=client,
        model="gpt-5.4-mini",
        temperature=0.0,
        max_tool_calls=8,
        max_output_tokens=1000,
        request_timeout_seconds=60,
        max_retries=3,
    )
    if configured:
        policy.configure_run_context(_run_identity())
    return policy


class TestLLMAgentPolicyName:
    def test_name_is_llm_agent(self) -> None:
        assert _policy(FakeLLMClient()).name == "llm_agent"


class TestLLMAgentPolicyDecide:
    def test_normal_submission_becomes_the_submitted_decision_action(self) -> None:
        observation = _observation()
        client = FakeLLMClient(
            submitted_action={
                "shipment_id": "s1",
                "action_type": "WAIT",
                "route_id": None,
                "reason_code": "LOWER_ESTIMATED_COST",
                "rationale": "wait it out",
            }
        )
        action = _policy(client).decide(observation)

        assert action.shipment_id == "s1"
        assert action.action_type is ActionType.WAIT
        assert action.route_id is None
        assert action.reason_code is ReasonCode.LOWER_ESTIMATED_COST
        assert action.rationale == "wait it out"

    def test_malformed_submission_becomes_abstain_policy_output_invalid(self) -> None:
        observation = _observation()
        client = FakeLLMClient(
            submitted_action={
                "shipment_id": "s1",
                "action_type": "WAIT",
                "route_id": "not_allowed_for_wait",  # WAIT requires route_id=None
                "reason_code": "LOWER_ESTIMATED_COST",
                "rationale": "bad",
            }
        )
        action = _policy(client).decide(observation)

        assert action.action_type is ActionType.ABSTAIN
        assert action.reason_code is ReasonCode.POLICY_OUTPUT_INVALID

    def test_submission_missing_a_field_becomes_abstain_policy_output_invalid(self) -> None:
        observation = _observation()
        client = FakeLLMClient(submitted_action={"shipment_id": "s1", "action_type": "WAIT"})
        action = _policy(client).decide(observation)

        assert action.action_type is ActionType.ABSTAIN
        assert action.reason_code is ReasonCode.POLICY_OUTPUT_INVALID

    def test_unknown_action_type_becomes_abstain_policy_output_invalid(self) -> None:
        observation = _observation()
        client = FakeLLMClient(
            submitted_action={
                "shipment_id": "s1",
                "action_type": "TELEPORT",
                "route_id": None,
                "reason_code": "LOWER_ESTIMATED_COST",
                "rationale": "n/a",
            }
        )
        action = _policy(client).decide(observation)

        assert action.action_type is ActionType.ABSTAIN
        assert action.reason_code is ReasonCode.POLICY_OUTPUT_INVALID

    def test_tool_limit_reached_becomes_abstain_tool_limit_reached(self) -> None:
        observation = _observation()
        client = FakeLLMClient(submitted_action=None, stop_reason="tool_limit_reached")
        action = _policy(client).decide(observation)

        assert action.action_type is ActionType.ABSTAIN
        assert action.reason_code is ReasonCode.TOOL_LIMIT_REACHED

    def test_no_submission_becomes_abstain_policy_output_invalid(self) -> None:
        observation = _observation()
        client = FakeLLMClient(submitted_action=None, stop_reason="no_submission")
        action = _policy(client).decide(observation)

        assert action.action_type is ActionType.ABSTAIN
        assert action.reason_code is ReasonCode.POLICY_OUTPUT_INVALID

    def test_abstain_rationale_never_exceeds_max_length(self) -> None:
        observation = _observation()
        client = FakeLLMClient(submitted_action=None, stop_reason="no_submission")
        action = _policy(client).decide(observation)
        assert len(action.rationale) <= 300

    def test_decide_before_configure_run_context_raises(self) -> None:
        observation = _observation()
        policy = _policy(FakeLLMClient(), configured=False)
        with pytest.raises(ValueError, match="configure_run_context"):
            policy.decide(observation)


class TestLLMAgentPolicyLastInteraction:
    def test_decision_key_matches_run_context_and_observation(self) -> None:
        observation = _observation()
        client = FakeLLMClient(
            submitted_action={
                "shipment_id": "s1",
                "action_type": "WAIT",
                "route_id": None,
                "reason_code": "LOWER_ESTIMATED_COST",
                "rationale": "wait",
            }
        )
        policy = _policy(client)
        policy.decide(observation)

        interaction = policy.last_interaction
        assert interaction is not None
        assert interaction.decision_key.experiment_id == "exp"
        assert interaction.decision_key.scenario_id == "scenario"
        assert interaction.decision_key.replication == 1
        assert interaction.decision_key.run_kind == "DISRUPTED"
        assert interaction.decision_key.day == observation.day
        assert interaction.decision_key.shipment_id == "s1"
        assert interaction.decision_key.observation_hash == compute_observation_hash(observation)

    def test_is_none_before_any_decision(self) -> None:
        assert _policy(FakeLLMClient()).last_interaction is None


class TestMakeToolExecutor:
    def test_get_shipment_context_matches_observation(self) -> None:
        observation = _observation()
        executor = make_tool_executor(observation)
        result = executor("get_shipment_context", {})
        assert result["shipment_id"] == "s1"
        assert result["quantity"] == 5

    def test_get_destination_context_matches_observation(self) -> None:
        observation = _observation()
        executor = make_tool_executor(observation)
        result = executor("get_destination_context", {})
        assert result["inventory_on_hand"] == observation.destination.inventory_on_hand
        assert result["mean_daily_demand"] == MEAN_DAILY_DEMAND

    def test_list_route_options_summarizes_every_option(self) -> None:
        observation = _observation()
        assert observation.route_options  # tiny network offers at least one route here
        executor = make_tool_executor(observation)
        result = executor("list_route_options", {})
        summaries = result["route_options"]
        assert isinstance(summaries, list)
        assert {summary["route_id"] for summary in summaries} == {
            option.route_id for option in observation.route_options
        }

    def test_inspect_route_returns_full_detail_for_a_known_route(self) -> None:
        observation = _observation()
        route_id = observation.route_options[0].route_id
        executor = make_tool_executor(observation)
        result = executor("inspect_route", {"route_id": route_id})
        assert result["route_id"] == route_id
        assert "estimated_total_cost" in result
        assert "currently_dispatchable" in result

    def test_inspect_route_reports_an_error_for_an_unapproved_route(self) -> None:
        observation = _observation()
        executor = make_tool_executor(observation)
        result = executor("inspect_route", {"route_id": "not_a_real_route"})
        assert "error" in result

    def test_unknown_tool_name_reports_an_error(self) -> None:
        observation = _observation()
        executor = make_tool_executor(observation)
        result = executor("not_a_real_tool", {})
        assert "error" in result

    def test_submit_action_is_not_dispatched_locally(self) -> None:
        """submit_action is handled by the LLMClient itself, never routed
        through the local executor (CLAUDE.md section 11.15/11.16)."""
        observation = _observation()
        executor = make_tool_executor(observation)
        result = executor("submit_action", {"shipment_id": "s1"})
        assert "error" in result


class TestPromptHash:
    def test_stable_across_calls(self) -> None:
        assert compute_prompt_hash() == compute_prompt_hash()

    def test_is_a_sha256_hex_digest(self) -> None:
        digest = compute_prompt_hash()
        assert len(digest) == 64
        int(digest, 16)  # raises ValueError if not valid hex

    def test_matches_a_direct_sha256_of_the_prompt_constant(self) -> None:
        assert compute_prompt_hash() == hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
