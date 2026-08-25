"""Checks that a fixed config and seed reproduce byte-identical results across runs, including a decision replayed from a recorded interaction with no network call."""

from __future__ import annotations

import json
from pathlib import Path

from supply_chain_simulator.data_io.loaders import (
    ScenarioConfig,
    ShockConfig,
    build_initial_state,
    build_network_definition,
    load_network_config,
    load_scenario_config,
)
from supply_chain_simulator.experiments.event_tape import build_disrupted_event_tape
from supply_chain_simulator.integrations.llm_client import (
    FakeLLMClient,
    LLMInteractionResult,
    ReplayLLMClient,
    interaction_to_dict,
)
from supply_chain_simulator.policies.fallback import WaitFallbackPolicy
from supply_chain_simulator.policies.heuristic import HeuristicPolicy
from supply_chain_simulator.policies.llm_agent import (
    LLMAgentPolicy,
    compute_prompt_hash,
)
from supply_chain_simulator.simulation.engine import RunIdentity, SimulationEngine

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_NETWORK_CONFIG = REPO_ROOT / "tests/fixtures/tiny_network.yaml"
TINY_SCENARIO_CONFIG = REPO_ROOT / "tests/fixtures/tiny_scenario.yaml"

HORIZON_DAYS = 3
DRAIN_DAYS = 2
REROUTE_COST_PER_UNIT = 1.0
EXPEDITE_PREMIUM_PER_UNIT = 2.0
TERMINAL_PENALTY_DAYS = 30

# The only shipment tiny_scenario.yaml's edge closure ever triggers a
# decision for, per test_full_simulation.py's TestFullSimulationHeuristicPolicy.
TRIGGERED_SHIPMENT_ID = "shipment_003_001"


def _run_identity(policy_name: str) -> RunIdentity:
    return RunIdentity(
        experiment_id="repro_experiment",
        scenario_id="tiny_edge_closure",
        replication=1,
        policy_name=policy_name,
        run_kind="DISRUPTED",
    )


def _build_disrupted_tape():
    network_config = load_network_config(TINY_NETWORK_CONFIG)
    network_definition = build_network_definition(network_config)
    scenario_config = load_scenario_config(TINY_SCENARIO_CONFIG)
    tape = build_disrupted_event_tape(
        network_definition=network_definition,
        demand_process=network_config.demand_process,
        replenishment_plan=network_config.replenishment_plan,
        scenario_config=scenario_config,
        replication=1,
        base_seed=42,
        horizon_days=HORIZON_DAYS,
        drain_days=DRAIN_DAYS,
    )
    return network_config, network_definition, tape


class TestEventTapeReproducibility:
    def test_same_seed_and_replication_produce_identical_tapes(self) -> None:
        _, _, first = _build_disrupted_tape()
        _, _, second = _build_disrupted_tape()
        assert first == second

    def test_different_replication_derives_a_different_seed(self) -> None:
        """tiny_network.yaml has zero demand variance and reliability=1.0
        (test_full_simulation.py's docstring: "hand-calculable"), so its
        `days` are identical across replications regardless of seed; the
        seed derivation itself differing is what this checks.
        """
        network_config, network_definition, first = _build_disrupted_tape()
        scenario_config = load_scenario_config(TINY_SCENARIO_CONFIG)
        second = build_disrupted_event_tape(
            network_definition=network_definition,
            demand_process=network_config.demand_process,
            replenishment_plan=network_config.replenishment_plan,
            scenario_config=scenario_config,
            replication=2,
            base_seed=42,
            horizon_days=HORIZON_DAYS,
            drain_days=DRAIN_DAYS,
        )
        assert first.seed != second.seed
        assert first.replication != second.replication


class TestShockAndReleaseRealizationReproducibility:
    """V2 §V2.9: identical config and seed reproduce identical realized
    shocks (start day, duration, information day) and identical realized
    release quantities, extending V1's existing event-tape reproducibility
    assertion (TestEventTapeReproducibility above) to V2's new randomness.
    """

    def _build(self):
        network_config = load_network_config(TINY_NETWORK_CONFIG)
        network_definition = build_network_definition(network_config)
        scenario_config = ScenarioConfig(
            schema_version=1,
            scenario_id="tiny_uncertain_shock",
            description="A shock with genuine timing/duration/information uncertainty.",
            shocks=[
                ShockConfig(
                    shock_id="uncertain_closure",
                    shock_type="EDGE_CLOSURE",
                    target_type="EDGE",
                    target_id="supplier_to_hub",
                    planned_start_day=3,
                    start_day_jitter_days=2,
                    minimum_duration_days=1,
                    duration_mean_days=2,
                    duration_std_days=1,
                    maximum_duration_days=4,
                    max_information_delay_days=2,
                )
            ],
        )
        replenishment_plan = network_config.replenishment_plan.model_copy(
            update={"shipment_quantity_std": 3, "maximum_shipment_quantity": 15}
        )
        return build_disrupted_event_tape(
            network_definition=network_definition,
            demand_process=network_config.demand_process,
            replenishment_plan=replenishment_plan,
            scenario_config=scenario_config,
            replication=1,
            base_seed=777,
            horizon_days=HORIZON_DAYS,
            drain_days=DRAIN_DAYS,
        )

    def test_identical_seed_reproduces_identical_shocks_and_quantities(self) -> None:
        first = self._build()
        second = self._build()

        assert first.shocks == second.shocks
        first_quantities = [
            event.quantity
            for day_events in first.days
            for event in day_events.shipment_release_events
        ]
        second_quantities = [
            event.quantity
            for day_events in second.days
            for event in day_events.shipment_release_events
        ]
        assert first_quantities == second_quantities

    def test_realization_is_not_accidentally_degenerate(self) -> None:
        """Sanity check that this fixture actually exercises uncertainty
        (jitter/duration/information-delay draws that can differ from the
        planned defaults), so the reproducibility assertion above is testing
        something real rather than an all-zero-variance edge case.
        """
        tape = self._build()
        shock = tape.shocks[0]
        realized_duration = shock.physical_end_day - shock.physical_start_day + 1
        assert (
            shock.physical_start_day != 3
            or realized_duration != 2
            or shock.information_day != shock.physical_start_day
        )


class TestHeuristicReproducibility:
    def _run(self):
        network_config, network_definition, tape = _build_disrupted_tape()
        initial_state = build_initial_state(network_definition, network_config)
        return SimulationEngine().run(
            initial_state=initial_state,
            event_tape=tape,
            start_day=1,
            horizon_day=HORIZON_DAYS,
            drain_days=DRAIN_DAYS,
            decision_enabled=True,
            run_identity=_run_identity("heuristic"),
            reroute_cost_per_unit=REROUTE_COST_PER_UNIT,
            expedite_premium_per_unit=EXPEDITE_PREMIUM_PER_UNIT,
            terminal_penalty_days=TERMINAL_PENALTY_DAYS,
            policy=HeuristicPolicy(expedite_trigger_lateness_days=2, cost_tolerance=1e-9),
            fallback_policy=WaitFallbackPolicy(),
            mean_daily_demand=network_config.demand_process.mean_daily_demand,
        )

    def test_repeated_runs_are_byte_for_byte_metric_identical(self) -> None:
        first = self._run()
        second = self._run()
        assert first.final_state.costs == second.final_state.costs
        assert first.final_state.service == second.final_state.service
        assert first.daily_metrics == second.daily_metrics
        assert (
            first.terminated_with_unresolved_state == second.terminated_with_unresolved_state
        )


_FIXED_SUBMISSION: dict[str, object] = {
    "shipment_id": TRIGGERED_SHIPMENT_ID,
    "action_type": "WAIT",
    "route_id": None,
    "reason_code": "LOWER_ESTIMATED_COST",
    "rationale": "wait deterministically",
}


def _fake_llm_policy() -> LLMAgentPolicy:
    return LLMAgentPolicy(
        client=FakeLLMClient(submitted_action=dict(_FIXED_SUBMISSION)),
        model="fake-model",
        temperature=0.0,
        max_tool_calls=8,
        max_output_tokens=100,
        request_timeout_seconds=60,
        max_retries=3,
    )


class TestFakeLLMReproducibility:
    def _run(self):
        network_config, network_definition, tape = _build_disrupted_tape()
        initial_state = build_initial_state(network_definition, network_config)
        return SimulationEngine().run(
            initial_state=initial_state,
            event_tape=tape,
            start_day=1,
            horizon_day=HORIZON_DAYS,
            drain_days=DRAIN_DAYS,
            decision_enabled=True,
            run_identity=_run_identity("llm_agent"),
            reroute_cost_per_unit=REROUTE_COST_PER_UNIT,
            expedite_premium_per_unit=EXPEDITE_PREMIUM_PER_UNIT,
            terminal_penalty_days=TERMINAL_PENALTY_DAYS,
            policy=_fake_llm_policy(),
            fallback_policy=WaitFallbackPolicy(),
            mean_daily_demand=network_config.demand_process.mean_daily_demand,
        )

    def test_repeated_runs_are_identical(self) -> None:
        first = self._run()
        second = self._run()
        assert first.final_state.costs == second.final_state.costs
        assert first.final_state.shipments == second.final_state.shipments
        assert first.daily_metrics == second.daily_metrics


class TestReplayReproducesRecordedDecisions:
    def _run_with_fake_and_record(self) -> tuple[object, list[LLMInteractionResult]]:
        network_config, network_definition, tape = _build_disrupted_tape()
        initial_state = build_initial_state(network_definition, network_config)
        sink: list[LLMInteractionResult] = []
        result = SimulationEngine().run(
            initial_state=initial_state,
            event_tape=tape,
            start_day=1,
            horizon_day=HORIZON_DAYS,
            drain_days=DRAIN_DAYS,
            decision_enabled=True,
            run_identity=_run_identity("llm_agent"),
            reroute_cost_per_unit=REROUTE_COST_PER_UNIT,
            expedite_premium_per_unit=EXPEDITE_PREMIUM_PER_UNIT,
            terminal_penalty_days=TERMINAL_PENALTY_DAYS,
            policy=_fake_llm_policy(),
            fallback_policy=WaitFallbackPolicy(),
            mean_daily_demand=network_config.demand_process.mean_daily_demand,
            llm_interaction_sink=sink,
        )
        return result, sink

    def _run_with_replay(self, trace_path: Path):
        network_config, network_definition, tape = _build_disrupted_tape()
        initial_state = build_initial_state(network_definition, network_config)
        policy = LLMAgentPolicy(
            client=ReplayLLMClient(trace_path),
            model="fake-model",
            temperature=0.0,
            max_tool_calls=8,
            max_output_tokens=100,
            request_timeout_seconds=60,
            max_retries=3,
        )
        return SimulationEngine().run(
            initial_state=initial_state,
            event_tape=tape,
            start_day=1,
            horizon_day=HORIZON_DAYS,
            drain_days=DRAIN_DAYS,
            decision_enabled=True,
            run_identity=_run_identity("llm_agent"),
            reroute_cost_per_unit=REROUTE_COST_PER_UNIT,
            expedite_premium_per_unit=EXPEDITE_PREMIUM_PER_UNIT,
            terminal_penalty_days=TERMINAL_PENALTY_DAYS,
            policy=policy,
            fallback_policy=WaitFallbackPolicy(),
            mean_daily_demand=network_config.demand_process.mean_daily_demand,
        )

    def test_replay_reproduces_the_same_physical_and_cost_outcome(self, tmp_path: Path) -> None:
        original_result, sink = self._run_with_fake_and_record()

        assert [interaction.decision_key.day for interaction in sink] == [3, 4]
        assert all(
            interaction.decision_key.shipment_id == TRIGGERED_SHIPMENT_ID
            for interaction in sink
        )

        prompt_hash = compute_prompt_hash()
        trace_path = tmp_path / "llm_interactions.jsonl"
        trace_path.write_text(
            "\n".join(
                json.dumps(interaction_to_dict(interaction, prompt_hash)) for interaction in sink
            )
            + "\n",
            encoding="utf-8",
        )

        replayed_result = self._run_with_replay(trace_path)

        assert replayed_result.final_state.costs == original_result.final_state.costs
        assert replayed_result.final_state.shipments == original_result.final_state.shipments
        assert replayed_result.daily_metrics == original_result.daily_metrics
