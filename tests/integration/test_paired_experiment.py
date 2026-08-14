"""Integration tests for the Milestone 7 trio: experiments/runner.py,
experiments/metrics.py, and data_io/writers.py, run together.

There is no separate test file for metrics.py or writers.py in the approved
repository structure (CLAUDE.md section 7), so their arithmetic and output
formats are exercised here too, alongside CLAUDE.md section 30.10's paired
invariants: branch state objects are distinct, the undisrupted tape differs
from the disrupted tape only by shocks, policies given equal observations
propose equal actions, TCD and delta are exact, and output rows cover all
four branches. `TestMultiReplicationFakePolicyExperimentWritesAllFiles` is
Milestone 7's own acceptance test verbatim: "a multi-replication
heuristic-versus-fake-policy experiment writes all required files" --
WaitFallbackPolicy stands in for the LLM agent, which does not exist until
Milestone 8. It does not test simulation/engine.py's own day-loop physics,
which tests/integration/test_full_simulation.py already covers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from supply_chain_simulator.data_io.loaders import (
    ExperimentConfig,
    PolicyConfigPathsConfig,
    ResolvedConfig,
    ScenarioConfig,
    ShockConfig,
    build_initial_state,
    build_network_definition,
    load_heuristic_policy_config,
    load_llm_policy_config,
    load_network_config,
    load_scenario_config,
)
from supply_chain_simulator.data_io.writers import ExperimentWriter
from supply_chain_simulator.domain.state import (
    CostCounters,
    DailyMetrics,
    ServiceCounters,
    SimulationResult,
)
from supply_chain_simulator.experiments.event_tape import (
    build_disrupted_event_tape,
    build_undisrupted_event_tape,
)
from supply_chain_simulator.experiments.metrics import (
    Winner,
    classify_winner,
    compute_replication_comparison,
    compute_run_metrics,
    summarize_experiment,
)
from supply_chain_simulator.experiments.runner import ExperimentRunner
from supply_chain_simulator.policies.fallback import WaitFallbackPolicy
from supply_chain_simulator.policies.heuristic import HeuristicPolicy

REPO_ROOT = Path(__file__).resolve().parents[2]
TINY_NETWORK_CONFIG = REPO_ROOT / "tests/fixtures/tiny_network.yaml"
TINY_SCENARIO_CONFIG = REPO_ROOT / "tests/fixtures/tiny_scenario.yaml"
HEURISTIC_CONFIG_PATH = REPO_ROOT / "configs/policies/heuristic.yaml"
LLM_CONFIG_PATH = REPO_ROOT / "configs/policies/llm_agent.yaml"


def _tiny_resolved_config(*, replications: int, base_seed: int) -> ResolvedConfig:
    """A ResolvedConfig built directly from the tiny fixtures plus the real
    policy configs, without going through resolve_config()'s file-based
    experiment config (no tiny experiment YAML exists among the approved
    fixtures). The tiny scenario's edge closure is on days 3-4, so
    warmup_days=2 keeps it safely inside the evaluation window.
    """
    experiment_config = ExperimentConfig(
        schema_version=1,
        experiment_id="tiny_paired_experiment",
        network_config="tiny_network.yaml",
        scenario_config="tiny_scenario.yaml",
        policy_configs=PolicyConfigPathsConfig(
            heuristic="heuristic.yaml", llm_agent="llm_agent.yaml"
        ),
        warmup_days=2,
        horizon_days=6,
        drain_days=3,
        terminal_penalty_days=30,
        replications=replications,
        base_seed=base_seed,
        counterfactual_mode="POLICY_SPECIFIC",
        fail_fast=True,
        output_root="outputs",
        write_event_tapes=True,
        write_daily_metrics=True,
        write_decision_traces=True,
        write_llm_interactions=True,
    )
    return ResolvedConfig(
        experiment=experiment_config,
        network=load_network_config(TINY_NETWORK_CONFIG),
        scenario=load_scenario_config(TINY_SCENARIO_CONFIG),
        heuristic_policy=load_heuristic_policy_config(HEURISTIC_CONFIG_PATH),
        llm_policy=load_llm_policy_config(LLM_CONFIG_PATH),
        experiment_config_path=REPO_ROOT
        / "configs/experiments/baseline_comparison.yaml",
        network_config_path=TINY_NETWORK_CONFIG,
        scenario_config_path=TINY_SCENARIO_CONFIG,
        heuristic_config_path=HEURISTIC_CONFIG_PATH,
        llm_config_path=LLM_CONFIG_PATH,
        output_root=REPO_ROOT / "outputs",
    )


def _heuristic_from(resolved_config: ResolvedConfig) -> HeuristicPolicy:
    return HeuristicPolicy(
        expedite_trigger_lateness_days=resolved_config.heuristic_policy.expedite_trigger_lateness_days,
        cost_tolerance=resolved_config.heuristic_policy.cost_tolerance,
    )


def _daily_metrics(day: int, backlog_units: int) -> DailyMetrics:
    return DailyMetrics(
        experiment_id="e",
        scenario_id="s",
        replication=1,
        policy="p",
        run_kind="DISRUPTED",
        day=day,
        inventory_units=0,
        backlog_units=backlog_units,
        shipments_at_node=0,
        shipments_in_transit=0,
        shipments_delivered=0,
        daily_demand_units=0,
        daily_same_day_fulfilled_units=0,
        daily_backlog_fulfilled_units=0,
        daily_transport_cost=0.0,
        daily_reroute_cost=0.0,
        daily_expedite_cost=0.0,
        daily_holding_cost=0.0,
        daily_backlog_cost=0.0,
        daily_late_cost=0.0,
        cumulative_total_cost=0.0,
        active_shock_ids=(),
    )


def _synthetic_result(
    *,
    costs: CostCounters,
    service: ServiceCounters,
    backlog_units_for_ending: int,
    daily_metrics: tuple[DailyMetrics, ...],
    terminated: bool = False,
) -> SimulationResult:
    network_config = load_network_config(TINY_NETWORK_CONFIG)
    network_definition = build_network_definition(network_config)
    state = build_initial_state(network_definition, network_config)
    state.costs = costs
    state.service = service
    state.backlog["plant_1"]["widget"] = backlog_units_for_ending
    return SimulationResult(
        experiment_id="e",
        scenario_id="s",
        replication=1,
        policy_name="p",
        run_kind="DISRUPTED",
        final_day=daily_metrics[-1].day if daily_metrics else 1,
        final_state=state,
        daily_metrics=daily_metrics,
        terminated_with_unresolved_state=terminated,
    )


class TestRunMetricsComputation:
    def test_costs_and_rates_match_hand_calculation(self) -> None:
        costs = CostCounters(
            transport=10.0,
            reroute=2.0,
            expedite=3.0,
            holding=1.0,
            backlog=0.5,
            late=0.25,
            terminal=0.0,
        )
        service = ServiceCounters(
            total_demand_units=100,
            same_day_fulfilled_units=80,
            delivered_shipment_units=50,
            late_delivered_units=5,
            total_lateness_unit_days=15,
            decision_count=20,
            invalid_action_count=2,
            abstention_count=1,
            fallback_count=3,
            reroute_count=4,
            expedite_count=1,
            expedited_units=5,
        )
        result = _synthetic_result(
            costs=costs,
            service=service,
            backlog_units_for_ending=7,
            daily_metrics=(_daily_metrics(1, 3), _daily_metrics(2, 7)),
        )
        metrics = compute_run_metrics(
            result, decision_latencies_ms=[1.0, 3.0, 2.0], shock_end_day=None
        )

        assert metrics.total_cost == pytest.approx(16.75)
        assert metrics.same_day_fill_rate == pytest.approx(0.8)
        assert metrics.final_fulfilment_rate == pytest.approx(93 / 100)
        assert metrics.ending_backlog_units == 7
        assert metrics.backlog_unit_days == 10
        assert metrics.late_delivery_rate == pytest.approx(5 / 50)
        assert metrics.average_lateness_days_weighted == pytest.approx(3.0)
        assert metrics.reroute_count == 4
        assert metrics.expedite_count == 1
        assert metrics.expedited_units == 5
        assert metrics.invalid_action_rate == pytest.approx(2 / 20)
        assert metrics.abstention_rate == pytest.approx(1 / 20)
        assert metrics.fallback_rate == pytest.approx(3 / 20)
        assert metrics.mean_decision_latency_ms == pytest.approx(2.0)
        assert metrics.days_to_clear_backlog_after_shock is None
        assert metrics.terminated_with_unresolved_state is False

    def test_rates_are_zero_when_denominator_is_zero(self) -> None:
        result = _synthetic_result(
            costs=CostCounters(),
            service=ServiceCounters(),
            backlog_units_for_ending=0,
            daily_metrics=(),
        )
        metrics = compute_run_metrics(
            result, decision_latencies_ms=[], shock_end_day=None
        )

        assert metrics.same_day_fill_rate == 0.0
        assert metrics.final_fulfilment_rate == 0.0
        assert metrics.late_delivery_rate == 0.0
        assert metrics.average_lateness_days_weighted == 0.0
        assert metrics.invalid_action_rate == 0.0
        assert metrics.abstention_rate == 0.0
        assert metrics.fallback_rate == 0.0
        assert metrics.mean_decision_latency_ms == 0.0

    def test_days_to_clear_backlog_after_shock(self) -> None:
        result = _synthetic_result(
            costs=CostCounters(),
            service=ServiceCounters(),
            backlog_units_for_ending=0,
            daily_metrics=(
                _daily_metrics(27, 5),
                _daily_metrics(28, 5),
                _daily_metrics(29, 0),
            ),
        )
        metrics = compute_run_metrics(
            result, decision_latencies_ms=[], shock_end_day=27
        )
        assert metrics.days_to_clear_backlog_after_shock == 1  # day 29 - (27 + 1)

    def test_days_to_clear_backlog_after_shock_none_when_it_never_clears(self) -> None:
        result = _synthetic_result(
            costs=CostCounters(),
            service=ServiceCounters(),
            backlog_units_for_ending=5,
            daily_metrics=(_daily_metrics(27, 5), _daily_metrics(28, 5)),
        )
        metrics = compute_run_metrics(
            result, decision_latencies_ms=[], shock_end_day=27
        )
        assert metrics.days_to_clear_backlog_after_shock is None

    def test_days_to_clear_backlog_after_shock_is_none_when_undisrupted(self) -> None:
        result = _synthetic_result(
            costs=CostCounters(),
            service=ServiceCounters(),
            backlog_units_for_ending=0,
            daily_metrics=(_daily_metrics(1, 0),),
        )
        metrics = compute_run_metrics(
            result, decision_latencies_ms=[], shock_end_day=None
        )
        assert metrics.days_to_clear_backlog_after_shock is None


class TestReplicationComparisonAndWinner:
    def test_tcd_and_delta_arithmetic(self) -> None:
        comparison = compute_replication_comparison(
            replication=1,
            seed=42,
            heuristic_undisrupted_cost=100.0,
            heuristic_disrupted_cost=150.0,
            llm_undisrupted_cost=90.0,
            llm_disrupted_cost=120.0,
        )
        assert comparison.heuristic_tcd == pytest.approx(50.0)
        assert comparison.llm_tcd == pytest.approx(30.0)
        assert comparison.delta == pytest.approx(-20.0)
        assert comparison.winner is Winner.LLM

    def test_winner_classification_boundaries(self) -> None:
        assert classify_winner(-0.02) is Winner.LLM
        assert classify_winner(-0.01) is Winner.TIE
        assert classify_winner(0.0) is Winner.TIE
        assert classify_winner(0.01) is Winner.TIE
        assert classify_winner(0.02) is Winner.HEURISTIC


class TestSummarizeExperiment:
    def test_aggregate_statistics_match_hand_calculation(self) -> None:
        comparisons = tuple(
            compute_replication_comparison(
                replication=i,
                seed=1000 + i,
                heuristic_undisrupted_cost=0.0,
                heuristic_disrupted_cost=0.0,
                llm_undisrupted_cost=0.0,
                llm_disrupted_cost=delta,
            )
            for i, delta in enumerate([1.0, 2.0, 3.0, 4.0, 5.0], start=1)
        )
        summary = summarize_experiment(comparisons)

        assert summary.replication_count == 5
        assert summary.mean_delta == pytest.approx(3.0)
        assert summary.median_delta == pytest.approx(3.0)
        assert summary.standard_deviation_delta == pytest.approx(1.5811388300841898)
        margin = 1.96 * summary.standard_deviation_delta / math.sqrt(5)
        assert summary.mean_delta_ci_95_lower == pytest.approx(3.0 - margin)
        assert summary.mean_delta_ci_95_upper == pytest.approx(3.0 + margin)
        assert summary.p10_delta == pytest.approx(1.4)
        assert summary.p90_delta == pytest.approx(4.6)
        assert summary.best_llm_delta == pytest.approx(1.0)
        assert summary.worst_llm_delta == pytest.approx(5.0)
        assert summary.heuristic_win_rate == pytest.approx(1.0)
        assert summary.llm_win_rate == pytest.approx(0.0)
        assert summary.tie_rate == pytest.approx(0.0)

    def test_raises_on_empty_sequence(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            summarize_experiment(())


class TestExperimentRunnerPairedInvariants:
    def test_identical_policies_produce_identical_disrupted_branches(
        self, tmp_path: Path
    ) -> None:
        """CLAUDE.md section 30.10: policies see equal observations for equal
        states. Two separate HeuristicPolicy instances (same deterministic
        rule) standing in for both slots means their disrupted branches --
        which the tiny scenario's edge closure actually triggers a decision
        on -- must reach exactly the same cost.
        """
        resolved_config = _tiny_resolved_config(replications=1, base_seed=200)
        runner = ExperimentRunner(
            heuristic_policy=_heuristic_from(resolved_config),
            heuristic_fallback_policy=WaitFallbackPolicy(),
            comparison_policy=_heuristic_from(resolved_config),
            comparison_fallback_policy=WaitFallbackPolicy(),
        )
        with ExperimentWriter(tmp_path) as writer:
            result = runner.run(resolved_config, writer)

        comparison = result.replication_comparisons[0]
        assert comparison.heuristic_disrupted_cost == pytest.approx(
            comparison.llm_disrupted_cost
        )
        assert comparison.heuristic_undisrupted_cost == pytest.approx(
            comparison.llm_undisrupted_cost
        )
        assert comparison.delta == pytest.approx(0.0, abs=1e-9)
        assert comparison.winner is Winner.TIE

    def test_undisrupted_tape_differs_from_disrupted_only_by_shocks(self) -> None:
        resolved_config = _tiny_resolved_config(replications=1, base_seed=200)
        network_definition = build_network_definition(resolved_config.network)
        disrupted = build_disrupted_event_tape(
            network_definition=network_definition,
            demand_process=resolved_config.network.demand_process,
            replenishment_plan=resolved_config.network.replenishment_plan,
            scenario_config=resolved_config.scenario,
            replication=1,
            base_seed=resolved_config.experiment.base_seed,
            horizon_days=resolved_config.experiment.horizon_days,
            drain_days=resolved_config.experiment.drain_days,
        )
        undisrupted = build_undisrupted_event_tape(disrupted)

        assert undisrupted.shocks == ()
        assert disrupted.shocks != ()
        for disrupted_day, undisrupted_day in zip(
            disrupted.days, undisrupted.days, strict=True
        ):
            assert disrupted_day.demand_events == undisrupted_day.demand_events
            assert (
                disrupted_day.shipment_release_events
                == undisrupted_day.shipment_release_events
            )
            assert (
                disrupted_day.edge_extra_delay_days
                == undisrupted_day.edge_extra_delay_days
            )

    def test_all_four_branches_produce_run_metrics_rows(self, tmp_path: Path) -> None:
        resolved_config = _tiny_resolved_config(replications=1, base_seed=200)
        runner = ExperimentRunner(
            heuristic_policy=_heuristic_from(resolved_config),
            heuristic_fallback_policy=WaitFallbackPolicy(),
            comparison_policy=WaitFallbackPolicy(),
            comparison_fallback_policy=WaitFallbackPolicy(),
        )
        with ExperimentWriter(tmp_path) as writer:
            runner.run(resolved_config, writer)

        rows = (tmp_path / "run_metrics.csv").read_text().splitlines()
        assert len(rows) == 1 + 4  # header + 4 branches
        pairs = {(row.split(",")[4], row.split(",")[5]) for row in rows[1:]}
        assert pairs == {
            ("heuristic", "UNDISRUPTED"),
            ("heuristic", "DISRUPTED"),
            ("llm_agent", "UNDISRUPTED"),
            ("llm_agent", "DISRUPTED"),
        }

    def test_tcd_and_delta_are_exact(self, tmp_path: Path) -> None:
        resolved_config = _tiny_resolved_config(replications=1, base_seed=200)
        runner = ExperimentRunner(
            heuristic_policy=_heuristic_from(resolved_config),
            heuristic_fallback_policy=WaitFallbackPolicy(),
            comparison_policy=WaitFallbackPolicy(),
            comparison_fallback_policy=WaitFallbackPolicy(),
        )
        with ExperimentWriter(tmp_path) as writer:
            result = runner.run(resolved_config, writer)

        comparison = result.replication_comparisons[0]
        assert comparison.heuristic_tcd == pytest.approx(
            comparison.heuristic_disrupted_cost - comparison.heuristic_undisrupted_cost
        )
        assert comparison.llm_tcd == pytest.approx(
            comparison.llm_disrupted_cost - comparison.llm_undisrupted_cost
        )
        assert comparison.delta == pytest.approx(
            comparison.llm_tcd - comparison.heuristic_tcd
        )


class TestExperimentWriterDirectly:
    def test_append_llm_interactions_writes_one_line_per_entry(
        self, tmp_path: Path
    ) -> None:
        with ExperimentWriter(tmp_path) as writer:
            writer.append_llm_interactions(
                [
                    {"decision_key": "a", "model": "gpt-test"},
                    {"decision_key": "b", "model": "gpt-test"},
                ]
            )

        lines = (tmp_path / "llm_interactions.jsonl").read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["decision_key"] == "a"
        assert json.loads(lines[1])["decision_key"] == "b"

    def test_write_summary_rounds_floats_inside_lists(self, tmp_path: Path) -> None:
        with ExperimentWriter(tmp_path) as writer:
            writer.write_summary({"deltas": [1.0000001, 2.0000009], "label": "x"})

        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["deltas"] == [1.0, 2.000001]
        assert summary["label"] == "x"


class TestMultiReplicationFakePolicyExperimentWritesAllFiles:
    """Milestone 6's build-sequence contract's own acceptance test for
    Milestone 7: "a multi-replication heuristic-versus-fake-policy experiment
    writes all required files." WaitFallbackPolicy stands in for the LLM
    agent (Milestone 8).
    """

    def test_writes_every_required_output_file(self, tmp_path: Path) -> None:
        resolved_config = _tiny_resolved_config(replications=3, base_seed=300)
        runner = ExperimentRunner(
            heuristic_policy=_heuristic_from(resolved_config),
            heuristic_fallback_policy=WaitFallbackPolicy(),
            comparison_policy=WaitFallbackPolicy(),
            comparison_fallback_policy=WaitFallbackPolicy(),
        )
        progress: list[tuple[int, str]] = []

        with ExperimentWriter(tmp_path) as writer:
            result = runner.run(
                resolved_config,
                writer,
                on_replication_complete=lambda replication, comparison: progress.append(
                    (replication, comparison.winner.value)
                ),
            )

        assert len(result.replication_comparisons) == 3
        assert [entry[0] for entry in progress] == [1, 2, 3]

        expected_files = {
            "manifest.json",
            "resolved_config.yaml",
            "event_tapes.jsonl",
            "run_metrics.csv",
            "daily_metrics.csv",
            "decision_traces.jsonl",
            "llm_interactions.jsonl",
            "replications.csv",
            "summary.json",
        }
        actual_files = {path.name for path in tmp_path.iterdir()}
        assert expected_files <= actual_files

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["experiment_id"] == "tiny_paired_experiment"
        assert manifest["base_seed"] == 300
        assert manifest["replications"] == 3
        assert len(manifest["network_config_sha256"]) == 64

        # No real LLM agent exists yet (Milestone 8), so no interactions are recorded.
        assert (tmp_path / "llm_interactions.jsonl").stat().st_size == 0

        replication_rows = (tmp_path / "replications.csv").read_text().splitlines()
        assert len(replication_rows) == 1 + 3  # header + 3 replications

        decision_trace_lines = (
            (tmp_path / "decision_traces.jsonl").read_text().splitlines()
        )
        assert (
            len(decision_trace_lines) > 0
        )  # the edge closure triggers at least one decision

        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["experiment_summary"]["replication_count"] == 3
        assert set(summary["cost_component_means"].keys()) == {
            "heuristic:UNDISRUPTED",
            "heuristic:DISRUPTED",
            "llm_agent:UNDISRUPTED",
            "llm_agent:DISRUPTED",
        }


class TestCompoundEventUndisruptedTapeStripping:
    """V2 §V2.3.4/§V2.9: a compound event's two shocks realize a shared
    start day but independent durations, and the undisrupted tape removes
    both while every other draw stays identical -- extending V1's existing
    "undisrupted tape differs only by shocks" assertion to grouped shocks.
    """

    def test_group_shares_start_day_but_strips_both_shocks(self) -> None:
        network_config = load_network_config(TINY_NETWORK_CONFIG)
        network_definition = build_network_definition(network_config)
        scenario_config = ScenarioConfig(
            schema_version=1,
            scenario_id="tiny_compound_event",
            description="Two correlated shocks sharing one start-day jitter draw.",
            shocks=[
                ShockConfig(
                    shock_id="a_edge_closure",
                    shock_type="EDGE_CLOSURE",
                    target_type="EDGE",
                    target_id="supplier_to_hub",
                    planned_start_day=3,
                    start_day_jitter_days=1,
                    minimum_duration_days=1,
                    duration_mean_days=1,
                    duration_std_days=0,
                    maximum_duration_days=1,
                    max_information_delay_days=0,
                    event_group_id="regional",
                ),
                ShockConfig(
                    shock_id="b_air_lead_time",
                    shock_type="EDGE_LEAD_TIME_INCREASE",
                    target_type="EDGE",
                    target_id="supplier_to_plant_air",
                    planned_start_day=3,
                    start_day_jitter_days=1,
                    minimum_duration_days=2,
                    duration_mean_days=2,
                    duration_std_days=0,
                    maximum_duration_days=2,
                    max_information_delay_days=0,
                    lead_time_multiplier=1.5,
                    event_group_id="regional",
                ),
            ],
        )
        disrupted = build_disrupted_event_tape(
            network_definition=network_definition,
            demand_process=network_config.demand_process,
            replenishment_plan=network_config.replenishment_plan,
            scenario_config=scenario_config,
            replication=1,
            base_seed=500,
            horizon_days=6,
            drain_days=3,
        )
        undisrupted = build_undisrupted_event_tape(disrupted)

        assert len(disrupted.shocks) == 2
        shock_a = next(s for s in disrupted.shocks if s.shock_id == "a_edge_closure")
        shock_b = next(s for s in disrupted.shocks if s.shock_id == "b_air_lead_time")
        # Shared jitter: both realized start days move by the same offset
        # from their (equal) planned_start_day.
        assert shock_a.physical_start_day == shock_b.physical_start_day
        # Independent durations: 1 day vs 2 days, so end days necessarily differ.
        assert shock_a.physical_end_day != shock_b.physical_end_day

        assert undisrupted.shocks == ()
        for disrupted_day, undisrupted_day in zip(disrupted.days, undisrupted.days, strict=True):
            assert disrupted_day.demand_events == undisrupted_day.demand_events
            assert (
                disrupted_day.shipment_release_events == undisrupted_day.shipment_release_events
            )
            assert disrupted_day.edge_extra_delay_days == undisrupted_day.edge_extra_delay_days
            assert undisrupted_day.newly_known_shock_ids == ()


def _topology_resolved_config(
    network_filename: str, *, replications: int, base_seed: int
) -> ResolvedConfig:
    """A ResolvedConfig built directly from a real topology-tier network file
    plus the real port_closure.yaml scenario (target_id port_primary exists
    in every tier by construction, V2 §V2.3.1), with a shortened horizon so
    the smoke test runs quickly.
    """
    network_path = REPO_ROOT / "configs/networks" / network_filename
    scenario_path = REPO_ROOT / "configs/scenarios/port_closure.yaml"
    experiment_config = ExperimentConfig(
        schema_version=1,
        experiment_id="topology_smoke_test",
        network_config=network_filename,
        scenario_config="port_closure.yaml",
        policy_configs=PolicyConfigPathsConfig(
            heuristic="heuristic.yaml", llm_agent="llm_agent.yaml"
        ),
        warmup_days=20,
        horizon_days=35,
        drain_days=10,
        terminal_penalty_days=30,
        replications=replications,
        base_seed=base_seed,
        counterfactual_mode="POLICY_SPECIFIC",
        fail_fast=True,
        output_root="outputs",
        write_event_tapes=True,
        write_daily_metrics=True,
        write_decision_traces=True,
        write_llm_interactions=True,
    )
    return ResolvedConfig(
        experiment=experiment_config,
        network=load_network_config(network_path),
        scenario=load_scenario_config(scenario_path),
        heuristic_policy=load_heuristic_policy_config(HEURISTIC_CONFIG_PATH),
        llm_policy=load_llm_policy_config(LLM_CONFIG_PATH),
        experiment_config_path=REPO_ROOT / "configs/experiments/baseline_comparison.yaml",
        network_config_path=network_path,
        scenario_config_path=scenario_path,
        heuristic_config_path=HEURISTIC_CONFIG_PATH,
        llm_config_path=LLM_CONFIG_PATH,
        output_root=REPO_ROOT / "outputs",
    )


class TestFullPairedRunPerTopologyTier:
    """V2 §V2.9: a full paired run on each topology tier completes and
    produces valid TCD/delta values -- proving no topology-specific code
    path is missing. WaitFallbackPolicy stands in for the LLM agent (no live
    API call, matching V1 §9's "no test ever calls a real API").
    """

    @pytest.mark.parametrize(
        "network_filename",
        ["topology_compact.yaml", "baseline_network.yaml", "topology_extended.yaml"],
    )
    def test_tier_completes_with_valid_tcd(self, network_filename: str, tmp_path: Path) -> None:
        resolved_config = _topology_resolved_config(network_filename, replications=1, base_seed=900)
        runner = ExperimentRunner(
            heuristic_policy=_heuristic_from(resolved_config),
            heuristic_fallback_policy=WaitFallbackPolicy(),
            comparison_policy=WaitFallbackPolicy(),
            comparison_fallback_policy=WaitFallbackPolicy(),
        )
        with ExperimentWriter(tmp_path) as writer:
            result = runner.run(resolved_config, writer)

        assert len(result.replication_comparisons) == 1
        comparison = result.replication_comparisons[0]
        assert comparison.heuristic_tcd == pytest.approx(
            comparison.heuristic_disrupted_cost - comparison.heuristic_undisrupted_cost
        )
        assert comparison.llm_tcd == pytest.approx(
            comparison.llm_disrupted_cost - comparison.llm_undisrupted_cost
        )
        assert comparison.delta == pytest.approx(comparison.llm_tcd - comparison.heuristic_tcd)
