"""Runs the complete paired experiment: warm-up, four branches, every replication.

Inside the experiments package, this module is CLAUDE.md section 11.18's
ExperimentRunner: for every replication it derives one seed, builds the
paired disrupted/undisrupted event tapes, runs a decisions-disabled warm-up
once, snapshots and resets it into one shared starting point, deep-clones
that snapshot into the heuristic-undisrupted, heuristic-disrupted,
comparison-undisrupted, and comparison-disrupted branches (in that fixed
order), and turns their costs into one paired TCD delta. In the full system,
this is where the research comparison actually happens — the two policies
being compared are injected as plain `Policy` values, so this module never
imports HeuristicPolicy or LLMAgentPolicy itself, and a fake policy can
stand in for the LLM agent before Milestone 8 exists. It does not compute
metrics itself (experiments/metrics.py does) and does not decide how a
result is serialized (data_io/writers.py does).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from supply_chain_simulator.data_io.loaders import (
    ResolvedConfig,
    build_initial_state,
    build_network_definition,
)
from supply_chain_simulator.data_io.writers import ExperimentWriter, hash_file
from supply_chain_simulator.domain.state import (
    CostCounters,
    ServiceCounters,
    SimulationState,
)
from supply_chain_simulator.experiments.event_tape import (
    build_disrupted_event_tape,
    build_undisrupted_event_tape,
)
from supply_chain_simulator.experiments.metrics import (
    ExperimentSummary,
    ReplicationComparison,
    RunMetrics,
    compute_replication_comparison,
    compute_run_metrics,
    summarize_experiment,
)
from supply_chain_simulator.policies.base import Policy
from supply_chain_simulator.simulation.engine import (
    DecisionTraceEntry,
    RunIdentity,
    SimulationEngine,
)
from supply_chain_simulator.simulation.transition import reset_daily_capacity_usage

_HEURISTIC_POLICY_NAME = "heuristic"
_COMPARISON_POLICY_NAME = "llm_agent"
_UNDISRUPTED = "UNDISRUPTED"
_DISRUPTED = "DISRUPTED"


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    experiment_id: str
    replication_comparisons: tuple[ReplicationComparison, ...]
    summary: ExperimentSummary


def _snapshot_after_warmup(state: SimulationState) -> SimulationState:
    """CLAUDE.md section 11.18 step 7: preserve inventory, backlog, shipment
    positions, and due dates; reset cost and service counters to zero. The
    warm-up used the undisrupted tape, so operational state, active/known
    shocks, and daily capacity usage are already at their normal defaults.
    """
    state.costs = CostCounters()
    state.service = ServiceCounters()
    state.pre_shock_inventory = {
        node_id: dict(products) for node_id, products in state.inventory.items()
    }
    state.pre_shock_backlog = {
        node_id: dict(products) for node_id, products in state.backlog.items()
    }
    reset_daily_capacity_usage(state)
    return state


class ExperimentRunner:
    """Runs one ResolvedConfig's full paired experiment and writes every replication.

    The heuristic and comparison policies (each with its own fallback) are
    constructor-injected rather than built from `resolved_config.llm_policy`
    here, since building a real LLM policy from that config requires
    policies/llm_agent.py (Milestone 8). The CLI is responsible for
    resolving configuration into concrete Policy objects; tests may inject a
    fake policy in the comparison slot, matching CLAUDE.md's own Milestone 7
    acceptance test ("a multi-replication heuristic-versus-fake-policy
    experiment").
    """

    def __init__(
        self,
        heuristic_policy: Policy,
        heuristic_fallback_policy: Policy,
        comparison_policy: Policy,
        comparison_fallback_policy: Policy,
        engine: SimulationEngine | None = None,
    ) -> None:
        self._heuristic_policy = heuristic_policy
        self._heuristic_fallback_policy = heuristic_fallback_policy
        self._comparison_policy = comparison_policy
        self._comparison_fallback_policy = comparison_fallback_policy
        self._engine = engine or SimulationEngine()

    def run(
        self,
        resolved_config: ResolvedConfig,
        writer: ExperimentWriter,
        llm_prompt_sha256: str | None = None,
        on_replication_complete: Callable[[int, ReplicationComparison], None]
        | None = None,
    ) -> ExperimentResult:
        experiment = resolved_config.experiment
        network = resolved_config.network
        network_definition = build_network_definition(network)
        mean_daily_demand = network.demand_process.mean_daily_demand
        reroute_cost_per_unit = network.action_costs.reroute_cost_per_unit
        expedite_premium_per_unit = network.action_costs.expedite_premium_per_unit

        writer.write_manifest(resolved_config, llm_prompt_sha256)
        writer.write_resolved_config(resolved_config)

        comparisons: list[ReplicationComparison] = []
        run_metrics_records: list[tuple[str, str, RunMetrics]] = []

        for replication in range(1, experiment.replications + 1):
            disrupted_tape = build_disrupted_event_tape(
                network_definition=network_definition,
                demand_process=network.demand_process,
                replenishment_plan=network.replenishment_plan,
                scenario_config=resolved_config.scenario,
                replication=replication,
                base_seed=experiment.base_seed,
                horizon_days=experiment.horizon_days,
                drain_days=experiment.drain_days,
            )
            undisrupted_tape = build_undisrupted_event_tape(disrupted_tape)
            if experiment.write_event_tapes:
                writer.append_event_tape(disrupted_tape)

            shock_end_day = max(
                (shock.physical_end_day for shock in disrupted_tape.shocks),
                default=None,
            )

            day_zero_state = build_initial_state(network_definition, network)
            warmup_result = self._engine.run(
                initial_state=day_zero_state,
                event_tape=undisrupted_tape,
                start_day=1,
                horizon_day=experiment.warmup_days,
                drain_days=0,
                decision_enabled=False,
                run_identity=RunIdentity(
                    experiment_id=experiment.experiment_id,
                    scenario_id=resolved_config.scenario.scenario_id,
                    replication=replication,
                    policy_name="none",
                    run_kind="WARMUP",
                ),
                reroute_cost_per_unit=reroute_cost_per_unit,
                expedite_premium_per_unit=expedite_premium_per_unit,
                terminal_penalty_days=experiment.terminal_penalty_days,
            )
            snapshot = _snapshot_after_warmup(warmup_result.final_state)

            branch_costs: dict[tuple[str, str], float] = {}
            branch_specs = (
                (
                    _HEURISTIC_POLICY_NAME,
                    self._heuristic_policy,
                    self._heuristic_fallback_policy,
                    undisrupted_tape,
                    _UNDISRUPTED,
                ),
                (
                    _HEURISTIC_POLICY_NAME,
                    self._heuristic_policy,
                    self._heuristic_fallback_policy,
                    disrupted_tape,
                    _DISRUPTED,
                ),
                (
                    _COMPARISON_POLICY_NAME,
                    self._comparison_policy,
                    self._comparison_fallback_policy,
                    undisrupted_tape,
                    _UNDISRUPTED,
                ),
                (
                    _COMPARISON_POLICY_NAME,
                    self._comparison_policy,
                    self._comparison_fallback_policy,
                    disrupted_tape,
                    _DISRUPTED,
                ),
            )

            for (
                policy_name,
                policy,
                fallback_policy,
                event_tape,
                run_kind,
            ) in branch_specs:
                run_identity = RunIdentity(
                    experiment_id=experiment.experiment_id,
                    scenario_id=resolved_config.scenario.scenario_id,
                    replication=replication,
                    policy_name=policy_name,
                    run_kind=run_kind,
                )
                decision_trace_sink: list[DecisionTraceEntry] = []
                result = self._engine.run(
                    initial_state=snapshot,
                    event_tape=event_tape,
                    start_day=experiment.warmup_days + 1,
                    horizon_day=experiment.horizon_days,
                    drain_days=experiment.drain_days,
                    decision_enabled=True,
                    run_identity=run_identity,
                    reroute_cost_per_unit=reroute_cost_per_unit,
                    expedite_premium_per_unit=expedite_premium_per_unit,
                    terminal_penalty_days=experiment.terminal_penalty_days,
                    policy=policy,
                    fallback_policy=fallback_policy,
                    mean_daily_demand=mean_daily_demand,
                    decision_trace_sink=decision_trace_sink,
                )

                run_metrics = compute_run_metrics(
                    result,
                    [entry.decision_latency_ms for entry in decision_trace_sink],
                    shock_end_day if run_kind == _DISRUPTED else None,
                )
                writer.append_run_metrics(
                    experiment_id=experiment.experiment_id,
                    scenario_id=resolved_config.scenario.scenario_id,
                    replication=replication,
                    seed=disrupted_tape.seed,
                    policy=policy_name,
                    run_kind=run_kind,
                    metrics=run_metrics,
                )
                if experiment.write_daily_metrics:
                    writer.append_daily_metrics(result.daily_metrics)
                if experiment.write_decision_traces:
                    writer.append_decision_traces(decision_trace_sink)

                branch_costs[(policy_name, run_kind)] = run_metrics.total_cost
                run_metrics_records.append((policy_name, run_kind, run_metrics))

            comparison = compute_replication_comparison(
                replication=replication,
                seed=disrupted_tape.seed,
                heuristic_undisrupted_cost=branch_costs[
                    (_HEURISTIC_POLICY_NAME, _UNDISRUPTED)
                ],
                heuristic_disrupted_cost=branch_costs[
                    (_HEURISTIC_POLICY_NAME, _DISRUPTED)
                ],
                llm_undisrupted_cost=branch_costs[
                    (_COMPARISON_POLICY_NAME, _UNDISRUPTED)
                ],
                llm_disrupted_cost=branch_costs[(_COMPARISON_POLICY_NAME, _DISRUPTED)],
            )
            writer.append_replication_comparison(comparison)
            comparisons.append(comparison)
            if on_replication_complete is not None:
                on_replication_complete(replication, comparison)

        summary = summarize_experiment(comparisons)
        writer.write_summary(
            _build_summary_payload(
                summary, run_metrics_records, resolved_config, llm_prompt_sha256
            )
        )

        return ExperimentResult(
            experiment_id=experiment.experiment_id,
            replication_comparisons=tuple(comparisons),
            summary=summary,
        )


_COST_FIELDS = (
    "total_cost",
    "transport_cost",
    "reroute_cost",
    "expedite_cost",
    "holding_cost",
    "backlog_cost",
    "late_cost",
    "terminal_cost",
)
_SERVICE_FIELDS = (
    "same_day_fill_rate",
    "final_fulfilment_rate",
    "ending_backlog_units",
    "backlog_unit_days",
    "late_delivered_units",
    "late_delivery_rate",
    "average_lateness_days_weighted",
    "reroute_count",
    "expedite_count",
    "expedited_units",
    "decision_count",
    "mean_decision_latency_ms",
)
_DECISION_RATE_FIELDS = ("invalid_action_rate", "abstention_rate", "fallback_rate")


def _group_key(policy_name: str, run_kind: str) -> str:
    return f"{policy_name}:{run_kind}"


def _mean_by_group(
    grouped: dict[tuple[str, str], list[RunMetrics]], fields: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    return {
        _group_key(*key): {
            field: sum(getattr(metrics, field) for metrics in metrics_list)
            / len(metrics_list)
            for field in fields
        }
        for key, metrics_list in grouped.items()
    }


def _build_summary_payload(
    summary: ExperimentSummary,
    run_metrics_records: list[tuple[str, str, RunMetrics]],
    resolved_config: ResolvedConfig,
    llm_prompt_sha256: str | None,
) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[RunMetrics]] = {}
    for policy_name, run_kind, metrics in run_metrics_records:
        grouped.setdefault((policy_name, run_kind), []).append(metrics)

    unresolved_run_counts = {
        _group_key(*key): sum(
            1 for metrics in metrics_list if metrics.terminated_with_unresolved_state
        )
        for key, metrics_list in grouped.items()
    }

    return {
        "experiment_summary": {
            "replication_count": summary.replication_count,
            "mean_heuristic_tcd": summary.mean_heuristic_tcd,
            "median_heuristic_tcd": summary.median_heuristic_tcd,
            "mean_llm_tcd": summary.mean_llm_tcd,
            "median_llm_tcd": summary.median_llm_tcd,
            "mean_delta": summary.mean_delta,
            "median_delta": summary.median_delta,
            "standard_deviation_delta": summary.standard_deviation_delta,
            "mean_delta_ci_95_lower": summary.mean_delta_ci_95_lower,
            "mean_delta_ci_95_upper": summary.mean_delta_ci_95_upper,
            "llm_win_rate": summary.llm_win_rate,
            "heuristic_win_rate": summary.heuristic_win_rate,
            "tie_rate": summary.tie_rate,
            "best_llm_delta": summary.best_llm_delta,
            "worst_llm_delta": summary.worst_llm_delta,
            "p10_delta": summary.p10_delta,
            "p90_delta": summary.p90_delta,
        },
        "cost_component_means": _mean_by_group(grouped, _COST_FIELDS),
        "service_metric_means": _mean_by_group(grouped, _SERVICE_FIELDS),
        "decision_rate_means": _mean_by_group(grouped, _DECISION_RATE_FIELDS),
        "unresolved_run_counts": unresolved_run_counts,
        "configuration_hashes": {
            "network_config_sha256": hash_file(resolved_config.network_config_path),
            "scenario_config_sha256": hash_file(resolved_config.scenario_config_path),
            "heuristic_config_sha256": hash_file(resolved_config.heuristic_config_path),
            "llm_config_sha256": hash_file(resolved_config.llm_config_path),
            "experiment_config_sha256": hash_file(
                resolved_config.experiment_config_path
            ),
            "llm_prompt_sha256": llm_prompt_sha256,
        },
    }
