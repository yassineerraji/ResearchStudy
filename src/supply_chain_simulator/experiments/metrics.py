"""Run-level metrics for one branch and experiment-level statistics across replications.

Inside the experiments package, this module turns one branch's finished
SimulationResult (plus that branch's decision-latency samples and, for
disrupted branches, the scenario's shock end day) into the exact RunMetrics
CLAUDE.md section 11.19 lists, turns four branches' costs for one replication
into a ReplicationComparison (the paired TCD delta and its winner, section
5.8), and turns a full set of replications' comparisons into the
ExperimentSummary CLAUDE.md section 27.8 requires (means, medians, the 95%
confidence interval, win rates, and percentiles). In the full system, this is
the only place TCD, delta, and every aggregate statistic are computed, so the
same arithmetic backs both the printed CLI summary and summary.json. It does
not run a simulation or decide what to compare — experiments/runner.py owns
that — and it never mutates a SimulationResult.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from supply_chain_simulator.domain.state import SimulationResult
from supply_chain_simulator.simulation.costs import total_cost

OUTPUT_TIE_TOLERANCE = 0.01


class Winner(Enum):
    LLM = "LLM"
    HEURISTIC = "HEURISTIC"
    TIE = "TIE"


@dataclass(frozen=True, slots=True)
class RunMetrics:
    total_cost: float
    transport_cost: float
    reroute_cost: float
    expedite_cost: float
    holding_cost: float
    backlog_cost: float
    late_cost: float
    terminal_cost: float
    same_day_fill_rate: float
    final_fulfilment_rate: float
    ending_backlog_units: int
    backlog_unit_days: int
    late_delivered_units: int
    late_delivery_rate: float
    average_lateness_days_weighted: float
    reroute_count: int
    expedite_count: int
    expedited_units: int
    decision_count: int
    invalid_action_rate: float
    abstention_rate: float
    fallback_rate: float
    mean_decision_latency_ms: float
    days_to_clear_backlog_after_shock: int | None
    terminated_with_unresolved_state: bool


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _days_to_clear_backlog_after_shock(
    result: SimulationResult, shock_end_day: int | None
) -> int | None:
    """CLAUDE.md section 11.19: days from the first day after shock end to the
    first day backlog is zero, counting the first day after shock end itself
    as 0 days elapsed. `None` if there was no shock, or backlog never clears.
    """
    if shock_end_day is None:
        return None
    first_day_after_shock = shock_end_day + 1
    for daily in result.daily_metrics:
        if daily.day < first_day_after_shock:
            continue
        if daily.backlog_units == 0:
            return daily.day - first_day_after_shock
    return None


def compute_run_metrics(
    result: SimulationResult,
    decision_latencies_ms: Sequence[float],
    shock_end_day: int | None,
) -> RunMetrics:
    state = result.final_state
    costs = state.costs
    service = state.service

    ending_backlog_units = sum(
        quantity
        for products in state.backlog.values()
        for quantity in products.values()
    )
    backlog_unit_days = sum(daily.backlog_units for daily in result.daily_metrics)

    return RunMetrics(
        total_cost=total_cost(state),
        transport_cost=costs.transport,
        reroute_cost=costs.reroute,
        expedite_cost=costs.expedite,
        holding_cost=costs.holding,
        backlog_cost=costs.backlog,
        late_cost=costs.late,
        terminal_cost=costs.terminal,
        same_day_fill_rate=_rate(
            service.same_day_fulfilled_units, service.total_demand_units
        ),
        final_fulfilment_rate=_rate(
            service.total_demand_units - ending_backlog_units,
            service.total_demand_units,
        ),
        ending_backlog_units=ending_backlog_units,
        backlog_unit_days=backlog_unit_days,
        late_delivered_units=service.late_delivered_units,
        late_delivery_rate=_rate(
            service.late_delivered_units, service.delivered_shipment_units
        ),
        average_lateness_days_weighted=(
            service.total_lateness_unit_days / service.late_delivered_units
            if service.late_delivered_units
            else 0.0
        ),
        reroute_count=service.reroute_count,
        expedite_count=service.expedite_count,
        expedited_units=service.expedited_units,
        decision_count=service.decision_count,
        invalid_action_rate=_rate(service.invalid_action_count, service.decision_count),
        abstention_rate=_rate(service.abstention_count, service.decision_count),
        fallback_rate=_rate(service.fallback_count, service.decision_count),
        mean_decision_latency_ms=(
            statistics.mean(decision_latencies_ms) if decision_latencies_ms else 0.0
        ),
        days_to_clear_backlog_after_shock=_days_to_clear_backlog_after_shock(
            result, shock_end_day
        ),
        terminated_with_unresolved_state=result.terminated_with_unresolved_state,
    )


def classify_winner(delta: float) -> Winner:
    if delta < -OUTPUT_TIE_TOLERANCE:
        return Winner.LLM
    if delta > OUTPUT_TIE_TOLERANCE:
        return Winner.HEURISTIC
    return Winner.TIE


@dataclass(frozen=True, slots=True)
class ReplicationComparison:
    replication: int
    seed: int
    heuristic_undisrupted_cost: float
    heuristic_disrupted_cost: float
    heuristic_tcd: float
    llm_undisrupted_cost: float
    llm_disrupted_cost: float
    llm_tcd: float
    delta: float
    winner: Winner


def compute_replication_comparison(
    replication: int,
    seed: int,
    heuristic_undisrupted_cost: float,
    heuristic_disrupted_cost: float,
    llm_undisrupted_cost: float,
    llm_disrupted_cost: float,
) -> ReplicationComparison:
    """CLAUDE.md section 5.8: TCD is each policy's disrupted-minus-undisrupted
    cost; delta is the LLM's TCD minus the heuristic's.
    """
    heuristic_tcd = heuristic_disrupted_cost - heuristic_undisrupted_cost
    llm_tcd = llm_disrupted_cost - llm_undisrupted_cost
    delta = llm_tcd - heuristic_tcd
    return ReplicationComparison(
        replication=replication,
        seed=seed,
        heuristic_undisrupted_cost=heuristic_undisrupted_cost,
        heuristic_disrupted_cost=heuristic_disrupted_cost,
        heuristic_tcd=heuristic_tcd,
        llm_undisrupted_cost=llm_undisrupted_cost,
        llm_disrupted_cost=llm_disrupted_cost,
        llm_tcd=llm_tcd,
        delta=delta,
        winner=classify_winner(delta),
    )


@dataclass(frozen=True, slots=True)
class ExperimentSummary:
    replication_count: int
    mean_heuristic_tcd: float
    median_heuristic_tcd: float
    mean_llm_tcd: float
    median_llm_tcd: float
    mean_delta: float
    median_delta: float
    standard_deviation_delta: float
    mean_delta_ci_95_lower: float
    mean_delta_ci_95_upper: float
    llm_win_rate: float
    heuristic_win_rate: float
    tie_rate: float
    best_llm_delta: float
    worst_llm_delta: float
    p10_delta: float
    p90_delta: float


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    """Deterministic linear interpolation over already-sorted values (CLAUDE.md
    section 11.19), matching the common "inclusive" percentile method.
    """
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    rank = (percentile / 100) * (n - 1)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = rank - lower_index
    return (
        sorted_values[lower_index]
        + (sorted_values[upper_index] - sorted_values[lower_index]) * fraction
    )


def summarize_experiment(
    comparisons: Sequence[ReplicationComparison],
) -> ExperimentSummary:
    replication_count = len(comparisons)
    if replication_count == 0:
        raise ValueError(
            "cannot summarize an empty sequence of replication comparisons"
        )

    heuristic_tcds = sorted(comparison.heuristic_tcd for comparison in comparisons)
    llm_tcds = sorted(comparison.llm_tcd for comparison in comparisons)
    deltas = [comparison.delta for comparison in comparisons]
    sorted_deltas = sorted(deltas)
    winners = [comparison.winner for comparison in comparisons]

    mean_delta = statistics.mean(deltas)
    standard_deviation_delta = (
        statistics.stdev(deltas) if replication_count >= 2 else 0.0
    )
    margin = (
        1.96 * standard_deviation_delta / math.sqrt(replication_count)
        if replication_count >= 2
        else 0.0
    )

    return ExperimentSummary(
        replication_count=replication_count,
        mean_heuristic_tcd=statistics.mean(heuristic_tcds),
        median_heuristic_tcd=_percentile(heuristic_tcds, 50),
        mean_llm_tcd=statistics.mean(llm_tcds),
        median_llm_tcd=_percentile(llm_tcds, 50),
        mean_delta=mean_delta,
        median_delta=_percentile(sorted_deltas, 50),
        standard_deviation_delta=standard_deviation_delta,
        mean_delta_ci_95_lower=mean_delta - margin,
        mean_delta_ci_95_upper=mean_delta + margin,
        llm_win_rate=winners.count(Winner.LLM) / replication_count,
        heuristic_win_rate=winners.count(Winner.HEURISTIC) / replication_count,
        tie_rate=winners.count(Winner.TIE) / replication_count,
        best_llm_delta=min(deltas),
        worst_llm_delta=max(deltas),
        p10_delta=_percentile(sorted_deltas, 10),
        p90_delta=_percentile(sorted_deltas, 90),
    )
