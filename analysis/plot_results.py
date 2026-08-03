"""Turns one or more experiment output directories into the plot set defined
by VISUAL_REPORTING_PLAN_ESSENTIAL.md.

Standalone script, deliberately outside src/supply_chain_simulator: it only
ever reads the CSV/JSON/JSONL files data_io/writers.py already produces
after a run and never touches simulation behavior, so it has no bearing on
scientific validity or fairness. Given one --experiment/--label pair, it
produces the eight per-scenario plots (01, 02, 03, 04, 05, 06, 09, 10). Given
two or more, it also produces the cross-scenario forest plot (07). Plot 08
(a severity x duration heatmap) is intentionally not implemented: it needs
disruption profiles to form a genuine 2D grid, and the current Light/Medium/
Heavy profiles vary by mechanism (capacity cut / closure duration / closure
plus congestion), not by two independent, crossed axes -- building that grid
would be a separate, larger experimental-design change. Requires the
optional "analysis" dependency group (`pip install -e ".[analysis]"`) since
it is the only place in this project that uses matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

_HEURISTIC_COLOR = "#4c72b0"
_LLM_COLOR = "#dd8452"
_TIE_COLOR = "#8c8c8c"

_COST_COMPONENTS = (
    "transport_cost",
    "reroute_cost",
    "expedite_cost",
    "holding_cost",
    "backlog_cost",
    "late_cost",
    "terminal_cost",
)
_HEATMAP_CATEGORIES = ("WAIT", "REROUTE", "EXPEDITE", "ABSTAIN", "INVALID", "FALLBACK")
_POLICY_LABELS = {"heuristic": "Heuristic", "llm_agent": "LLM agent"}
_POLICY_COLORS = {"heuristic": _HEURISTIC_COLOR, "llm_agent": _LLM_COLOR}

_RUN_METRICS_FLOAT_FIELDS = (
    "total_cost",
    "transport_cost",
    "reroute_cost",
    "expedite_cost",
    "holding_cost",
    "backlog_cost",
    "late_cost",
    "terminal_cost",
    "same_day_fill_rate",
    "final_fulfilment_rate",
    "late_delivery_rate",
    "average_lateness_days_weighted",
    "invalid_action_rate",
    "abstention_rate",
    "fallback_rate",
    "mean_decision_latency_ms",
)
_RUN_METRICS_INT_FIELDS = (
    "replication",
    "seed",
    "ending_backlog_units",
    "backlog_unit_days",
    "late_delivered_units",
    "reroute_count",
    "expedite_count",
    "expedited_units",
    "decision_count",
)
_REPLICATIONS_FLOAT_FIELDS = (
    "heuristic_undisrupted_cost",
    "heuristic_disrupted_cost",
    "heuristic_tcd",
    "llm_undisrupted_cost",
    "llm_disrupted_cost",
    "llm_tcd",
    "delta",
)
_DAILY_INT_FIELDS = (
    "replication",
    "day",
    "inventory_units",
    "backlog_units",
    "shipments_at_node",
    "shipments_in_transit",
    "shipments_delivered",
    "daily_demand_units",
    "daily_same_day_fulfilled_units",
    "daily_backlog_fulfilled_units",
)
_DAILY_FLOAT_FIELDS = (
    "daily_transport_cost",
    "daily_reroute_cost",
    "daily_expedite_cost",
    "daily_holding_cost",
    "daily_backlog_cost",
    "daily_late_cost",
    "cumulative_total_cost",
)


# --- loading ------------------------------------------------------------


@dataclass
class ExperimentData:
    label: str
    path: Path
    summary: dict[str, object]
    replications: list[dict[str, object]]
    run_metrics: list[dict[str, object]]
    daily_metrics: list[dict[str, object]]


def _slug(label: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in label).strip("_")


def load_replications(path: Path) -> list[dict[str, object]]:
    with (path / "replications.csv").open(newline="", encoding="utf-8") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            row["replication"] = int(raw["replication"])
            row["seed"] = int(raw["seed"])
            for field in _REPLICATIONS_FLOAT_FIELDS:
                row[field] = float(raw[field])
            rows.append(row)
        return rows


def load_run_metrics(path: Path) -> list[dict[str, object]]:
    with (path / "run_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            for field in _RUN_METRICS_FLOAT_FIELDS:
                row[field] = float(raw[field])
            for field in _RUN_METRICS_INT_FIELDS:
                row[field] = int(raw[field])
            row["terminated_with_unresolved_state"] = raw["terminated_with_unresolved_state"] == "True"
            clear_day = raw["days_to_clear_backlog_after_shock"]
            row["days_to_clear_backlog_after_shock"] = int(clear_day) if clear_day not in ("", "None") else None
            rows.append(row)
        return rows


def load_daily_metrics(path: Path) -> list[dict[str, object]]:
    with (path / "daily_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            for field in _DAILY_INT_FIELDS:
                row[field] = int(raw[field])
            for field in _DAILY_FLOAT_FIELDS:
                row[field] = float(raw[field])
            rows.append(row)
        return rows


def load_decision_behavior(path: Path) -> dict[str, dict[str, float]]:
    """Streams decision_traces.jsonl (which can be large for a real
    100-replication run) rather than holding every entry in memory: only
    the policy, proposed action_type, validity, and fallback flag are kept.
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[str, int] = defaultdict(int)
    trace_path = path / "decision_traces.jsonl"
    if not trace_path.exists():
        return {}

    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            policy = entry["policy"]
            proposed = entry["proposed_action"]
            counts[policy][proposed["action_type"]] += 1
            if not entry["proposal_validation"]["is_valid"]:
                counts[policy]["INVALID"] += 1
            if entry["fallback_invoked"]:
                counts[policy]["FALLBACK"] += 1
            totals[policy] += 1

    return {
        policy: {category: count / totals[policy] for category, count in policy_counts.items()}
        for policy, policy_counts in counts.items()
        if totals[policy] > 0
    }


def load_experiment(path: Path, label: str) -> ExperimentData:
    return ExperimentData(
        label=label,
        path=path,
        summary=json.loads((path / "summary.json").read_text(encoding="utf-8")),
        replications=load_replications(path),
        run_metrics=load_run_metrics(path),
        daily_metrics=load_daily_metrics(path),
    )


# --- shared statistics helpers -------------------------------------------


def _mean_ci(values: list[float]) -> tuple[float | None, float | None, float | None]:
    """Mean and 95% CI using the same normal approximation as
    experiments/metrics.py's summarize_experiment: mean +/- 1.96 * sd / sqrt(n).
    """
    if not values:
        return None, None, None
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, mean, mean
    margin = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - margin, mean + margin


def _percentile(sorted_values: list[float], percentile: float) -> float:
    """Deterministic linear interpolation, matching experiments/metrics.py's
    own _percentile exactly, so plotted percentiles agree with summary.json.
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
    return sorted_values[lower_index] + (sorted_values[upper_index] - sorted_values[lower_index]) * fraction


def _ecdf(values: list[float]) -> tuple[list[float], list[float]]:
    xs = sorted(values)
    n = len(xs)
    ys = [(i + 1) / n for i in range(n)]
    return xs, ys


def _run_metrics_index(run_metrics: list[dict[str, object]]) -> dict[tuple[int, str, str], dict[str, object]]:
    return {(row["replication"], row["policy"], row["run_kind"]): row for row in run_metrics}


def compute_cost_component_deltas(data: ExperimentData) -> dict[str, list[float]]:
    """Per-replication, per-component paired difference:
    (LLM disrupted-minus-undisrupted) - (heuristic disrupted-minus-undisrupted).
    """
    indexed = _run_metrics_index(data.run_metrics)
    result: dict[str, list[float]] = {component: [] for component in _COST_COMPONENTS}
    for rep_row in data.replications:
        replication = rep_row["replication"]
        try:
            heuristic_disrupted = indexed[(replication, "heuristic", "DISRUPTED")]
            heuristic_undisrupted = indexed[(replication, "heuristic", "UNDISRUPTED")]
            llm_disrupted = indexed[(replication, "llm_agent", "DISRUPTED")]
            llm_undisrupted = indexed[(replication, "llm_agent", "UNDISRUPTED")]
        except KeyError:
            continue
        for component in _COST_COMPONENTS:
            heuristic_tcd_component = heuristic_disrupted[component] - heuristic_undisrupted[component]
            llm_tcd_component = llm_disrupted[component] - llm_undisrupted[component]
            result[component].append(llm_tcd_component - heuristic_tcd_component)
    return result


# --- plot 01: paired TCD estimation ---------------------------------------


def plot_01_paired_tcd_estimation(data: ExperimentData, output_dir: Path) -> Path:
    stats = data.summary["experiment_summary"]
    heuristic = [r["heuristic_tcd"] for r in data.replications]
    llm = [r["llm_tcd"] for r in data.replications]
    deltas = [r["delta"] for r in data.replications]

    fig, (ax_pairs, ax_delta) = plt.subplots(1, 2, figsize=(11, 5))

    for h, l in zip(heuristic, llm, strict=True):
        ax_pairs.plot([0, 1], [h, l], color="gray", alpha=0.25, linewidth=0.8, zorder=1)
    ax_pairs.scatter([0] * len(heuristic), heuristic, color=_HEURISTIC_COLOR, zorder=2, s=25)
    ax_pairs.scatter([1] * len(llm), llm, color=_LLM_COLOR, zorder=2, s=25)
    ax_pairs.axhline(0, color="black", linewidth=0.6)
    ax_pairs.set_xticks([0, 1])
    ax_pairs.set_xticklabels(["Heuristic", "LLM agent"])
    ax_pairs.set_ylabel("Total Cost of Disruption (TCD)")
    ax_pairs.set_title("Paired outcomes per replication")

    bin_count = min(20, max(5, len(deltas) // 2)) if len(deltas) > 1 else 1
    ax_delta.hist(deltas, bins=bin_count, color=_LLM_COLOR, alpha=0.7, edgecolor="white")
    ax_delta.axvline(0, color="black", linewidth=1.0, label="zero")
    ax_delta.axvline(stats["mean_delta"], color="red", linestyle="--", linewidth=1.2, label=f"mean {stats['mean_delta']:.0f}")
    ax_delta.axvline(
        stats["median_delta"], color="green", linestyle=":", linewidth=1.2, label=f"median {stats['median_delta']:.0f}"
    )
    ax_delta.set_xlabel("delta = LLM TCD - heuristic TCD")
    ax_delta.set_ylabel("Replications")
    ax_delta.set_title("Delta distribution")
    ax_delta.legend(fontsize=8)

    fig.suptitle(
        f"{data.label} — mean delta {stats['mean_delta']:.0f} "
        f"[95% CI {stats['mean_delta_ci_95_lower']:.0f}, {stats['mean_delta_ci_95_upper']:.0f}] — "
        f"LLM win {stats['llm_win_rate']:.0%}, heuristic win {stats['heuristic_win_rate']:.0%}, "
        f"tie {stats['tie_rate']:.0%}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    out_path = output_dir / "01_paired_tcd_estimation.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --- plot 02: ranked replication delta ------------------------------------


def plot_02_ranked_replication_delta(data: ExperimentData, output_dir: Path) -> Path:
    stats = data.summary["experiment_summary"]
    deltas_sorted = sorted(r["delta"] for r in data.replications)
    ranks = list(range(1, len(deltas_sorted) + 1))
    colors = [_LLM_COLOR if d < 0 else (_HEURISTIC_COLOR if d > 0 else _TIE_COLOR) for d in deltas_sorted]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ranks, deltas_sorted, color=colors, width=1.0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(stats["median_delta"], color="green", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Replications ranked by delta (most negative → most positive)")
    ax.set_ylabel("delta")
    ax.set_title(
        f"Ranked replication delta — {data.label}\n"
        f"median {stats['median_delta']:.0f} | LLM win {stats['llm_win_rate']:.0%} | "
        f"p10 {stats['p10_delta']:.0f} | p90 {stats['p90_delta']:.0f} | "
        f"best LLM {stats['best_llm_delta']:.0f} | worst LLM {stats['worst_llm_delta']:.0f}",
        fontsize=9,
    )
    fig.tight_layout()

    out_path = output_dir / "02_ranked_replication_delta.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --- plot 03: TCD empirical CDF -------------------------------------------


def plot_03_tcd_ecdf(data: ExperimentData, output_dir: Path) -> Path:
    heuristic = [r["heuristic_tcd"] for r in data.replications]
    llm = [r["llm_tcd"] for r in data.replications]
    heuristic_sorted = sorted(heuristic)
    llm_sorted = sorted(llm)

    fig, ax = plt.subplots(figsize=(7, 5))
    hx, hy = _ecdf(heuristic)
    lx, ly = _ecdf(llm)
    ax.step(hx, hy, where="post", color=_HEURISTIC_COLOR, label="Heuristic")
    ax.step(lx, ly, where="post", color=_LLM_COLOR, label="LLM agent")
    ax.axvline(_percentile(heuristic_sorted, 50), color=_HEURISTIC_COLOR, linestyle=":", linewidth=1)
    ax.axvline(_percentile(llm_sorted, 50), color=_LLM_COLOR, linestyle=":", linewidth=1)
    ax.axvline(_percentile(heuristic_sorted, 90), color=_HEURISTIC_COLOR, linestyle="--", linewidth=1)
    ax.axvline(_percentile(llm_sorted, 90), color=_LLM_COLOR, linestyle="--", linewidth=1)
    ax.set_xlabel("TCD")
    ax.set_ylabel("Proportion of replications with TCD below x")
    ax.set_title(f"TCD empirical CDF — {data.label}  (dotted = median, dashed = 90th pct)", fontsize=10)
    ax.legend()
    fig.tight_layout()

    out_path = output_dir / "03_tcd_ecdf.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --- plot 04: cost-component effect forest --------------------------------


def plot_04_cost_component_effects(data: ExperimentData, output_dir: Path) -> Path:
    component_deltas = compute_cost_component_deltas(data)
    rows = []
    for component in _COST_COMPONENTS:
        mean, lower, upper = _mean_ci(component_deltas[component])
        rows.append((component, mean or 0.0, lower or 0.0, upper or 0.0))

    fig, ax = plt.subplots(figsize=(7, 5))
    y_positions = list(range(len(rows)))
    means = [row[1] for row in rows]
    errors_low = [row[1] - row[2] for row in rows]
    errors_high = [row[3] - row[1] for row in rows]
    colors = [_LLM_COLOR if m < 0 else _HEURISTIC_COLOR for m in means]
    ax.errorbar(
        means, y_positions, xerr=[errors_low, errors_high], fmt="none", ecolor="black", capsize=4, zorder=2
    )
    ax.scatter(means, y_positions, color=colors, zorder=3, s=70)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row[0].replace("_cost", "") for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Paired difference in TCD component (LLM - heuristic), mean + 95% CI")
    ax.set_title(f"Cost-component effects — {data.label}")
    fig.tight_layout()

    out_path = output_dir / "04_cost_component_effects.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --- plot 05: cost-service trade-off ---------------------------------------


def plot_05_cost_service_tradeoff(data: ExperimentData, output_dir: Path) -> Path:
    indexed = _run_metrics_index(data.run_metrics)
    heuristic_points: list[tuple[float, float]] = []
    llm_points: list[tuple[float, float]] = []
    lines: list[tuple[tuple[float, float], tuple[float, float]]] = []

    for rep_row in data.replications:
        replication = rep_row["replication"]
        heuristic_row = indexed.get((replication, "heuristic", "DISRUPTED"))
        llm_row = indexed.get((replication, "llm_agent", "DISRUPTED"))
        if heuristic_row is None or llm_row is None:
            continue
        heuristic_point = (rep_row["heuristic_tcd"], heuristic_row["same_day_fill_rate"])
        llm_point = (rep_row["llm_tcd"], llm_row["same_day_fill_rate"])
        heuristic_points.append(heuristic_point)
        llm_points.append(llm_point)
        lines.append((heuristic_point, llm_point))

    fig, ax = plt.subplots(figsize=(7, 6))
    for (x1, y1), (x2, y2) in lines:
        ax.plot([x1, x2], [y1, y2], color="gray", alpha=0.2, linewidth=0.7, zorder=1)
    ax.scatter(
        [p[0] for p in heuristic_points], [p[1] for p in heuristic_points], color=_HEURISTIC_COLOR, label="Heuristic", zorder=2, s=20
    )
    ax.scatter([p[0] for p in llm_points], [p[1] for p in llm_points], color=_LLM_COLOR, label="LLM agent", zorder=2, s=20)
    ax.set_xlabel("TCD (lower is better)")
    ax.set_ylabel("Same-day fill rate under disruption (higher is better)")
    ax.set_title(f"Cost vs. service trade-off — {data.label}\n(desirable region: upper-left)", fontsize=10)
    ax.legend()
    fig.tight_layout()

    out_path = output_dir / "05_cost_service_tradeoff.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --- plot 06: disruption and recovery dynamics ------------------------------


def _daily_grouped(
    daily_metrics: list[dict[str, object]],
) -> dict[tuple[str, str, int], list[dict[str, object]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in daily_metrics:
        grouped[(row["policy"], row["run_kind"], row["day"])].append(row)
    return grouped


def _find_disruption_window(daily_metrics: list[dict[str, object]]) -> tuple[int | None, int | None]:
    active_days = sorted(
        {row["day"] for row in daily_metrics if row["run_kind"] == "DISRUPTED" and row["active_shock_ids"]}
    )
    if not active_days:
        return None, None
    return active_days[0], active_days[-1]


def plot_06_recovery_dynamics(data: ExperimentData, output_dir: Path) -> Path:
    grouped = _daily_grouped(data.daily_metrics)
    start_day, end_day = _find_disruption_window(data.daily_metrics)

    undisrupted_cost_index: dict[tuple[str, int, int], float] = {}
    for row in data.daily_metrics:
        if row["run_kind"] == "UNDISRUPTED":
            undisrupted_cost_index[(row["policy"], row["day"], row["replication"])] = row["cumulative_total_cost"]

    all_days = sorted({row["day"] for row in data.daily_metrics if row["run_kind"] == "DISRUPTED"})
    policies = [p for p in ("heuristic", "llm_agent") if any(row["policy"] == p for row in data.daily_metrics)]

    backlog_series: dict[str, list[tuple[float | None, float | None, float | None]]] = {p: [] for p in policies}
    inventory_series: dict[str, list[tuple[float | None, float | None, float | None]]] = {p: [] for p in policies}
    fill_series: dict[str, list[tuple[float | None, float | None, float | None]]] = {p: [] for p in policies}
    incremental_cost_series: dict[str, list[tuple[float | None, float | None, float | None]]] = {p: [] for p in policies}

    x_axis = [day - start_day if start_day is not None else day for day in all_days]
    for day in all_days:
        for policy in policies:
            rows = grouped.get((policy, "DISRUPTED", day), [])
            backlog_series[policy].append(_mean_ci([row["backlog_units"] for row in rows]))
            inventory_series[policy].append(_mean_ci([row["inventory_units"] for row in rows]))
            fill_values = [
                row["daily_same_day_fulfilled_units"] / row["daily_demand_units"]
                for row in rows
                if row["daily_demand_units"] > 0
            ]
            fill_series[policy].append(_mean_ci(fill_values))
            incremental_values = []
            for row in rows:
                undisrupted_cost = undisrupted_cost_index.get((policy, day, row["replication"]))
                if undisrupted_cost is not None:
                    incremental_values.append(row["cumulative_total_cost"] - undisrupted_cost)
            incremental_cost_series[policy].append(_mean_ci(incremental_values))

    fig, axes = plt.subplots(4, 1, figsize=(9, 14), sharex=True)
    panels = (
        (axes[0], backlog_series, "Mean backlog (units)"),
        (axes[1], inventory_series, "Mean destination inventory (units)"),
        (axes[2], incremental_cost_series, "Cumulative incremental cost\n(disrupted - undisrupted)"),
        (axes[3], fill_series, "Same-day fill rate"),
    )
    for ax, series, ylabel in panels:
        for policy in policies:
            means = [point[0] for point in series[policy]]
            lowers = [point[1] for point in series[policy]]
            uppers = [point[2] for point in series[policy]]
            xs = [x for x, mean in zip(x_axis, means, strict=True) if mean is not None]
            ys = [mean for mean in means if mean is not None]
            los = [lo for lo, mean in zip(lowers, means, strict=True) if mean is not None]
            his = [hi for hi, mean in zip(uppers, means, strict=True) if mean is not None]
            ax.plot(xs, ys, color=_POLICY_COLORS[policy], label=_POLICY_LABELS[policy])
            if los and his:
                ax.fill_between(xs, los, his, color=_POLICY_COLORS[policy], alpha=0.2)
        if start_day is not None and end_day is not None:
            ax.axvspan(0, end_day - start_day, color="gray", alpha=0.12)
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.axvline(end_day - start_day, color="black", linestyle="--", linewidth=0.8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Days relative to disruption start")
    fig.suptitle(f"Disruption and recovery dynamics — {data.label}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    out_path = output_dir / "06_recovery_dynamics.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --- plot 07: scenario-level policy effect forest (cross-experiment) -------


def plot_07_policy_effect_by_disruption(datasets: list[ExperimentData], output_dir: Path) -> Path:
    rows = []
    for data in datasets:
        stats = data.summary["experiment_summary"]
        rows.append(
            (
                data.label,
                stats["mean_delta"],
                stats["mean_delta_ci_95_lower"],
                stats["mean_delta_ci_95_upper"],
                stats["llm_win_rate"],
                stats["replication_count"],
            )
        )
    rows.sort(key=lambda row: row[1])  # strongest LLM advantage (most negative) first

    means = [row[1] for row in rows]
    errors_low = [row[1] - row[2] for row in rows]
    errors_high = [row[3] - row[1] for row in rows]
    colors = [_LLM_COLOR if mean < 0 else _HEURISTIC_COLOR for mean in means]

    fig, ax = plt.subplots(figsize=(8, 1.5 + len(rows)))
    y_positions = list(range(len(rows)))
    ax.errorbar(means, y_positions, xerr=[errors_low, errors_high], fmt="none", ecolor="black", capsize=4, zorder=2)
    ax.scatter(means, y_positions, color=colors, s=90, zorder=3)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{row[0]}  (LLM win {row[4]:.0%}, n={row[5]})" for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Mean delta (95% CI) — negative = LLM cheaper")
    ax.set_title("Policy effect by disruption profile")
    fig.tight_layout()

    out_path = output_dir / "07_policy_effect_by_disruption.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --- plot 09: policy-behavior heatmap ---------------------------------------


def plot_09_policy_behavior_heatmap(data: ExperimentData, output_dir: Path) -> Path | None:
    behavior = load_decision_behavior(data.path)
    if not behavior:
        return None
    policies = [p for p in ("heuristic", "llm_agent") if p in behavior]
    matrix = [[behavior[p].get(category, 0.0) * 100 for category in _HEATMAP_CATEGORIES] for p in policies]

    fig, ax = plt.subplots(figsize=(8, 2 + len(policies)))
    image = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(_HEATMAP_CATEGORIES)))
    ax.set_xticklabels(_HEATMAP_CATEGORIES)
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels([_POLICY_LABELS[p] for p in policies])
    for i in range(len(policies)):
        for j in range(len(_HEATMAP_CATEGORIES)):
            ax.text(j, i, f"{matrix[i][j]:.1f}%", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="% of decision opportunities")
    ax.set_title(f"Policy behavior — {data.label}\n(WAIT/REROUTE/EXPEDITE/ABSTAIN sum to 100%; INVALID/FALLBACK are independent rates)", fontsize=9)
    fig.tight_layout()

    out_path = output_dir / "09_policy_behavior_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --- plot 10: tail-risk ------------------------------------------------------


def plot_10_tcd_tail_risk(data: ExperimentData, output_dir: Path) -> Path:
    heuristic_sorted = sorted(r["heuristic_tcd"] for r in data.replications)
    llm_sorted = sorted(r["llm_tcd"] for r in data.replications)
    percentile_labels = ("median", "p75", "p90", "p95", "max")
    percentile_values = (50, 75, 90, 95, 100)

    def _tail_values(sorted_values: list[float]) -> list[float]:
        return [sorted_values[-1] if pct == 100 else _percentile(sorted_values, pct) for pct in percentile_values]

    heuristic_vals = _tail_values(heuristic_sorted)
    llm_vals = _tail_values(llm_sorted)

    fig, ax = plt.subplots(figsize=(7, 5))
    x_positions = range(len(percentile_labels))
    ax.plot(x_positions, heuristic_vals, marker="o", color=_HEURISTIC_COLOR, label="Heuristic")
    ax.plot(x_positions, llm_vals, marker="o", color=_LLM_COLOR, label="LLM agent")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(percentile_labels)
    ax.set_ylabel("TCD")
    ax.set_title(f"Tail risk — {data.label}")
    ax.legend()

    by_delta = sorted(data.replications, key=lambda r: r["delta"])
    best_llm = by_delta[0]
    worst_llm = by_delta[-1]
    median_delta = statistics.median(r["delta"] for r in data.replications)
    closest_to_median = min(data.replications, key=lambda r: abs(r["delta"] - median_delta))
    note = (
        f"Largest LLM win: rep {best_llm['replication']} (delta {best_llm['delta']:.0f})\n"
        f"Closest to median: rep {closest_to_median['replication']} (delta {closest_to_median['delta']:.0f})\n"
        f"Largest LLM loss: rep {worst_llm['replication']} (delta {worst_llm['delta']:.0f})"
    )
    ax.text(
        0.02,
        0.02,
        note,
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    fig.tight_layout()

    out_path = output_dir / "10_tcd_tail_risk.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --- CLI ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot supply-chain-agent-evaluation experiment results "
        "per VISUAL_REPORTING_PLAN_ESSENTIAL.md."
    )
    parser.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        required=True,
        metavar="PATH",
        help="Path to one experiment's output directory. Repeatable.",
    )
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        required=True,
        metavar="NAME",
        help="Label for the matching --experiment, same order, same count.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/plots"),
        help="Directory to write PNG files into (default: analysis/plots).",
    )
    args = parser.parse_args(argv)

    if len(args.experiments) != len(args.labels):
        parser.error("--experiment and --label must be given the same number of times")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [
        load_experiment(Path(path), label)
        for path, label in zip(args.experiments, args.labels, strict=True)
    ]

    written: list[Path] = []
    for data in datasets:
        scenario_dir = args.output_dir / _slug(data.label)
        scenario_dir.mkdir(parents=True, exist_ok=True)
        written.append(plot_01_paired_tcd_estimation(data, scenario_dir))
        written.append(plot_02_ranked_replication_delta(data, scenario_dir))
        written.append(plot_03_tcd_ecdf(data, scenario_dir))
        written.append(plot_04_cost_component_effects(data, scenario_dir))
        written.append(plot_05_cost_service_tradeoff(data, scenario_dir))
        written.append(plot_06_recovery_dynamics(data, scenario_dir))
        behavior_plot = plot_09_policy_behavior_heatmap(data, scenario_dir)
        if behavior_plot is not None:
            written.append(behavior_plot)
        written.append(plot_10_tcd_tail_risk(data, scenario_dir))

    if len(datasets) >= 2:
        written.append(plot_07_policy_effect_by_disruption(datasets, args.output_dir))
        print(
            "Skipping 08_disruption_robustness_heatmap.png: needs disruption profiles to form a "
            "severity x duration grid; current profiles vary by mechanism, not a crossed grid."
        )

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
