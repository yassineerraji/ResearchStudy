"""Turns one or more experiment output directories into the plot set defined

Standalone script, deliberately outside src/supply_chain_simulator: it only
ever reads the CSV/JSON/JSONL files data_io/writers.py already produces
after a run and never touches simulation behavior, so it has no bearing on
scientific validity or fairness. Given one --experiment/--label pair, it
produces the eight per-scenario plots (01, 02, 03, 04, 05, 06, 09, 10). Given
two or more, it also produces the cross-scenario forest plot (07).

Given one or more --cell TOPOLOGY SEVERITY PATH triples (V2's topology x
severity grid, CLAUDE.md V2.8.1), it additionally produces six grid-level
plots (08, 11-15) into <output-dir>/grid/: a significance-masked mean-delta
heatmap, a win-rate heatmap, a topology x severity interaction plot, a
cross-cell win/loss/tie summary, a signal-to-noise-by-cell chart (the direct
check on whether V2's redesign fixed the 100%/0%-win-rate problem V1's audit
found), and a QA plot confirming the realized shock/quantity randomness
matches what each cell's config describes. Each --cell also gets the normal
per-scenario plot set, labeled "<Topology> x <Severity>". The grid need not
be all nine cells -- missing combinations render as blank, labeled cells.

Requires the optional "analysis" dependency group
(`pip install -e ".[analysis]"`) since it is the only place in this project
that uses matplotlib.
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

_HEURISTIC_COLOR = "#4c72b0"
_LLM_COLOR = "#dd8452"
_TIE_COLOR = "#8c8c8c"

# Grid axes (CLAUDE.md V2.3.1, V2.8.1). Fixed, deterministic order -- never
# derived from dict/insertion order, per this project's own V1 SS3 convention.
TOPOLOGY_ORDER = ("Compact", "Standard", "Extended")
SEVERITY_ORDER = ("Light", "Medium", "Heavy")

# One fixed hue per topology tier, chosen distinct from _HEURISTIC_COLOR/
# _LLM_COLOR (those two are reserved project-wide for policy identity; reusing
# either here would make a topology-tier line read as a policy line).
# Validated with the dataviz skill's validate_palette.js: all three checks
# pass; the e7298a<->7570b3 adjacent pair sits in the 6-8 CVD-floor band,
# which is legal only with secondary encoding -- covered here by giving each
# tier both a distinct marker shape and a direct end-of-line label.
_TOPOLOGY_COLORS = {"Compact": "#1b9e77", "Standard": "#7570b3", "Extended": "#e7298a"}
_TOPOLOGY_MARKERS = {"Compact": "o", "Standard": "s", "Extended": "^"}

# Diverging: LLM-favorable (negative delta) <-> neutral <-> heuristic-favorable
# (positive delta), built from the same two hues used everywhere else in this
# file so a reader who already knows "orange = LLM, blue = heuristic" from
# plots 01-10 (e.g. plot_02's `_LLM_COLOR if d < 0 else _HEURISTIC_COLOR`)
# does not have to learn a second color language for the grid. Low value
# (vmin, negative delta) must map to _LLM_COLOR and high value (vmax,
# positive delta) to _HEURISTIC_COLOR to match that existing sign convention.
_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "llm_heuristic_diverging", [_LLM_COLOR, "#f5f3ee", _HEURISTIC_COLOR]
)
# Sequential: white -> LLM color, used only for "share of replications the
# LLM won" (a magnitude, not a polarity -- one hue, light to dark).
_LLM_SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "llm_win_rate", ["#f5f3ee", _LLM_COLOR]
)

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
            row["terminated_with_unresolved_state"] = (
                raw["terminated_with_unresolved_state"] == "True"
            )
            clear_day = raw["days_to_clear_backlog_after_shock"]
            row["days_to_clear_backlog_after_shock"] = (
                int(clear_day) if clear_day not in ("", "None") else None
            )
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
        policy: {
            category: count / totals[policy]
            for category, count in policy_counts.items()
        }
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


@dataclass
class RealizedRandomness:
    """One entry per realized shock instance / release event, pooled across
    every replication in one experiment. Durations/delays are derived here
    (physical_end_day - physical_start_day + 1, information_day -
    physical_start_day) rather than stored duplicated per-shock in the event
    tape.
    """

    start_days: list[int]
    duration_days: list[int]
    information_delay_days: list[int]
    release_quantities: list[int]


def load_realized_randomness(path: Path) -> RealizedRandomness:
    """Streams event_tapes.jsonl (one line per replication) rather than
    loading it whole, matching load_decision_behavior's approach for the
    other large per-replication file.
    """
    start_days: list[int] = []
    duration_days: list[int] = []
    information_delay_days: list[int] = []
    release_quantities: list[int] = []

    tape_path = path / "event_tapes.jsonl"
    if not tape_path.exists():
        return RealizedRandomness([], [], [], [])

    with tape_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            for shock in entry["shocks"]:
                start_days.append(shock["physical_start_day"])
                duration_days.append(
                    shock["physical_end_day"] - shock["physical_start_day"] + 1
                )
                information_delay_days.append(
                    shock["information_day"] - shock["physical_start_day"]
                )
            for release in entry["shipment_release_events"]:
                release_quantities.append(release["quantity"])

    return RealizedRandomness(
        start_days, duration_days, information_delay_days, release_quantities
    )


# A grid cell is addressed by (topology, severity); missing combinations are
# simply absent keys, so a partial grid (e.g. calibration-only cells) is a
# first-class case, not an error.
GridCells = dict[tuple[str, str], ExperimentData]


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
    return (
        sorted_values[lower_index]
        + (sorted_values[upper_index] - sorted_values[lower_index]) * fraction
    )


def _ecdf(values: list[float]) -> tuple[list[float], list[float]]:
    xs = sorted(values)
    n = len(xs)
    ys = [(i + 1) / n for i in range(n)]
    return xs, ys


def _run_metrics_index(
    run_metrics: list[dict[str, object]],
) -> dict[tuple[int, str, str], dict[str, object]]:
    return {
        (row["replication"], row["policy"], row["run_kind"]): row for row in run_metrics
    }


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
            heuristic_tcd_component = (
                heuristic_disrupted[component] - heuristic_undisrupted[component]
            )
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
    ax_pairs.scatter(
        [0] * len(heuristic), heuristic, color=_HEURISTIC_COLOR, zorder=2, s=25
    )
    ax_pairs.scatter([1] * len(llm), llm, color=_LLM_COLOR, zorder=2, s=25)
    ax_pairs.axhline(0, color="black", linewidth=0.6)
    ax_pairs.set_xticks([0, 1])
    ax_pairs.set_xticklabels(["Heuristic", "LLM agent"])
    ax_pairs.set_ylabel("Total Cost of Disruption (TCD)")
    ax_pairs.set_title("Paired outcomes per replication")

    bin_count = min(20, max(5, len(deltas) // 2)) if len(deltas) > 1 else 1
    ax_delta.hist(
        deltas, bins=bin_count, color=_LLM_COLOR, alpha=0.7, edgecolor="white"
    )
    ax_delta.axvline(0, color="black", linewidth=1.0, label="zero")
    ax_delta.axvline(
        stats["mean_delta"],
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"mean {stats['mean_delta']:.0f}",
    )
    ax_delta.axvline(
        stats["median_delta"],
        color="green",
        linestyle=":",
        linewidth=1.2,
        label=f"median {stats['median_delta']:.0f}",
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
    colors = [
        _LLM_COLOR if d < 0 else (_HEURISTIC_COLOR if d > 0 else _TIE_COLOR)
        for d in deltas_sorted
    ]

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
    ax.axvline(
        _percentile(heuristic_sorted, 50),
        color=_HEURISTIC_COLOR,
        linestyle=":",
        linewidth=1,
    )
    ax.axvline(
        _percentile(llm_sorted, 50), color=_LLM_COLOR, linestyle=":", linewidth=1
    )
    ax.axvline(
        _percentile(heuristic_sorted, 90),
        color=_HEURISTIC_COLOR,
        linestyle="--",
        linewidth=1,
    )
    ax.axvline(
        _percentile(llm_sorted, 90), color=_LLM_COLOR, linestyle="--", linewidth=1
    )
    ax.set_xlabel("TCD")
    ax.set_ylabel("Proportion of replications with TCD below x")
    ax.set_title(
        f"TCD empirical CDF — {data.label}  (dotted = median, dashed = 90th pct)",
        fontsize=10,
    )
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
        means,
        y_positions,
        xerr=[errors_low, errors_high],
        fmt="none",
        ecolor="black",
        capsize=4,
        zorder=2,
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
        heuristic_point = (
            rep_row["heuristic_tcd"],
            heuristic_row["same_day_fill_rate"],
        )
        llm_point = (rep_row["llm_tcd"], llm_row["same_day_fill_rate"])
        heuristic_points.append(heuristic_point)
        llm_points.append(llm_point)
        lines.append((heuristic_point, llm_point))

    fig, ax = plt.subplots(figsize=(7, 6))
    for (x1, y1), (x2, y2) in lines:
        ax.plot([x1, x2], [y1, y2], color="gray", alpha=0.2, linewidth=0.7, zorder=1)
    ax.scatter(
        [p[0] for p in heuristic_points],
        [p[1] for p in heuristic_points],
        color=_HEURISTIC_COLOR,
        label="Heuristic",
        zorder=2,
        s=20,
    )
    ax.scatter(
        [p[0] for p in llm_points],
        [p[1] for p in llm_points],
        color=_LLM_COLOR,
        label="LLM agent",
        zorder=2,
        s=20,
    )
    ax.set_xlabel("TCD (lower is better)")
    ax.set_ylabel("Same-day fill rate under disruption (higher is better)")
    ax.set_title(
        f"Cost vs. service trade-off — {data.label}\n(desirable region: upper-left)",
        fontsize=10,
    )
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


def _find_disruption_window(
    daily_metrics: list[dict[str, object]],
) -> tuple[int | None, int | None]:
    active_days = sorted(
        {
            row["day"]
            for row in daily_metrics
            if row["run_kind"] == "DISRUPTED" and row["active_shock_ids"]
        }
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
            undisrupted_cost_index[(row["policy"], row["day"], row["replication"])] = (
                row["cumulative_total_cost"]
            )

    all_days = sorted(
        {row["day"] for row in data.daily_metrics if row["run_kind"] == "DISRUPTED"}
    )
    policies = [
        p
        for p in ("heuristic", "llm_agent")
        if any(row["policy"] == p for row in data.daily_metrics)
    ]

    backlog_series: dict[str, list[tuple[float | None, float | None, float | None]]] = {
        p: [] for p in policies
    }
    inventory_series: dict[
        str, list[tuple[float | None, float | None, float | None]]
    ] = {p: [] for p in policies}
    fill_series: dict[str, list[tuple[float | None, float | None, float | None]]] = {
        p: [] for p in policies
    }
    incremental_cost_series: dict[
        str, list[tuple[float | None, float | None, float | None]]
    ] = {p: [] for p in policies}

    x_axis = [day - start_day if start_day is not None else day for day in all_days]
    for day in all_days:
        for policy in policies:
            rows = grouped.get((policy, "DISRUPTED", day), [])
            backlog_series[policy].append(
                _mean_ci([row["backlog_units"] for row in rows])
            )
            inventory_series[policy].append(
                _mean_ci([row["inventory_units"] for row in rows])
            )
            fill_values = [
                row["daily_same_day_fulfilled_units"] / row["daily_demand_units"]
                for row in rows
                if row["daily_demand_units"] > 0
            ]
            fill_series[policy].append(_mean_ci(fill_values))
            incremental_values = []
            for row in rows:
                undisrupted_cost = undisrupted_cost_index.get(
                    (policy, day, row["replication"])
                )
                if undisrupted_cost is not None:
                    incremental_values.append(
                        row["cumulative_total_cost"] - undisrupted_cost
                    )
            incremental_cost_series[policy].append(_mean_ci(incremental_values))

    fig, axes = plt.subplots(4, 1, figsize=(9, 14), sharex=True)
    panels = (
        (axes[0], backlog_series, "Mean backlog (units)"),
        (axes[1], inventory_series, "Mean destination inventory (units)"),
        (
            axes[2],
            incremental_cost_series,
            "Cumulative incremental cost\n(disrupted - undisrupted)",
        ),
        (axes[3], fill_series, "Same-day fill rate"),
    )
    for ax, series, ylabel in panels:
        for policy in policies:
            means = [point[0] for point in series[policy]]
            lowers = [point[1] for point in series[policy]]
            uppers = [point[2] for point in series[policy]]
            xs = [x for x, mean in zip(x_axis, means, strict=True) if mean is not None]
            ys = [mean for mean in means if mean is not None]
            los = [
                lo for lo, mean in zip(lowers, means, strict=True) if mean is not None
            ]
            his = [
                hi for hi, mean in zip(uppers, means, strict=True) if mean is not None
            ]
            ax.plot(xs, ys, color=_POLICY_COLORS[policy], label=_POLICY_LABELS[policy])
            if los and his:
                ax.fill_between(xs, los, his, color=_POLICY_COLORS[policy], alpha=0.2)
        if start_day is not None and end_day is not None:
            ax.axvspan(0, end_day - start_day, color="gray", alpha=0.12)
            ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
            ax.axvline(
                end_day - start_day, color="black", linestyle="--", linewidth=0.8
            )
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


def plot_07_policy_effect_by_disruption(
    datasets: list[ExperimentData], output_dir: Path
) -> Path:
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
    ax.errorbar(
        means,
        y_positions,
        xerr=[errors_low, errors_high],
        fmt="none",
        ecolor="black",
        capsize=4,
        zorder=2,
    )
    ax.scatter(means, y_positions, color=colors, s=90, zorder=3)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [f"{row[0]}  (LLM win {row[4]:.0%}, n={row[5]})" for row in rows]
    )
    ax.invert_yaxis()
    ax.set_xlabel("Mean delta (95% CI) — negative = LLM cheaper")
    ax.set_title("Policy effect by disruption profile")
    fig.tight_layout()

    out_path = output_dir / "07_policy_effect_by_disruption.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --- plot 09: policy-behavior heatmap ---------------------------------------


def plot_09_policy_behavior_heatmap(
    data: ExperimentData, output_dir: Path
) -> Path | None:
    behavior = load_decision_behavior(data.path)
    if not behavior:
        return None
    policies = [p for p in ("heuristic", "llm_agent") if p in behavior]
    matrix = [
        [behavior[p].get(category, 0.0) * 100 for category in _HEATMAP_CATEGORIES]
        for p in policies
    ]

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
    ax.set_title(
        f"Policy behavior — {data.label}\n(WAIT/REROUTE/EXPEDITE/ABSTAIN sum to 100%; INVALID/FALLBACK are independent rates)",
        fontsize=9,
    )
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
        return [
            sorted_values[-1] if pct == 100 else _percentile(sorted_values, pct)
            for pct in percentile_values
        ]

    heuristic_vals = _tail_values(heuristic_sorted)
    llm_vals = _tail_values(llm_sorted)

    fig, ax = plt.subplots(figsize=(7, 5))
    x_positions = range(len(percentile_labels))
    ax.plot(
        x_positions,
        heuristic_vals,
        marker="o",
        color=_HEURISTIC_COLOR,
        label="Heuristic",
    )
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
    closest_to_median = min(
        data.replications, key=lambda r: abs(r["delta"] - median_delta)
    )
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


# --- grid plots (V2 topology x severity, CLAUDE.md V2.8.1) -----------------


def _grid_stats(
    grid: GridCells, topology: str, severity: str
) -> dict[str, object] | None:
    data = grid.get((topology, severity))
    if data is None:
        return None
    stats: dict[str, object] = data.summary["experiment_summary"]
    return stats


def _is_significant(stats: dict[str, object]) -> bool:
    """95% CI on mean delta excludes zero -- i.e. distinguishable from a tie
    at the same confidence level summary.json already reports, not a new
    statistical test.
    """
    lower = stats["mean_delta_ci_95_lower"]
    upper = stats["mean_delta_ci_95_upper"]
    return bool(lower > 0 or upper < 0)


def _draw_grid_axes(ax: plt.Axes, title: str) -> None:
    ax.set_xticks(range(len(SEVERITY_ORDER)))
    ax.set_xticklabels(SEVERITY_ORDER)
    ax.set_yticks(range(len(TOPOLOGY_ORDER)))
    ax.set_yticklabels(TOPOLOGY_ORDER)
    ax.set_xlabel("Severity")
    ax.set_ylabel("Topology")
    ax.set_title(title, fontsize=10)


def _mark_missing_cell(ax: plt.Axes, row: int, col: int) -> None:
    ax.add_patch(
        Rectangle(
            (col - 0.5, row - 0.5),
            1,
            1,
            facecolor="#e8e6df",
            edgecolor="#b8b6ae",
            hatch="xxx",
            zorder=2,
        )
    )
    ax.text(
        col,
        row,
        "no\ndata",
        ha="center",
        va="center",
        fontsize=8,
        color="#8c8c8c",
        zorder=3,
    )


def plot_08_disruption_robustness_heatmap(grid: GridCells, output_dir: Path) -> Path:
    """The one plot Milestone 15 (CLAUDE.md V2.10) exists to produce: a
    topology x severity heatmap of mean delta, color = magnitude/direction,
    hatch overlay = "95% CI includes zero" (not distinguishable from a tie
    at this sample size -- material at 33, and expected on 3-replication
    calibration data). Missing cells (grid not yet run for that combination)
    render as a labeled blank rather than silently as zero.
    """
    matrix = [[float("nan")] * len(SEVERITY_ORDER) for _ in TOPOLOGY_ORDER]
    present_means = []
    for row, topology in enumerate(TOPOLOGY_ORDER):
        for col, severity in enumerate(SEVERITY_ORDER):
            stats = _grid_stats(grid, topology, severity)
            if stats is not None:
                matrix[row][col] = stats["mean_delta"]
                present_means.append(stats["mean_delta"])

    span = max((abs(m) for m in present_means), default=1.0) or 1.0

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(
        matrix, cmap=_DIVERGING_CMAP, vmin=-span, vmax=span, aspect="auto", zorder=1
    )
    fig.colorbar(image, ax=ax, label="mean delta (negative = LLM cheaper)")

    for row, topology in enumerate(TOPOLOGY_ORDER):
        for col, severity in enumerate(SEVERITY_ORDER):
            stats = _grid_stats(grid, topology, severity)
            if stats is None:
                _mark_missing_cell(ax, row, col)
                continue
            n = stats["replication_count"]
            significant = _is_significant(stats)
            if not significant:
                ax.add_patch(
                    Rectangle(
                        (col - 0.5, row - 0.5),
                        1,
                        1,
                        facecolor="none",
                        edgecolor="#5a5a5a",
                        hatch="///",
                        zorder=2,
                    )
                )
            label = f"{stats['mean_delta']:.0f}\nn={n}" + (
                "" if significant else "\n(n.s.)"
            )
            weight = "bold" if significant else "normal"
            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight=weight,
                zorder=3,
            )

    _draw_grid_axes(
        ax, "Disruption-robustness heatmap: mean delta by topology x severity"
    )
    fig.text(
        0.02,
        0.01,
        "Hatched = 95% CI on mean delta includes zero (not distinguishable from a tie at this n).\n"
        "Blue = heuristic cheaper, orange = LLM cheaper. n<10 cells are calibration-stage, low-power estimates.",
        fontsize=7.5,
        color="#52514e",
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))

    out_path = output_dir / "08_disruption_robustness_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_11_win_rate_heatmap(grid: GridCells, output_dir: Path) -> Path:
    """Companion to plot 08: magnitude (delta) and frequency (win rate) tell
    different stories -- a policy can win often by a little, or rarely by a
    lot -- so this is deliberately a separate figure rather than a second
    color channel on the same heatmap (dataviz skill: one axis, no dual
    encodings on one mark).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    matrix = [[float("nan")] * len(SEVERITY_ORDER) for _ in TOPOLOGY_ORDER]
    for row, topology in enumerate(TOPOLOGY_ORDER):
        for col, severity in enumerate(SEVERITY_ORDER):
            stats = _grid_stats(grid, topology, severity)
            if stats is not None:
                matrix[row][col] = stats["llm_win_rate"] * 100

    image = ax.imshow(
        matrix, cmap=_LLM_SEQUENTIAL_CMAP, vmin=0, vmax=100, aspect="auto", zorder=1
    )
    fig.colorbar(image, ax=ax, label="LLM win rate (%)")

    for row, topology in enumerate(TOPOLOGY_ORDER):
        for col, severity in enumerate(SEVERITY_ORDER):
            stats = _grid_stats(grid, topology, severity)
            if stats is None:
                _mark_missing_cell(ax, row, col)
                continue
            label = (
                f"LLM {stats['llm_win_rate']:.0%}\n"
                f"tie {stats['tie_rate']:.0%}\n"
                f"heur. {stats['heuristic_win_rate']:.0%}\n"
                f"n={stats['replication_count']}"
            )
            ax.text(col, row, label, ha="center", va="center", fontsize=7.5, zorder=3)

    _draw_grid_axes(ax, "LLM win rate by topology x severity")
    fig.tight_layout()

    out_path = output_dir / "11_win_rate_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_12_topology_severity_interaction(grid: GridCells, output_dir: Path) -> Path:
    """Classic interaction plot: parallel lines mean topology and severity
    act independently on the LLM's advantage; converging/crossing lines mean
    they interact -- exactly the question V2's mission delta (CLAUDE.md
    V2.1) asks ("how does that advantage vary with network complexity and
    disruption character").
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x_positions = list(range(len(SEVERITY_ORDER)))

    for topology in TOPOLOGY_ORDER:
        xs, ys, los, his = [], [], [], []
        for col, severity in enumerate(SEVERITY_ORDER):
            stats = _grid_stats(grid, topology, severity)
            if stats is None:
                continue
            xs.append(col)
            ys.append(stats["mean_delta"])
            los.append(stats["mean_delta_ci_95_lower"])
            his.append(stats["mean_delta_ci_95_upper"])
        if not xs:
            continue
        color = _TOPOLOGY_COLORS[topology]
        ax.plot(
            xs,
            ys,
            color=color,
            marker=_TOPOLOGY_MARKERS[topology],
            markersize=7,
            linewidth=2,
            label=topology,
            zorder=3,
        )
        ax.fill_between(xs, los, his, color=color, alpha=0.15, zorder=1)
        ax.text(
            xs[-1] + 0.08,
            ys[-1],
            topology,
            color=color,
            fontsize=9,
            va="center",
            fontweight="bold",
        )

    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(SEVERITY_ORDER)
    ax.set_xlim(-0.3, len(SEVERITY_ORDER) - 0.3)
    ax.set_xlabel("Severity")
    ax.set_ylabel("Mean delta (negative = LLM cheaper), 95% CI band")
    ax.set_title("Topology x severity interaction")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    out_path = output_dir / "12_topology_severity_interaction.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_13_grid_win_loss_tie(grid: GridCells, output_dir: Path) -> Path:
    """Frequency companion to the magnitude-focused heatmap/interaction
    plots: one stacked bar per present cell, fixed topology-then-severity
    order (never sorted by value -- this project never relies on an
    incidental ordering, V1 SS3).
    """
    rows = []
    for topology in TOPOLOGY_ORDER:
        for severity in SEVERITY_ORDER:
            stats = _grid_stats(grid, topology, severity)
            if stats is not None:
                rows.append((f"{topology} x {severity}", stats))

    fig, ax = plt.subplots(figsize=(8, 1 + 0.55 * len(rows)))
    y_positions = list(range(len(rows)))
    for y, (label, stats) in zip(y_positions, rows, strict=True):
        llm = stats["llm_win_rate"] * 100
        tie = stats["tie_rate"] * 100
        heuristic = stats["heuristic_win_rate"] * 100
        ax.barh(y, llm, color=_LLM_COLOR, label="LLM win" if y == 0 else None)
        ax.barh(y, tie, left=llm, color=_TIE_COLOR, label="Tie" if y == 0 else None)
        ax.barh(
            y,
            heuristic,
            left=llm + tie,
            color=_HEURISTIC_COLOR,
            label="Heuristic win" if y == 0 else None,
        )
        ax.text(
            101,
            y,
            f"n={stats['replication_count']}",
            va="center",
            fontsize=7.5,
            color="#52514e",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([row[0] for row in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel("Share of replications (%)")
    ax.set_title("Win / tie / loss rate by grid cell")
    ax.legend(loc="lower right", fontsize=8, ncol=3)
    fig.tight_layout()

    out_path = output_dir / "13_grid_win_loss_tie.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_14_signal_to_noise_by_cell(
    grid: GridCells, output_dir: Path, v1_reference: list[ExperimentData]
) -> Path:
    """Direct check on the thing V2 exists to fix (CLAUDE.md "Why V2
    exists"): V1's audit found the mean policy gap was 7.6x-10.2x larger
    than replication-to-replication noise in every profile, which is why
    every one of 100 replications agreed on the winner. This plots the same
    ratio -- |mean delta| / stdev(delta) -- for every V2 grid cell, computed
    the same way, next to that same ratio computed fresh from whichever real
    V1 100-replication runs are passed as --v1-reference (never hardcoded
    from the prose in CLAUDE.md, so this figure can't silently drift out of
    sync with the actual historical data).
    """

    def _ratio(data: ExperimentData) -> float | None:
        deltas = [r["delta"] for r in data.replications]
        if len(deltas) < 2:
            return None
        sd = statistics.stdev(deltas)
        if sd == 0:
            return None
        return abs(statistics.mean(deltas)) / sd

    cells = []
    for topology in TOPOLOGY_ORDER:
        for severity in SEVERITY_ORDER:
            data = grid.get((topology, severity))
            if data is None:
                continue
            ratio = _ratio(data)
            if ratio is not None:
                cells.append(
                    (
                        f"{topology}\n{severity}",
                        ratio,
                        _TOPOLOGY_COLORS[topology],
                        data.summary["experiment_summary"]["replication_count"],
                    )
                )

    v1_ratios = [r for r in (_ratio(d) for d in v1_reference) if r is not None]

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(cells)), 5.5))
    x_positions = list(range(len(cells)))
    ax.bar(
        x_positions,
        [c[1] for c in cells],
        color=[c[2] for c in cells],
        width=0.6,
        zorder=3,
    )
    for x, (label, ratio, _color, n) in zip(x_positions, cells, strict=True):
        ax.text(x, ratio + 0.15, f"n={n}", ha="center", fontsize=7.5, color="#52514e")

    if v1_ratios:
        lo, hi = min(v1_ratios), max(v1_ratios)
        ax.axhspan(lo, hi, color="#8c8c8c", alpha=0.2, zorder=1)
        ax.axhline(
            statistics.mean(v1_ratios),
            color="#5a5a5a",
            linestyle="--",
            linewidth=1.2,
            zorder=2,
        )
        ax.text(
            len(cells) - 0.4,
            hi,
            f"V1 range {lo:.1f}x-{hi:.1f}x (n={len(v1_ratios)} profiles)",
            fontsize=8,
            color="#3a3a3a",
            ha="right",
            va="bottom",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([c[0] for c in cells], fontsize=8)
    ax.set_ylabel("|mean delta| / stdev(delta) across replications")
    ax.set_title("Signal-to-noise ratio by cell, vs. V1's audited baseline")
    fig.tight_layout()

    out_path = output_dir / "14_signal_to_noise_by_cell.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_15_randomness_realization_qa(grid: GridCells, output_dir: Path) -> Path:
    """Not a results plot -- a trust plot. Confirms the new randomness V2
    exists to add (CLAUDE.md V2.3.3/V2.3.5) is actually producing spread,
    not silently degenerating back to V1's fixed values. One row per present
    cell; for the three timing columns, one point is drawn per replication's
    lexicographically-first shock_id (the same "ascending shock_id" tie-break
    convention V2.3.4 already uses), so a multi-shock scenario's distinct
    shock templates are never pooled into one misleading distribution.
    """
    cells = [
        (topology, severity, grid[(topology, severity)])
        for topology in TOPOLOGY_ORDER
        for severity in SEVERITY_ORDER
        if (topology, severity) in grid
    ]
    columns = (
        "Start day",
        "Duration (days)",
        "Information delay (days)",
        "Release quantity",
    )

    fig, axes = plt.subplots(
        len(cells), 4, figsize=(15, 2.1 * len(cells) + 1), squeeze=False, sharex="col"
    )
    for row, (topology, severity, data) in enumerate(cells):
        randomness = load_realized_randomness(data.path)
        series = (
            randomness.start_days,
            randomness.duration_days,
            randomness.information_delay_days,
            randomness.release_quantities,
        )
        for col, (values, title) in enumerate(zip(series, columns, strict=True)):
            ax = axes[row][col]
            if values:
                bins = min(12, max(3, len(set(values))))
                ax.hist(
                    values,
                    bins=bins,
                    color=_TOPOLOGY_COLORS[topology],
                    alpha=0.75,
                    edgecolor="white",
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "no shocks",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
            if row == 0:
                ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{topology}\n{severity}", fontsize=8.5)

    fig.suptitle(
        "Realized randomness QA: what each cell's replications actually drew",
        fontsize=11,
    )
    fig.text(
        0.01,
        0.005,
        "Timing columns show one draw per replication (that replication's lowest-shock_id shock). "
        "Sparse-looking columns reflect small calibration sample sizes, not a bug.",
        fontsize=7.5,
        color="#52514e",
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.97))

    out_path = output_dir / "15_randomness_realization_qa.png"
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
        default=[],
        metavar="PATH",
        help="Path to one experiment's output directory. Repeatable.",
    )
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        default=[],
        metavar="NAME",
        help="Label for the matching --experiment, same order, same count.",
    )
    parser.add_argument(
        "--cell",
        action="append",
        dest="cells",
        default=[],
        nargs=3,
        metavar=("TOPOLOGY", "SEVERITY", "PATH"),
        help="One topology x severity grid cell (CLAUDE.md V2.8.1), e.g. "
        "--cell Compact Light outputs/compact_light_comparison__.... "
        "Repeatable, 0-9 times; the grid need not be complete. Each cell "
        "also gets the normal per-scenario plot set, and >=1 --cell "
        "additionally produces the grid-level plots (08, 11-15) into "
        "<output-dir>/grid/.",
    )
    parser.add_argument(
        "--v1-reference",
        action="append",
        dest="v1_reference",
        default=[],
        metavar="PATH",
        help="Optional real V1 100-replication run directory (e.g. Standard "
        "x Light/Medium/Heavy), used only by plot 14 as the historical "
        "signal-to-noise baseline. Repeatable.",
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
    if not args.experiments and not args.cells:
        parser.error("at least one --experiment or --cell is required")
    for topology, severity, _path in args.cells:
        if topology not in TOPOLOGY_ORDER:
            parser.error(
                f"--cell topology must be one of {TOPOLOGY_ORDER}, got {topology!r}"
            )
        if severity not in SEVERITY_ORDER:
            parser.error(
                f"--cell severity must be one of {SEVERITY_ORDER}, got {severity!r}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [
        load_experiment(Path(path), label)
        for path, label in zip(args.experiments, args.labels, strict=True)
    ]

    grid: GridCells = {}
    for topology, severity, path in args.cells:
        data = load_experiment(Path(path), f"{topology} x {severity}")
        grid[(topology, severity)] = data
        datasets.append(data)

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

    if grid:
        grid_dir = args.output_dir / "grid"
        grid_dir.mkdir(parents=True, exist_ok=True)
        v1_reference = [
            load_experiment(Path(path), f"V1 reference ({Path(path).name})")
            for path in args.v1_reference
        ]
        written.append(plot_08_disruption_robustness_heatmap(grid, grid_dir))
        written.append(plot_11_win_rate_heatmap(grid, grid_dir))
        written.append(plot_12_topology_severity_interaction(grid, grid_dir))
        written.append(plot_13_grid_win_loss_tie(grid, grid_dir))
        written.append(plot_14_signal_to_noise_by_cell(grid, grid_dir, v1_reference))
        written.append(plot_15_randomness_realization_qa(grid, grid_dir))

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
