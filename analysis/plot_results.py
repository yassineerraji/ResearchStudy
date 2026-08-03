"""Turns one or more experiment output directories into PNG plots.

Standalone script, deliberately outside src/supply_chain_simulator: it only
ever reads the CSV/JSON files data_io/writers.py already produces after a
run and never touches simulation behavior, so it has no bearing on
scientific validity or fairness. Given one --experiment/--label pair, it
produces per-experiment plots (TCD comparison, delta distribution, win/loss
breakdown, cost-component breakdown, decision-quality rates). Given two or
more, it also produces one cross-profile plot comparing mean delta across
experiments, so a Light/Medium/Heavy disruption comparison can be seen in
one chart. Requires the optional "analysis" dependency group
(`pip install -e ".[analysis]"`) since it is the only place in this project
that uses matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

_HEURISTIC_COLOR = "#4c72b0"
_LLM_COLOR = "#dd8452"
_COST_COMPONENTS = (
    "transport_cost",
    "reroute_cost",
    "expedite_cost",
    "holding_cost",
    "backlog_cost",
    "late_cost",
    "terminal_cost",
)
_DECISION_RATE_FIELDS = ("invalid_action_rate", "abstention_rate", "fallback_rate")


@dataclass
class ExperimentData:
    label: str
    path: Path
    summary: dict[str, object]
    replications: list[dict[str, object]]


def _slug(label: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in label).strip("_")


def load_experiment(path: Path, label: str) -> ExperimentData:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))

    numeric_fields = (
        "heuristic_undisrupted_cost",
        "heuristic_disrupted_cost",
        "heuristic_tcd",
        "llm_undisrupted_cost",
        "llm_disrupted_cost",
        "llm_tcd",
        "delta",
    )
    with (path / "replications.csv").open(newline="", encoding="utf-8") as handle:
        replications = []
        for row in csv.DictReader(handle):
            parsed: dict[str, object] = dict(row)
            parsed["replication"] = int(row["replication"])
            for field in numeric_fields:
                parsed[field] = float(row[field])
            replications.append(parsed)

    return ExperimentData(label=label, path=path, summary=summary, replications=replications)


def plot_tcd_comparison(data: ExperimentData, output_dir: Path) -> Path:
    """Heuristic vs. LLM disruption cost: mean bars plus every replication's
    individual point, so the spread is visible alongside the average.
    """
    stats = data.summary["experiment_summary"]
    heuristic_tcds = [r["heuristic_tcd"] for r in data.replications]
    llm_tcds = [r["llm_tcd"] for r in data.replications]
    rng = random.Random(0)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(
        ["Heuristic", "LLM agent"],
        [stats["mean_heuristic_tcd"], stats["mean_llm_tcd"]],
        color=[_HEURISTIC_COLOR, _LLM_COLOR],
        alpha=0.7,
    )
    for x_center, values in ((0, heuristic_tcds), (1, llm_tcds)):
        jittered_x = [x_center + (rng.random() - 0.5) * 0.35 for _ in values]
        ax.scatter(jittered_x, values, color="black", alpha=0.4, s=18, zorder=3)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("Total Cost of Disruption (TCD)")
    ax.set_title(f"TCD comparison — {data.label} ({stats['replication_count']} replications)")
    fig.tight_layout()

    out_path = output_dir / f"{_slug(data.label)}_tcd_comparison.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_delta_distribution(data: ExperimentData, output_dir: Path) -> Path:
    """Histogram of (LLM TCD - heuristic TCD) per replication: negative
    means the LLM was cheaper that replication, positive means the
    heuristic was.
    """
    deltas = [r["delta"] for r in data.replications]
    bin_count = min(20, max(5, len(deltas) // 2)) if len(deltas) > 1 else 1

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(deltas, bins=bin_count, color=_LLM_COLOR, alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel("delta = LLM TCD - heuristic TCD  (negative = LLM cheaper)")
    ax.set_ylabel("Replications")
    ax.set_title(f"Delta distribution — {data.label}")
    fig.tight_layout()

    out_path = output_dir / f"{_slug(data.label)}_delta_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_win_breakdown(data: ExperimentData, output_dir: Path) -> Path:
    stats = data.summary["experiment_summary"]
    labels = ["LLM win", "Heuristic win", "Tie"]
    rates = [stats["llm_win_rate"], stats["heuristic_win_rate"], stats["tie_rate"]]
    colors = [_LLM_COLOR, _HEURISTIC_COLOR, "#8c8c8c"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(labels, [rate * 100 for rate in rates], color=colors, alpha=0.8)
    ax.set_ylabel("Share of replications (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Win / loss / tie — {data.label}")
    for index, rate in enumerate(rates):
        ax.text(index, rate * 100 + 1, f"{rate * 100:.1f}%", ha="center")
    fig.tight_layout()

    out_path = output_dir / f"{_slug(data.label)}_win_breakdown.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_cost_breakdown(data: ExperimentData, output_dir: Path) -> Path:
    """Where each policy's disrupted-branch money actually goes."""
    means = data.summary["cost_component_means"]
    heuristic = means["heuristic:DISRUPTED"]
    llm = means["llm_agent:DISRUPTED"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bottoms = [0.0, 0.0]
    for component in _COST_COMPONENTS:
        values = [heuristic[component], llm[component]]
        ax.bar(["Heuristic", "LLM agent"], values, bottom=bottoms, label=component)
        bottoms = [b + v for b, v in zip(bottoms, values, strict=True)]
    ax.set_ylabel("Mean cost per replication (disrupted branch)")
    ax.set_title(f"Cost breakdown — {data.label}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    fig.tight_layout()

    out_path = output_dir / f"{_slug(data.label)}_cost_breakdown.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_decision_quality(data: ExperimentData, output_dir: Path) -> Path:
    means = data.summary["decision_rate_means"]
    heuristic = means["heuristic:DISRUPTED"]
    llm = means["llm_agent:DISRUPTED"]

    x_positions = range(len(_DECISION_RATE_FIELDS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(
        [x - width / 2 for x in x_positions],
        [heuristic[field] * 100 for field in _DECISION_RATE_FIELDS],
        width=width,
        label="Heuristic",
        color=_HEURISTIC_COLOR,
    )
    ax.bar(
        [x + width / 2 for x in x_positions],
        [llm[field] * 100 for field in _DECISION_RATE_FIELDS],
        width=width,
        label="LLM agent",
        color=_LLM_COLOR,
    )
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels([field.replace("_", " ") for field in _DECISION_RATE_FIELDS])
    ax.set_ylabel("Rate (%)")
    ax.set_title(f"Decision quality — {data.label}")
    ax.legend()
    fig.tight_layout()

    out_path = output_dir / f"{_slug(data.label)}_decision_quality.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_cross_profile_delta(datasets: list[ExperimentData], output_dir: Path) -> Path:
    """Mean delta (with 95% CI) side by side across every supplied profile."""
    labels = [data.label for data in datasets]
    means = [data.summary["experiment_summary"]["mean_delta"] for data in datasets]
    lowers = [data.summary["experiment_summary"]["mean_delta_ci_95_lower"] for data in datasets]
    uppers = [data.summary["experiment_summary"]["mean_delta_ci_95_upper"] for data in datasets]
    errors = [
        [mean - lower for mean, lower in zip(means, lowers, strict=True)],
        [upper - mean for mean, upper in zip(means, uppers, strict=True)],
    ]

    fig, ax = plt.subplots(figsize=(6 + len(labels), 5))
    colors = [_LLM_COLOR if mean < 0 else _HEURISTIC_COLOR for mean in means]
    ax.bar(labels, means, yerr=errors, capsize=6, color=colors, alpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("Mean delta (95% CI)  —  negative = LLM cheaper")
    ax.set_title("Mean delta across disruption profiles")
    fig.tight_layout()

    out_path = output_dir / "cross_profile_delta.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot supply-chain-agent-evaluation experiment results."
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
        written.append(plot_tcd_comparison(data, args.output_dir))
        written.append(plot_delta_distribution(data, args.output_dir))
        written.append(plot_win_breakdown(data, args.output_dir))
        written.append(plot_cost_breakdown(data, args.output_dir))
        written.append(plot_decision_quality(data, args.output_dir))

    if len(datasets) >= 2:
        written.append(plot_cross_profile_delta(datasets, args.output_dir))

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
