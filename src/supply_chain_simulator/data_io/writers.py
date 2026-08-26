"""Serializes one run's manifest, resolved config, event tapes, metrics, traces, and summary into its output directory; decides only how results are written, never what happened."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Self, TextIO

import yaml 

from supply_chain_simulator.data_io.loaders import ResolvedConfig
from supply_chain_simulator.decisions.observation import observation_to_canonical_dict
from supply_chain_simulator.domain.actions import DecisionAction, ValidationResult
from supply_chain_simulator.domain.events import EventTape
from supply_chain_simulator.domain.state import DailyMetrics
from supply_chain_simulator.experiments.metrics import ReplicationComparison, RunMetrics
from supply_chain_simulator.simulation.engine import DecisionTraceEntry

_DEPENDENCY_PACKAGES = ("networkx", "pydantic", "PyYAML", "openai")

RUN_METRICS_COLUMNS = (
    "experiment_id",
    "scenario_id",
    "replication",
    "seed",
    "policy",
    "run_kind",
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
    "ending_backlog_units",
    "backlog_unit_days",
    "late_delivered_units",
    "late_delivery_rate",
    "average_lateness_days_weighted",
    "reroute_count",
    "expedite_count",
    "expedited_units",
    "decision_count",
    "invalid_action_rate",
    "abstention_rate",
    "fallback_rate",
    "mean_decision_latency_ms",
    "days_to_clear_backlog_after_shock",
    "terminated_with_unresolved_state",
)

DAILY_METRICS_COLUMNS = (
    "experiment_id",
    "scenario_id",
    "replication",
    "policy",
    "run_kind",
    "day",
    "inventory_units",
    "backlog_units",
    "shipments_at_node",
    "shipments_in_transit",
    "shipments_delivered",
    "daily_demand_units",
    "daily_same_day_fulfilled_units",
    "daily_backlog_fulfilled_units",
    "daily_transport_cost",
    "daily_reroute_cost",
    "daily_expedite_cost",
    "daily_holding_cost",
    "daily_backlog_cost",
    "daily_late_cost",
    "cumulative_total_cost",
    "active_shock_ids",
)

REPLICATIONS_COLUMNS = (
    "replication",
    "seed",
    "heuristic_undisrupted_cost",
    "heuristic_disrupted_cost",
    "heuristic_tcd",
    "llm_undisrupted_cost",
    "llm_disrupted_cost",
    "llm_tcd",
    "delta",
    "winner",
)

_PACKAGE_LOGGER_NAME = "supply_chain_simulator"


class ExperimentWriter:
    """Owns one experiment's output directory and every file inside it.

    Incremental files (event tapes, run/daily metrics, decision traces, LLM
    interactions, replications) are opened once and flushed after each
    replication. Complete files (manifest, resolved config, summary) are
    written once, atomically, via a temp-file-then-rename.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Deliberately does not raise supply_chain_simulator's logger level:
        # doing so would also lower the effective level seen by the console
        # handler cli.py's logging.basicConfig(level=INFO) installs on the
        # root logger, since Python only gates a record once, at the
        # originating logger. Only this handler's own level controls what
        # reaches run.log; today that's whatever already clears the
        # console's INFO gate (cli.py's error-path logging), and it becomes
        # a true DEBUG file the moment cli.py's setup is deliberately
        # changed to allow DEBUG records through in the first place.
        self._package_logger = logging.getLogger(_PACKAGE_LOGGER_NAME)
        self._log_file_handler = logging.FileHandler(
            self.output_dir / "run.log", encoding="utf-8"
        )
        self._log_file_handler.setLevel(logging.DEBUG)
        self._log_file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        self._package_logger.addHandler(self._log_file_handler)

        self._event_tape_file = self._open_jsonl("event_tapes.jsonl")
        self._decision_trace_file = self._open_jsonl("decision_traces.jsonl")
        self._llm_interaction_file = self._open_jsonl("llm_interactions.jsonl")

        self._run_metrics_file = self._open_csv("run_metrics.csv")
        self._run_metrics_writer = csv.writer(self._run_metrics_file)
        self._run_metrics_writer.writerow(RUN_METRICS_COLUMNS)
        self._run_metrics_file.flush()

        self._daily_metrics_file = self._open_csv("daily_metrics.csv")
        self._daily_metrics_writer = csv.writer(self._daily_metrics_file)
        self._daily_metrics_writer.writerow(DAILY_METRICS_COLUMNS)
        self._daily_metrics_file.flush()

        self._replications_file = self._open_csv("replications.csv")
        self._replications_writer = csv.writer(self._replications_file)
        self._replications_writer.writerow(REPLICATIONS_COLUMNS)
        self._replications_file.flush()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._package_logger.removeHandler(self._log_file_handler)
        self._log_file_handler.close()
        for handle in (
            self._event_tape_file,
            self._decision_trace_file,
            self._llm_interaction_file,
            self._run_metrics_file,
            self._daily_metrics_file,
            self._replications_file,
        ):
            handle.close()

    def _open_jsonl(self, name: str) -> TextIO:
        return (self.output_dir / name).open("a", encoding="utf-8")

    def _open_csv(self, name: str) -> TextIO:
        return (self.output_dir / name).open("w", encoding="utf-8", newline="")

    # --- complete files, written once -------------------------------------

    def write_manifest(
        self, resolved_config: ResolvedConfig, llm_prompt_sha256: str | None
    ) -> None:
        manifest: dict[str, object] = {
            "experiment_id": resolved_config.experiment.experiment_id,
            "created_at_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
            "project_version": _project_version(),
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "dependency_versions": _dependency_versions(),
            "network_config_sha256": hash_file(resolved_config.network_config_path),
            "scenario_config_sha256": hash_file(resolved_config.scenario_config_path),
            "heuristic_config_sha256": hash_file(resolved_config.heuristic_config_path),
            "llm_config_sha256": hash_file(resolved_config.llm_config_path),
            "experiment_config_sha256": hash_file(
                resolved_config.experiment_config_path
            ),
            "llm_prompt_sha256": llm_prompt_sha256,
            "llm_execution_mode": resolved_config.llm_policy.execution_mode,
            "llm_model": _env_var(
                resolved_config.llm_policy.model_environment_variable
            ),
            "base_seed": resolved_config.experiment.base_seed,
            "replications": resolved_config.experiment.replications,
        }
        _atomic_write_json(self.output_dir / "manifest.json", manifest)

    def write_resolved_config(self, resolved_config: ResolvedConfig) -> None:
        payload = {
            "experiment": resolved_config.experiment.model_dump(mode="json"),
            "network": resolved_config.network.model_dump(mode="json"),
            "scenario": resolved_config.scenario.model_dump(mode="json"),
            "heuristic_policy": resolved_config.heuristic_policy.model_dump(
                mode="json"
            ),
            "llm_policy": resolved_config.llm_policy.model_dump(mode="json"),
            "resolved_paths": {
                "experiment_config_path": str(resolved_config.experiment_config_path),
                "network_config_path": str(resolved_config.network_config_path),
                "scenario_config_path": str(resolved_config.scenario_config_path),
                "heuristic_config_path": str(resolved_config.heuristic_config_path),
                "llm_config_path": str(resolved_config.llm_config_path),
                "output_root": str(resolved_config.output_root),
            },
        }
        _atomic_write_text(
            self.output_dir / "resolved_config.yaml",
            yaml.safe_dump(payload, sort_keys=True),
        )

    def write_summary(self, payload: dict[str, object]) -> None:
        _atomic_write_json(self.output_dir / "summary.json", payload)

    # --- incremental files, appended per replication ------------------------

    def append_event_tape(self, event_tape: EventTape) -> None:
        payload = _event_tape_to_dict(event_tape)
        payload["event_tape_sha256"] = _compute_canonical_hash(payload)
        self._event_tape_file.write(json.dumps(payload, sort_keys=True) + "\n")
        self._event_tape_file.flush()

    def append_run_metrics(
        self,
        experiment_id: str,
        scenario_id: str,
        replication: int,
        seed: int,
        policy: str,
        run_kind: str,
        metrics: RunMetrics,
    ) -> None:
        row = [
            experiment_id,
            scenario_id,
            replication,
            seed,
            policy,
            run_kind,
            _fmt(metrics.total_cost),
            _fmt(metrics.transport_cost),
            _fmt(metrics.reroute_cost),
            _fmt(metrics.expedite_cost),
            _fmt(metrics.holding_cost),
            _fmt(metrics.backlog_cost),
            _fmt(metrics.late_cost),
            _fmt(metrics.terminal_cost),
            _fmt(metrics.same_day_fill_rate),
            _fmt(metrics.final_fulfilment_rate),
            metrics.ending_backlog_units,
            metrics.backlog_unit_days,
            metrics.late_delivered_units,
            _fmt(metrics.late_delivery_rate),
            _fmt(metrics.average_lateness_days_weighted),
            metrics.reroute_count,
            metrics.expedite_count,
            metrics.expedited_units,
            metrics.decision_count,
            _fmt(metrics.invalid_action_rate),
            _fmt(metrics.abstention_rate),
            _fmt(metrics.fallback_rate),
            _fmt(metrics.mean_decision_latency_ms),
            metrics.days_to_clear_backlog_after_shock,
            metrics.terminated_with_unresolved_state,
        ]
        self._run_metrics_writer.writerow(row)
        self._run_metrics_file.flush()

    def append_daily_metrics(self, daily_metrics: Sequence[DailyMetrics]) -> None:
        for daily in daily_metrics:
            row = [
                daily.experiment_id,
                daily.scenario_id,
                daily.replication,
                daily.policy,
                daily.run_kind,
                daily.day,
                daily.inventory_units,
                daily.backlog_units,
                daily.shipments_at_node,
                daily.shipments_in_transit,
                daily.shipments_delivered,
                daily.daily_demand_units,
                daily.daily_same_day_fulfilled_units,
                daily.daily_backlog_fulfilled_units,
                _fmt(daily.daily_transport_cost),
                _fmt(daily.daily_reroute_cost),
                _fmt(daily.daily_expedite_cost),
                _fmt(daily.daily_holding_cost),
                _fmt(daily.daily_backlog_cost),
                _fmt(daily.daily_late_cost),
                _fmt(daily.cumulative_total_cost),
                ";".join(daily.active_shock_ids),
            ]
            self._daily_metrics_writer.writerow(row)
        self._daily_metrics_file.flush()

    def append_decision_traces(self, entries: Sequence[DecisionTraceEntry]) -> None:
        for entry in entries:
            decision_key = {
                "experiment_id": entry.run_identity.experiment_id,
                "scenario_id": entry.run_identity.scenario_id,
                "replication": entry.run_identity.replication,
                "run_kind": entry.run_identity.run_kind,
                "day": entry.day,
                "shipment_id": entry.shipment_id,
                "observation_hash": entry.observation_hash,
            }
            payload = {
                "decision_key": decision_key,
                "experiment_id": entry.run_identity.experiment_id,
                "scenario_id": entry.run_identity.scenario_id,
                "replication": entry.run_identity.replication,
                "policy": entry.run_identity.policy_name,
                "run_kind": entry.run_identity.run_kind,
                "day": entry.day,
                "shipment_id": entry.shipment_id,
                "observation_hash": entry.observation_hash,
                "observation": observation_to_canonical_dict(entry.observation),
                "proposed_action": _action_to_dict(entry.proposed_action),
                "proposal_validation": _validation_to_dict(entry.proposal_validation),
                "fallback_invoked": entry.fallback_invoked,
                "fallback_action": _action_to_dict(entry.fallback_action),
                "fallback_validation": _validation_to_dict(entry.fallback_validation),
                "executed_action": _action_to_dict(entry.executed_action),
                "decision_latency_ms": round(entry.decision_latency_ms, 6),
            }
            self._decision_trace_file.write(json.dumps(payload, sort_keys=True) + "\n")
        self._decision_trace_file.flush()

    def append_llm_interactions(
        self, interactions: Sequence[dict[str, object]]
    ) -> None:
        """Only ever called with real entries once policies/llm_agent.py exists
        and actually produces LLM interactions; until then the file is
        created and stays empty.
        """
        for interaction in interactions:
            self._llm_interaction_file.write(
                json.dumps(interaction, sort_keys=True) + "\n"
            )
        self._llm_interaction_file.flush()

    def append_replication_comparison(self, comparison: ReplicationComparison) -> None:
        row = [
            comparison.replication,
            comparison.seed,
            _fmt(comparison.heuristic_undisrupted_cost),
            _fmt(comparison.heuristic_disrupted_cost),
            _fmt(comparison.heuristic_tcd),
            _fmt(comparison.llm_undisrupted_cost),
            _fmt(comparison.llm_disrupted_cost),
            _fmt(comparison.llm_tcd),
            _fmt(comparison.delta),
            comparison.winner.value,
        ]
        self._replications_writer.writerow(row)
        self._replications_file.flush()


def _fmt(value: float) -> str:
    """CSV and JSON cost outputs use six decimal places."""
    return f"{value:.6f}"


def _round_floats(value: object) -> object:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def _atomic_write_text(path: Path, content: str) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    _atomic_write_text(
        path, json.dumps(_round_floats(payload), indent=2, sort_keys=True) + "\n"
    )


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compute_canonical_hash(payload: dict[str, object]) -> str:
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _env_var(name: str) -> str | None:
    return os.environ.get(name)


def _project_version() -> str | None:
    try:
        return metadata.version("supply-chain-agent-evaluation")
    except metadata.PackageNotFoundError:
        return None


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in _DEPENDENCY_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _action_to_dict(action: DecisionAction | None) -> dict[str, object] | None:
    if action is None:
        return None
    return {
        "shipment_id": action.shipment_id,
        "action_type": action.action_type.value,
        "route_id": action.route_id,
        "reason_code": action.reason_code.value,
        "rationale": action.rationale,
    }


def _validation_to_dict(result: ValidationResult | None) -> dict[str, object] | None:
    if result is None:
        return None
    return {
        "code": result.code.value,
        "detail": result.detail,
        "is_valid": result.is_valid,
    }


def _event_tape_to_dict(event_tape: EventTape) -> dict[str, object]:
    demand_events = [
        {
            "day": event.day,
            "destination_node_id": event.destination_node_id,
            "product_id": event.product_id,
            "quantity": event.quantity,
        }
        for day in event_tape.days
        for event in day.demand_events
    ]
    shipment_release_events = [
        {
            "day": event.day,
            "shipment_id": event.shipment_id,
            "product_id": event.product_id,
            "quantity": event.quantity,
            "origin_node_id": event.origin_node_id,
            "destination_node_id": event.destination_node_id,
            "due_day": event.due_day,
            "initial_route_edge_ids": list(event.initial_route_edge_ids),
        }
        for day in event_tape.days
        for event in day.shipment_release_events
    ]
    edge_extra_delay_days = {
        str(day.day): dict(day.edge_extra_delay_days) for day in event_tape.days
    }
    shocks = [
        {
            "shock_id": shock.shock_id,
            "shock_type": shock.shock_type.value,
            "target_type": shock.target_type.value,
            "target_id": shock.target_id,
            "physical_start_day": shock.physical_start_day,
            "physical_end_day": shock.physical_end_day,
            "information_day": shock.information_day,
            "capacity_multiplier": shock.capacity_multiplier,
            "lead_time_multiplier": shock.lead_time_multiplier,
            "cost_multiplier": shock.cost_multiplier,
        }
        for shock in event_tape.shocks
    ]
    return {
        "scenario_id": event_tape.scenario_id,
        "replication": event_tape.replication,
        "seed": event_tape.seed,
        "demand_events": demand_events,
        "shipment_release_events": shipment_release_events,
        "edge_extra_delay_days": edge_extra_delay_days,
        "shocks": shocks,
    }
