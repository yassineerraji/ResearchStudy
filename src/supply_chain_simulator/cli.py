"""Command-line entry point: `validate-config` and `run`.

At the top of the project, this module owns argument parsing and maps
whatever goes wrong during a run to CLAUDE.md section 11.22's exit codes (0
success, 1 unexpected error, 2 configuration error, 3 simulation invariant
error, 4 LLM integration error). In the full system, it is the only place
that turns a validated ResolvedConfig into concrete Policy objects and wires
them into an ExperimentRunner and ExperimentWriter — it implements no
business logic of its own. `run` currently always raises LLMIntegrationError
before starting: policies/llm_agent.py (Milestone 8) does not exist yet, so
there is no real LLM policy to build from a config's llm_agent section.
`validate-config` is unaffected, since it never needs to instantiate a
policy.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from supply_chain_simulator.data_io.loaders import (
    ConfigurationError,
    ResolvedConfig,
    resolve_config,
)
from supply_chain_simulator.data_io.writers import ExperimentWriter
from supply_chain_simulator.experiments.metrics import ReplicationComparison
from supply_chain_simulator.experiments.runner import ExperimentRunner
from supply_chain_simulator.policies.base import Policy
from supply_chain_simulator.policies.fallback import WaitFallbackPolicy
from supply_chain_simulator.policies.heuristic import HeuristicPolicy
from supply_chain_simulator.simulation.transition import SimulationInvariantError

logger = logging.getLogger(__name__)

EXIT_SUCCESS = 0
EXIT_UNEXPECTED_ERROR = 1
EXIT_CONFIGURATION_ERROR = 2
EXIT_SIMULATION_INVARIANT_ERROR = 3
EXIT_LLM_INTEGRATION_ERROR = 4


class LLMIntegrationError(Exception):
    """Raised when the configured LLM policy cannot be built or reached."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m supply_chain_simulator.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate-config", help="load and validate an experiment configuration"
    )
    validate_parser.add_argument("--config", required=True, type=Path)

    run_parser = subparsers.add_parser("run", help="run the paired experiment")
    run_parser.add_argument("--config", required=True, type=Path)

    return parser


def _validate_config(config_path: Path, repo_root: Path) -> None:
    resolved_config = resolve_config(config_path.resolve(), repo_root)
    print(
        f"Configuration is valid: experiment_id={resolved_config.experiment.experiment_id}"
    )


def _build_comparison_policy(resolved_config: ResolvedConfig) -> tuple[Policy, Policy]:
    raise LLMIntegrationError(
        "the LLM agent policy is not implemented yet (Milestone 8); cannot build "
        f"the configured llm_agent policy from {resolved_config.llm_config_path}. "
        "Construct an ExperimentRunner directly with an injected Policy to run a "
        "heuristic-versus-fake-policy experiment in the meantime."
    )


def _run_experiment(config_path: Path, repo_root: Path) -> None:
    resolved_config = resolve_config(config_path.resolve(), repo_root)
    print(f"Experiment ID: {resolved_config.experiment.experiment_id}")

    heuristic_policy = HeuristicPolicy(
        expedite_trigger_lateness_days=resolved_config.heuristic_policy.expedite_trigger_lateness_days,
        cost_tolerance=resolved_config.heuristic_policy.cost_tolerance,
    )
    comparison_policy, comparison_fallback_policy = _build_comparison_policy(
        resolved_config
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        resolved_config.output_root
        / f"{resolved_config.experiment.experiment_id}__{timestamp}"
    )
    total_replications = resolved_config.experiment.replications

    def _print_progress(replication: int, comparison: ReplicationComparison) -> None:
        print(
            f"Replication {replication}/{total_replications}: "
            f"delta={comparison.delta:.6f} winner={comparison.winner.value}"
        )

    runner = ExperimentRunner(
        heuristic_policy=heuristic_policy,
        heuristic_fallback_policy=WaitFallbackPolicy(),
        comparison_policy=comparison_policy,
        comparison_fallback_policy=comparison_fallback_policy,
    )
    with ExperimentWriter(output_dir) as writer:
        result = runner.run(
            resolved_config, writer, on_replication_complete=_print_progress
        )

    print(f"Output written to: {output_dir}")
    print(
        f"Mean delta: {result.summary.mean_delta:.6f} "
        f"(LLM win rate {result.summary.llm_win_rate:.2%}, "
        f"heuristic win rate {result.summary.heuristic_win_rate:.2%}, "
        f"tie rate {result.summary.tie_rate:.2%})"
    )


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    repo_root = _repo_root()

    try:
        if args.command == "validate-config":
            _validate_config(args.config, repo_root)
        else:
            _run_experiment(args.config, repo_root)
        return EXIT_SUCCESS
    except ConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        return EXIT_CONFIGURATION_ERROR
    except SimulationInvariantError as exc:
        logger.error("Simulation invariant error: %s", exc)
        return EXIT_SIMULATION_INVARIANT_ERROR
    except LLMIntegrationError as exc:
        logger.error("LLM integration error: %s", exc)
        return EXIT_LLM_INTEGRATION_ERROR
    except Exception:
        logger.exception("Unexpected error")
        return EXIT_UNEXPECTED_ERROR


if __name__ == "__main__":
    sys.exit(main())
