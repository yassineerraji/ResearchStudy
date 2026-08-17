"""Command-line entry point: `validate-config` and `run`.

At the top of the project, this module owns argument parsing and maps
whatever goes wrong during a run to CLAUDE.md section 11.22's exit codes (0
success, 1 unexpected error, 2 configuration error, 3 simulation invariant
error, 4 LLM integration error). In the full system, it is the only place
that turns a validated ResolvedConfig into concrete Policy objects and wires
them into an ExperimentRunner and ExperimentWriter — it implements no
business logic of its own. `_build_comparison_policy` reads the LLM
credentials/model from the environment variables the config names (never
logging their values), builds a live `OpenAIResponsesClient` or a
`ReplayLLMClient` per `execution_mode`, and wraps it in `LLMAgentPolicy` with
its configured fallback. `validate-config` never needs any of this, since it
never instantiates a policy. `main` also loads a repo-root `.env` file (if
present) into the process environment before dispatching either command, so
`OPENAI_API_KEY`/`LLM_MODEL` can live in that gitignored file instead of the
shell — `_load_dotenv` never overrides an already-set variable and never
logs what it reads.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from supply_chain_simulator.data_io.loaders import (
    ConfigurationError,
    LLMPolicyConfig,
    ResolvedConfig,
    resolve_config,
)
from supply_chain_simulator.data_io.writers import ExperimentWriter
from supply_chain_simulator.experiments.metrics import ReplicationComparison
from supply_chain_simulator.experiments.runner import ExperimentRunner
from supply_chain_simulator.integrations.llm_client import (
    LLMClient,
    LLMIntegrationError,
    OpenAIResponsesClient,
    ReplayLLMClient,
)
from supply_chain_simulator.policies.base import Policy
from supply_chain_simulator.policies.fallback import (
    HeuristicFallbackPolicy,
    WaitFallbackPolicy,
)
from supply_chain_simulator.policies.heuristic import HeuristicPolicy
from supply_chain_simulator.policies.llm_agent import (
    LLMAgentPolicy,
    compute_prompt_hash,
)
from supply_chain_simulator.simulation.transition import SimulationInvariantError

logger = logging.getLogger(__name__)

EXIT_SUCCESS = 0
EXIT_UNEXPECTED_ERROR = 1
EXIT_CONFIGURATION_ERROR = 2
EXIT_SIMULATION_INVARIANT_ERROR = 3
EXIT_LLM_INTEGRATION_ERROR = 4


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_dotenv(repo_root: Path) -> None:
    """Loads KEY=VALUE pairs from a .env file at the repo root into the
    process environment, without overriding any variable already set there
    (so a real shell export always wins) and without loading a key whose
    value is blank (so an unfilled-in placeholder line behaves exactly as
    if it were absent, and _required_env_var's existing error still fires).
    Deliberately dependency-free -- no python-dotenv -- per this project's
    minimal-dependencies convention; .env syntax here is limited to simple
    KEY=VALUE lines, blank lines, and '#'-prefixed comments, which is all
    .env.example ever needs. The real .env file is gitignored and this
    function never logs or prints anything it reads from it.
    """
    env_path = repo_root / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


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


def _required_env_var(variable_name: str, purpose: str) -> str:
    value = os.environ.get(variable_name)
    if not value:
        raise LLMIntegrationError(
            f"environment variable {variable_name!r} is not set; required to {purpose}"
        )
    return value


def _resolve_replay_trace_path(
    llm_config: LLMPolicyConfig, resolved_config: ResolvedConfig, repo_root: Path
) -> Path:
    replay_trace_path = llm_config.replay_trace_path
    if not replay_trace_path:
        raise LLMIntegrationError(
            "execution_mode is REPLAY but replay_trace_path is not set "
            "(data_io/loaders.py should already have rejected this configuration)"
        )
    experiment_dir = resolved_config.experiment_config_path.parent
    resolved = (experiment_dir / replay_trace_path).resolve()
    if not resolved.is_relative_to(repo_root):
        raise LLMIntegrationError(
            f"replay_trace_path {replay_trace_path!r} escapes the repository root {repo_root}"
        )
    if not resolved.is_file():
        raise LLMIntegrationError(f"replay trace file does not exist: {resolved}")
    return resolved


def _build_comparison_policy(
    resolved_config: ResolvedConfig, repo_root: Path
) -> tuple[Policy, Policy]:
    llm_config = resolved_config.llm_policy
    model = os.environ.get(llm_config.model_environment_variable) or ""

    client: LLMClient
    if llm_config.execution_mode == "LIVE":
        model = _required_env_var(
            llm_config.model_environment_variable, "build a LIVE LLM policy"
        )
        api_key = _required_env_var(
            llm_config.api_key_environment_variable, "build a LIVE LLM policy"
        )
        client = OpenAIResponsesClient(api_key=api_key)
    else:
        replay_trace_path = _resolve_replay_trace_path(llm_config, resolved_config, repo_root)
        client = ReplayLLMClient(replay_trace_path)

    comparison_policy: Policy = LLMAgentPolicy(
        client=client,
        model=model,
        temperature=llm_config.temperature,
        max_tool_calls=llm_config.max_tool_calls,
        max_output_tokens=llm_config.max_output_tokens,
        request_timeout_seconds=llm_config.request_timeout_seconds,
        max_retries=llm_config.max_retries,
    )

    comparison_fallback_policy: Policy
    if llm_config.fallback_policy == "HEURISTIC":
        comparison_fallback_policy = HeuristicFallbackPolicy(
            HeuristicPolicy(
                expedite_trigger_lateness_days=(
                    resolved_config.heuristic_policy.expedite_trigger_lateness_days
                ),
                cost_tolerance=resolved_config.heuristic_policy.cost_tolerance,
            )
        )
    else:
        comparison_fallback_policy = WaitFallbackPolicy()

    return comparison_policy, comparison_fallback_policy


def _run_experiment(config_path: Path, repo_root: Path) -> None:
    resolved_config = resolve_config(config_path.resolve(), repo_root)
    print(f"Experiment ID: {resolved_config.experiment.experiment_id}")

    heuristic_policy = HeuristicPolicy(
        expedite_trigger_lateness_days=resolved_config.heuristic_policy.expedite_trigger_lateness_days,
        cost_tolerance=resolved_config.heuristic_policy.cost_tolerance,
    )
    comparison_policy, comparison_fallback_policy = _build_comparison_policy(
        resolved_config, repo_root
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
            resolved_config,
            writer,
            llm_prompt_sha256=compute_prompt_hash(),
            on_replication_complete=_print_progress,
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
    _load_dotenv(repo_root)

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
