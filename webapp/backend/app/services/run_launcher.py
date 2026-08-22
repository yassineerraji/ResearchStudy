"""Orchestrates a sandbox run: validate, sandbox, launch, monitor, finish.

Inside `app.services`, `RunLauncher` is the only place a hosted visitor's
submitted config and API key turn into a running process. It deliberately
never imports `ExperimentRunner`/`SimulationEngine` directly — it shells out
to the same CLI a human operator runs (`research_python_executable()` from
`app.core.paths`), because that stdout/exit-code contract is far more
stable across future changes to the research package's internals than its
Python APIs. The command to run is injectable (`command_builder`) so tests
can substitute a fast fake script instead of spending real OpenAI credits
and waiting minutes per replication. It does not decide *what* a valid
config looks like — `resolve_config` does — and it does not read or
serialize results; `app.services.gallery_reader` does that once a run's
`result_directory` is known.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from supply_chain_simulator.data_io.loaders import ConfigurationError, resolve_config

from app.core.paths import repo_root, research_python_executable, sandbox_root
from app.services import run_monitor
from app.services.config_bundle import write_bundle_files
from app.services.run_registry import (
    RunNotFoundError,
    RunRecord,
    RunRegistry,
    RunStatus,
    get_registry,
)

CommandBuilder = Callable[[Path], list[str]]

# A visitor-supplied model name is never shell-interpreted (it only ever
# becomes one subprocess environment variable's value via
# `asyncio.create_subprocess_exec`'s `env=` mapping, not a shell command
# line), so this is a sanity/typo guard, not an injection defense -- the
# OpenAI API itself is what actually rejects an unknown model name.
_MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._:\-]{1,100}$")


def default_command_builder(experiment_config_path: Path) -> list[str]:
    return [
        research_python_executable(),
        "-m",
        "supply_chain_simulator.cli",
        "run",
        "--config",
        str(experiment_config_path),
    ]


@dataclass
class _ActiveRun:
    task: asyncio.Task[None]
    process: asyncio.subprocess.Process | None = None


def _find_result_directory(output_root: Path) -> Path:
    candidates = [p for p in output_root.iterdir() if p.is_dir()] if output_root.is_dir() else []
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one experiment output directory under {output_root}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


class RunLauncher:
    def __init__(
        self,
        registry: RunRegistry,
        max_concurrent_runs: int,
        run_timeout_seconds: int,
        command_builder: CommandBuilder | None = None,
    ) -> None:
        self._registry = registry
        self._semaphore = asyncio.Semaphore(max_concurrent_runs)
        self._timeout_seconds = run_timeout_seconds
        self._command_builder = command_builder or default_command_builder
        self._active: dict[str, _ActiveRun] = {}

    def _prepare_sandbox(
        self,
        run_id: str,
        network: dict[str, Any],
        scenario: dict[str, Any],
        heuristic_policy: dict[str, Any],
        llm_policy: dict[str, Any],
        experiment: dict[str, Any],
        api_key: str,
        model: str | None,
        max_replications: int,
    ) -> tuple[Path, Path, int, dict[str, str]]:
        """Synchronous, potentially slow work: validate caps, write files,
        resolve the bundle for real. Runs off the event loop via
        `asyncio.to_thread` in `submit`. Raises `ConfigurationError` on any
        rejection — the same exception `/configs/validate` raises, mapped to
        422 by `app.core.errors`.
        """
        replications = experiment.get("replications")
        if not isinstance(replications, int) or replications < 1:
            raise ConfigurationError("experiment.replications must be a positive integer")
        if replications > max_replications:
            raise ConfigurationError(
                f"replications must be at most {max_replications} for a sandbox run "
                f"(requested {replications}) — this keeps a single hosted run's cost "
                "and runtime bounded"
            )
        if llm_policy.get("execution_mode") != "LIVE":
            raise ConfigurationError(
                "sandbox runs require llm_policy.execution_mode to be LIVE "
                "(REPLAY needs a pre-recorded trace this endpoint has no way to supply)"
            )
        api_key_var = llm_policy.get("api_key_environment_variable")
        if not isinstance(api_key_var, str) or not api_key_var:
            raise ConfigurationError("llm_policy.api_key_environment_variable is required")

        # The model name is either visitor-supplied (this request's `model`,
        # set as this run's own `model_var` override below) or, if omitted,
        # must already be configured on this deployment's own environment.
        # Check the latter now rather than after the run is already RUNNING:
        # a deployment missing this would otherwise fail every single
        # submitted run identically, deep inside the subprocess, after
        # already occupying a concurrency slot.
        model_var = llm_policy.get("model_environment_variable")
        if not isinstance(model_var, str) or not model_var:
            raise ConfigurationError("llm_policy.model_environment_variable is required")
        if model is not None:
            if not _MODEL_NAME_PATTERN.match(model):
                raise ConfigurationError(
                    "model must be a non-empty model name (letters, digits, "
                    "'.', '_', '-', ':' only)"
                )
        elif not os.environ.get(model_var):
            raise ConfigurationError(
                f"this server has no {model_var!r} environment variable configured, "
                "and no model was supplied with this request"
            )

        run_dir = sandbox_root() / "runs" / run_id
        output_root = run_dir / "results"
        try:
            experiment_path = write_bundle_files(
                run_dir / "config",
                network,
                scenario,
                heuristic_policy,
                llm_policy,
                experiment,
                output_root=str(output_root),
            )
            try:
                resolve_config(experiment_path, repo_root())
            except ValidationError as exc:
                raise ConfigurationError(f"invalid configuration bundle:\n{exc}") from exc
        except ConfigurationError:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise

        env_overrides = {api_key_var: api_key}
        if model is not None:
            env_overrides[model_var] = model
        return experiment_path, output_root, replications, env_overrides

    async def submit(
        self,
        network: dict[str, Any],
        scenario: dict[str, Any],
        heuristic_policy: dict[str, Any],
        llm_policy: dict[str, Any],
        experiment: dict[str, Any],
        api_key: str,
        model: str | None,
        max_replications: int,
    ) -> RunRecord:
        run_id = uuid.uuid4().hex
        experiment_path, output_root, replications, env_overrides = await asyncio.to_thread(
            self._prepare_sandbox,
            run_id,
            network,
            scenario,
            heuristic_policy,
            llm_policy,
            experiment,
            api_key,
            model,
            max_replications,
        )
        record = await self._registry.create(run_id, total_replications=replications)
        task = asyncio.create_task(
            self._run(run_id, experiment_path, output_root, env_overrides)
        )
        self._active[run_id] = _ActiveRun(task=task)
        return record

    async def cancel(self, run_id: str) -> None:
        active = self._active.get(run_id)
        if active is None:
            raise RunNotFoundError(run_id)
        active.task.cancel()

    async def _run(
        self,
        run_id: str,
        experiment_path: Path,
        output_root: Path,
        env_overrides: dict[str, str],
    ) -> None:
        async with self._semaphore:
            process: asyncio.subprocess.Process | None = None
            try:
                await self._registry.update(run_id, status=RunStatus.RUNNING)
                command = self._command_builder(experiment_path)
                env = {**os.environ, **env_overrides}
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(repo_root()),
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                self._active[run_id].process = process
                stdout_lines: list[str] = []

                async def _drain_and_wait(proc: asyncio.subprocess.Process) -> int:
                    assert proc.stdout is not None
                    async for raw_line in proc.stdout:
                        line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
                        stdout_lines.append(line)
                        progress = run_monitor.parse_replication_progress(line)
                        if progress is not None:
                            await self._registry.update(
                                run_id,
                                completed_replications=progress.completed,
                                total_replications=progress.total,
                            )
                        experiment_id = run_monitor.parse_experiment_id(line)
                        if experiment_id is not None:
                            await self._registry.update(run_id, experiment_id=experiment_id)
                    return await proc.wait()

                try:
                    exit_code = await asyncio.wait_for(
                        _drain_and_wait(process), timeout=self._timeout_seconds
                    )
                except TimeoutError:
                    process.kill()
                    await process.wait()
                    await self._registry.update(
                        run_id,
                        status=RunStatus.FAILED,
                        error=f"run exceeded {self._timeout_seconds}s timeout and was killed",
                    )
                    return

                if exit_code == 0:
                    result_dir = _find_result_directory(output_root)
                    await self._registry.update(
                        run_id, status=RunStatus.COMPLETED, result_directory=result_dir
                    )
                else:
                    tail = "\n".join(stdout_lines[-20:])
                    await self._registry.update(
                        run_id,
                        status=RunStatus.FAILED,
                        error=f"CLI exited with code {exit_code}:\n{tail}",
                    )
            except asyncio.CancelledError:
                if process is not None and process.returncode is None:
                    process.kill()
                    await process.wait()
                await self._registry.update(run_id, status=RunStatus.CANCELLED)
                raise
            except Exception as exc:  # noqa: BLE001
                # Last-resort boundary: an unexpected bug here must become a
                # FAILED run visible over the API, not a silently dropped
                # background-task exception no caller ever sees.
                await self._registry.update(run_id, status=RunStatus.FAILED, error=str(exc))
            finally:
                self._active.pop(run_id, None)


_launcher: RunLauncher | None = None


def get_launcher() -> RunLauncher:
    """Process-wide singleton — see `run_registry.get_registry` for the same
    pattern and why a module-level instance rather than DI is used here.
    """
    global _launcher
    if _launcher is None:
        from app.config import get_settings

        settings = get_settings()
        _launcher = RunLauncher(
            registry=get_registry(),
            max_concurrent_runs=settings.max_concurrent_runs,
            run_timeout_seconds=settings.run_timeout_seconds,
        )
    return _launcher


def set_launcher_for_tests(launcher: RunLauncher) -> None:
    global _launcher
    _launcher = launcher
