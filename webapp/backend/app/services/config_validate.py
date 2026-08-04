"""Validates a candidate config bundle through the real research-package loader.

Inside `app.services`, this module is the only place a submitted config gets
checked for correctness, and it checks it the same way the CLI's
`validate-config` command does: by writing it to a throwaway directory and
calling `supply_chain_simulator.data_io.loaders.resolve_config`. In the full
backend, this guarantees the API never accepts something the simulator would
later reject — no validation rule is duplicated here. It does not run any
simulation or persist the bundle; the sandbox directory is deleted before
this function returns.
"""

from __future__ import annotations

import shutil
import uuid
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError
from supply_chain_simulator.data_io.loaders import (
    ConfigurationError,
    ResolvedConfig,
    resolve_config,
)

from app.core.paths import repo_root, sandbox_root


def validate_config_bundle(
    network: dict[str, Any],
    scenario: dict[str, Any],
    heuristic_policy: dict[str, Any],
    llm_policy: dict[str, Any],
    experiment: dict[str, Any],
) -> ResolvedConfig:
    """Resolves a full config bundle, raising `ConfigurationError` if invalid.

    `experiment`'s path-referencing fields (`network_config`,
    `scenario_config`, `policy_configs`, `output_root`) are overwritten to
    point at the sandboxed files written here — callers only need to supply
    the experiment's administrative fields (id, horizon, replications, ...).
    """
    validation_dir = sandbox_root() / "_validate" / uuid.uuid4().hex
    validation_dir.mkdir(parents=True, exist_ok=True)
    try:
        (validation_dir / "network.yaml").write_text(
            yaml.safe_dump(network), encoding="utf-8"
        )
        (validation_dir / "scenario.yaml").write_text(
            yaml.safe_dump(scenario), encoding="utf-8"
        )
        (validation_dir / "heuristic.yaml").write_text(
            yaml.safe_dump(heuristic_policy), encoding="utf-8"
        )
        (validation_dir / "llm.yaml").write_text(
            yaml.safe_dump(llm_policy), encoding="utf-8"
        )

        experiment_document: dict[str, Any] = dict(experiment)
        experiment_document.update(
            network_config="network.yaml",
            scenario_config="scenario.yaml",
            policy_configs={"heuristic": "heuristic.yaml", "llm_agent": "llm.yaml"},
            output_root="./results",
        )
        experiment_path = validation_dir / "experiment.yaml"
        experiment_path.write_text(
            yaml.safe_dump(experiment_document), encoding="utf-8"
        )

        try:
            return resolve_config(experiment_path, repo_root())
        except ValidationError as exc:
            # resolve_config's per-file loads already wrap ValidationError as
            # ConfigurationError, but ResolvedConfig's own cross-config
            # @model_validator (shocks vs. network/warmup) runs after all
            # files are loaded and raises a raw pydantic ValidationError —
            # normalize it the same way so every validation failure the API
            # sees is a ConfigurationError (app.core.errors maps that to 422).
            raise ConfigurationError(f"invalid configuration bundle:\n{exc}") from exc
    finally:
        shutil.rmtree(validation_dir, ignore_errors=True)
