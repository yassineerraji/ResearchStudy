"""Validates a candidate config bundle through the real research-package loader.

Inside `app.services`, this module is the only place a submitted config gets
checked for correctness, and it checks it the same way the CLI's
`validate-config` command does: by writing it to a throwaway directory (via
`app.services.config_bundle`) and calling
`supply_chain_simulator.data_io.loaders.resolve_config`. In the full
backend, this guarantees the API never accepts something the simulator would
later reject — no validation rule is duplicated here. It does not run any
simulation or persist the bundle; the sandbox directory is deleted before
this function returns. `app.services.run_launcher` reuses the same
`config_bundle.write_bundle_files` helper for runs that must persist.
"""

from __future__ import annotations

import shutil
import uuid
from typing import Any

from pydantic import ValidationError
from supply_chain_simulator.data_io.loaders import (
    ConfigurationError,
    ResolvedConfig,
    resolve_config,
)

from app.core.paths import repo_root, sandbox_root
from app.services.config_bundle import write_bundle_files


def validate_config_bundle(
    network: dict[str, Any],
    scenario: dict[str, Any],
    heuristic_policy: dict[str, Any],
    llm_policy: dict[str, Any],
    experiment: dict[str, Any],
) -> ResolvedConfig:
    """Resolves a full config bundle, raising `ConfigurationError` if invalid."""
    validation_dir = sandbox_root() / "_validate" / uuid.uuid4().hex
    try:
        experiment_path = write_bundle_files(
            validation_dir, network, scenario, heuristic_policy, llm_policy, experiment
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
