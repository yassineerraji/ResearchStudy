"""Exposes the research package's config models as JSON Schema and defaults.

Inside `app.services`, this module wraps `supply_chain_simulator`'s existing
Pydantic config classes (`NetworkConfig`, `ScenarioConfig`, ...) so a
frontend can render an editing form from their real, current field shapes
instead of a hand-maintained copy that would drift as the research package
evolves. In the full backend, it also serves the checked-in baseline YAML
files as the form's starting values. It does not validate cross-field rules
(day ordering, cross-references between configs) — `.model_json_schema()`
cannot express those `@model_validator` checks, so real validation stays in
`app.services.config_validate`, which calls the actual loader.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel
from supply_chain_simulator.data_io.loaders import (
    ExperimentConfig,
    HeuristicPolicyConfig,
    LLMPolicyConfig,
    NetworkConfig,
    ScenarioConfig,
)

from app.core.paths import configs_dir


class ConfigType(str, Enum):
    NETWORK = "network"
    SCENARIO = "scenario"
    HEURISTIC_POLICY = "heuristic_policy"
    LLM_POLICY = "llm_policy"
    EXPERIMENT = "experiment"


_MODELS: dict[ConfigType, type[BaseModel]] = {
    ConfigType.NETWORK: NetworkConfig,
    ConfigType.SCENARIO: ScenarioConfig,
    ConfigType.HEURISTIC_POLICY: HeuristicPolicyConfig,
    ConfigType.LLM_POLICY: LLMPolicyConfig,
    ConfigType.EXPERIMENT: ExperimentConfig,
}


def _default_file(config_type: ConfigType) -> Path:
    relative_paths: dict[ConfigType, Path] = {
        ConfigType.NETWORK: Path("networks/baseline_network.yaml"),
        ConfigType.SCENARIO: Path("scenarios/port_closure.yaml"),
        ConfigType.HEURISTIC_POLICY: Path("policies/heuristic.yaml"),
        ConfigType.LLM_POLICY: Path("policies/llm_agent.yaml"),
        ConfigType.EXPERIMENT: Path("experiments/baseline_comparison.yaml"),
    }
    return configs_dir() / relative_paths[config_type]


def get_schema(config_type: ConfigType) -> dict[str, Any]:
    return _MODELS[config_type].model_json_schema()


def get_defaults(config_type: ConfigType) -> dict[str, Any]:
    path = _default_file(config_type)
    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise TypeError(f"expected a mapping at the top level of {path}")
    return content
