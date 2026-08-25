"""Named topology x severity presets for the sandbox 'Run Your Own' builder.

Inside `app.services`, this module lets a visitor pick one of the nine real
(topology, severity) grid cells and load the exact `configs/` files already
validated and run for that cell -- a preset picker
only, per Yassine's explicit scope decision, never a free-form shock editor.
It reuses `app.services.config_schema`'s plain YAML-reading approach rather
than inventing a second one. It shares `app.services.grid`'s topology/
severity vocabulary but reads *input* config files, never completed *output*
directories -- the two modules stay independent.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import yaml  # type: ignore[import-untyped]

from app.core.paths import configs_dir


class PresetNotFoundError(Exception):
    """Raised when a preset id isn't one of the nine known grid cells."""


class _PresetFiles(NamedTuple):
    network: str
    scenario: str
    experiment: str


# Exactly the nine files this project's config layout names for the real
# grid -- each `experiment` file is only read for its administrative fields
# (base_seed, warmup/horizon/drain/terminal_penalty days), never for its
# `replications` (the sandbox's own cap governs that, see run_launcher.py).
_PRESETS: dict[str, _PresetFiles] = {
    "compact_light": _PresetFiles(
        "networks/topology_compact.yaml",
        "scenarios/port_partial_capacity.yaml",
        "experiments/compact_light_comparison.yaml",
    ),
    "compact_medium": _PresetFiles(
        "networks/topology_compact.yaml",
        "scenarios/port_closure.yaml",
        "experiments/compact_medium_comparison.yaml",
    ),
    "compact_heavy": _PresetFiles(
        "networks/topology_compact.yaml",
        "scenarios/port_extended_closure_compact.yaml",
        "experiments/compact_heavy_comparison.yaml",
    ),
    "standard_light": _PresetFiles(
        "networks/baseline_network.yaml",
        "scenarios/port_partial_capacity.yaml",
        "experiments/light_disruption_comparison.yaml",
    ),
    "standard_medium": _PresetFiles(
        "networks/baseline_network.yaml",
        "scenarios/port_closure.yaml",
        "experiments/baseline_comparison.yaml",
    ),
    "standard_heavy": _PresetFiles(
        "networks/baseline_network.yaml",
        "scenarios/port_extended_closure.yaml",
        "experiments/heavy_disruption_comparison.yaml",
    ),
    "extended_light": _PresetFiles(
        "networks/topology_extended.yaml",
        "scenarios/hub_partial_capacity_extended.yaml",
        "experiments/extended_light_comparison.yaml",
    ),
    "extended_medium": _PresetFiles(
        "networks/topology_extended.yaml",
        "scenarios/hub_closure_extended.yaml",
        "experiments/extended_medium_comparison.yaml",
    ),
    "extended_heavy": _PresetFiles(
        "networks/topology_extended.yaml",
        "scenarios/hub_extended_closure_with_congestion.yaml",
        "experiments/extended_heavy_comparison.yaml",
    ),
}

_TOPOLOGY_LABELS = {"compact": "Compact", "standard": "Standard", "extended": "Extended"}
_SEVERITY_LABELS = {"light": "Light", "medium": "Medium", "heavy": "Heavy"}


def list_presets() -> list[dict[str, str]]:
    presets = []
    for preset_id in _PRESETS:
        topology_key, severity_key = preset_id.split("_", 1)
        presets.append(
            {
                "id": preset_id,
                "topology": _TOPOLOGY_LABELS[topology_key],
                "severity": _SEVERITY_LABELS[severity_key],
            }
        )
    return presets


def _load_yaml(relative_path: str) -> dict[str, Any]:
    path = configs_dir() / relative_path
    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    if not isinstance(content, dict):
        raise TypeError(f"expected a mapping at the top level of {path}")
    return content


def get_preset_content(preset_id: str) -> dict[str, Any]:
    files = _PRESETS.get(preset_id)
    if files is None:
        raise PresetNotFoundError(preset_id)
    experiment = _load_yaml(files.experiment)
    return {
        "network": _load_yaml(files.network),
        "scenario": _load_yaml(files.scenario),
        "base_seed": experiment.get("base_seed"),
        "warmup_days": experiment.get("warmup_days"),
        "horizon_days": experiment.get("horizon_days"),
        "drain_days": experiment.get("drain_days"),
        "terminal_penalty_days": experiment.get("terminal_penalty_days"),
    }
