"""Writes a candidate config bundle to disk in the shape `resolve_config` expects.

Inside `app.services`, this module owns the one piece of file-writing logic
shared by `config_validate` (a throwaway directory, deleted immediately
after resolving) and `run_launcher` (a persistent per-run sandbox directory,
kept so the CLI subprocess can read it). Both callers need the exact same
five-file layout and the same experiment-document path rewriting, so it
lives in one place rather than two copies that could drift. It does not
validate anything itself — that is `resolve_config`'s job, invoked by the
caller after this function returns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


def write_bundle_files(
    target_dir: Path,
    network: dict[str, Any],
    scenario: dict[str, Any],
    heuristic_policy: dict[str, Any],
    llm_policy: dict[str, Any],
    experiment: dict[str, Any],
    output_root: str = "./results",
) -> Path:
    """Writes the bundle into `target_dir`, returning the experiment.yaml path.

    `experiment`'s path-referencing fields (`network_config`,
    `scenario_config`, `policy_configs`, `output_root`) are overwritten to
    point at the sibling files written here — callers only need to supply
    the experiment's administrative fields (id, horizon, replications, ...).
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    (target_dir / "network.yaml").write_text(yaml.safe_dump(network), encoding="utf-8")
    (target_dir / "scenario.yaml").write_text(yaml.safe_dump(scenario), encoding="utf-8")
    (target_dir / "heuristic.yaml").write_text(
        yaml.safe_dump(heuristic_policy), encoding="utf-8"
    )
    (target_dir / "llm.yaml").write_text(yaml.safe_dump(llm_policy), encoding="utf-8")

    experiment_document: dict[str, Any] = dict(experiment)
    experiment_document.update(
        network_config="network.yaml",
        scenario_config="scenario.yaml",
        policy_configs={"heuristic": "heuristic.yaml", "llm_agent": "llm.yaml"},
        output_root=output_root,
    )
    experiment_path = target_dir / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment_document), encoding="utf-8")
    return experiment_path
