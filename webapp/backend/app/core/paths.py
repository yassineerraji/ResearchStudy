"""Resolves the repository layout the backend reads from and writes into.

Inside `app.core`, this module is the single place that knows where the
repository root, the research package's `configs/` and `outputs/`
directories, and the backend's own sandbox area live on disk. It mirrors the
`_repo_root()` convention already used by `supply_chain_simulator.cli`
(`Path(__file__).resolve().parents[N]`) so both the CLI a human operator runs
and this backend agree on the same root without either hardcoding the other.
It does not read or validate any config file content itself.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """The Supply-Chain Agent Evaluation repository root.

    Defaults to the directory four levels above this file
    (`webapp/backend/app/core/paths.py` -> repo root), matching where
    `configs/`, `outputs/`, and `src/` live. Overridable via
    `SCAE_REPO_ROOT` for deployments where the backend is installed
    elsewhere relative to the research package checkout.
    """
    override = os.environ.get("SCAE_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[4]


def configs_dir() -> Path:
    return repo_root() / "configs"


def outputs_dir() -> Path:
    return repo_root() / "outputs"


def sandbox_root() -> Path:
    """Root directory for per-run generated configs and results (M3+).

    Kept inside `outputs/` (not a sibling) because `resolve_config`'s
    containment check only requires resolved paths to stay inside
    `repo_root` (see `data_io/loaders.py`'s `_resolve_within_repo`) — placing
    the sandbox anywhere under the repo satisfies that with no core changes.
    """
    return outputs_dir() / "_webapp_runs"
