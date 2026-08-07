"""Application-wide settings for the FastAPI backend.

Inside `app`, this module gathers the handful of values that vary by
deployment (allowed CORS origins, filesystem roots) into one `Settings`
object built once at startup. In the full backend, routers and services read
from this object instead of touching `os.environ` directly, keeping
configuration lookups in a single, testable place. It does not hold any
research-domain configuration (network/scenario/policy) — that belongs to
`supply_chain_simulator`'s own config models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from app.core.paths import configs_dir, outputs_dir, repo_root, sandbox_root


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    configs_dir: Path
    outputs_dir: Path
    sandbox_root: Path
    cors_allow_origins: list[str] = field(default_factory=list)

    # M3 sandbox-run guardrails. Deliberately conservative defaults: a
    # hosted sandbox run spends a visitor's own real OpenAI credits and this
    # host's CPU, and nothing upstream of this file enforces any limit on
    # either — see the M3 design notes in the plan for why these three
    # specific values exist rather than being left unbounded.
    max_sandbox_replications: int = 3
    max_concurrent_runs: int = 2
    run_timeout_seconds: int = 900


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    origins = [
        origin.strip()
        for origin in os.environ.get("SCAE_CORS_ORIGINS", "http://localhost:5173").split(",")
        if origin.strip()
    ]
    return Settings(
        repo_root=repo_root(),
        configs_dir=configs_dir(),
        outputs_dir=outputs_dir(),
        sandbox_root=sandbox_root(),
        cors_allow_origins=origins,
        max_sandbox_replications=int(os.environ.get("SCAE_MAX_SANDBOX_REPLICATIONS", "3")),
        max_concurrent_runs=int(os.environ.get("SCAE_MAX_CONCURRENT_RUNS", "2")),
        run_timeout_seconds=int(os.environ.get("SCAE_RUN_TIMEOUT_SECONDS", "900")),
    )
