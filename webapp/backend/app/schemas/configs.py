"""Request/response shapes for the `/configs` API.

Inside `app.schemas`, this module defines what a client sends to and
receives from the config schema/defaults/validate endpoints. These are
plain API contracts, not the research package's own config models — they
add API-specific fields (like `note`) that don't belong in
`supply_chain_simulator`'s strict config classes. It performs no validation
beyond basic JSON shape; domain validation happens in
`app.services.config_validate`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ConfigSchemaResponse(BaseModel):
    config_type: str
    json_schema: dict[str, Any]
    note: str


class ConfigDefaultsResponse(BaseModel):
    config_type: str
    content: dict[str, Any]


class ConfigValidateRequest(BaseModel):
    network: dict[str, Any]
    scenario: dict[str, Any]
    heuristic_policy: dict[str, Any]
    llm_policy: dict[str, Any]
    experiment: dict[str, Any]


class ConfigValidateResponse(BaseModel):
    valid: bool
    experiment_id: str


class PresetSummary(BaseModel):
    id: str
    topology: str
    severity: str


class PresetListResponse(BaseModel):
    presets: list[PresetSummary]


class PresetContentResponse(BaseModel):
    network: dict[str, Any]
    scenario: dict[str, Any]
    base_seed: int | None = None
    warmup_days: int | None = None
    horizon_days: int | None = None
    drain_days: int | None = None
    terminal_penalty_days: int | None = None
