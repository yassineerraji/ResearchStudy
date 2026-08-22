"""Request/response shapes for the `/gallery` API.

Inside `app.schemas`, this module defines what the Results Gallery endpoints
return: experiment listings, one experiment's full detail, and a single
day-by-day replay slice. Nested manifest/summary/network/decision content is
typed loosely (`dict[str, Any]`) because it is a passthrough of
`supply_chain_simulator`'s own already-serialized output files — re-typing it
field-for-field here would duplicate a schema that already lives in
`data_io/writers.py` and would drift as the research package evolves. It
performs no file reading itself; that belongs to `app.services.gallery_reader`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ExperimentListItem(BaseModel):
    directory: str
    manifest: dict[str, Any]
    experiment_summary: dict[str, Any] | None = None


class ExperimentListResponse(BaseModel):
    experiments: list[ExperimentListItem]


class ExperimentDetailResponse(BaseModel):
    directory: str
    manifest: dict[str, Any]
    summary: dict[str, Any] | None = None
    replications: list[dict[str, Any]]
    run_metrics: list[dict[str, Any]]
    network: dict[str, Any]
    scenario: dict[str, Any]


class ReplaySliceResponse(BaseModel):
    directory: str
    replication: int
    policy: str
    run_kind: str
    daily_metrics: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    llm_interactions: list[dict[str, Any]] = []


class GridCell(BaseModel):
    topology: str
    severity: str
    directory: str | None = None
    manifest: dict[str, Any] | None = None
    experiment_summary: dict[str, Any] | None = None


class GridResponse(BaseModel):
    topologies: list[str]
    severities: list[str]
    cells: list[GridCell]
