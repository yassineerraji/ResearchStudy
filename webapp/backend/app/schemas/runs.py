"""Request/response shapes for the `/runs` sandbox-execution API.

Inside `app.schemas`, this module defines what a client submits to start a
sandbox run and what it polls back for status. `detail`/`replay` responses
reuse `app.schemas.gallery`'s `ExperimentDetailResponse`/`ReplaySliceResponse`
unchanged — a completed sandbox run's output directory has the exact same
shape as any gallery experiment, so there is no separate schema to maintain
for those two endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.services.run_registry import RunRecord


class RunSubmitRequest(BaseModel):
    network: dict[str, Any]
    scenario: dict[str, Any]
    heuristic_policy: dict[str, Any]
    llm_policy: dict[str, Any]
    experiment: dict[str, Any]
    api_key: str = Field(min_length=1, description="Never logged, persisted, or echoed back.")


class RunLimitsResponse(BaseModel):
    """Lets the frontend render honest input constraints (e.g. a number
    input's `max`) instead of hardcoding a copy of `Settings` that could
    silently drift from what the server actually enforces.
    """

    max_sandbox_replications: int
    max_concurrent_runs: int


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    experiment_id: str | None
    total_replications: int
    completed_replications: int
    error: str | None
    created_at: str

    @classmethod
    def from_record(cls, record: RunRecord) -> RunStatusResponse:
        return cls(
            run_id=record.run_id,
            status=record.status.value,
            experiment_id=record.experiment_id,
            total_replications=record.total_replications,
            completed_replications=record.completed_replications,
            error=record.error,
            created_at=record.created_at.isoformat(),
        )
