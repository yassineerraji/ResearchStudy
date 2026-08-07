"""Routes for submitting and monitoring sandbox runs.

Inside `app.routers`, this module exposes `app.services.run_launcher` and
`app.services.run_registry` over HTTP: submit a config bundle + API key,
poll status/progress, cancel, and — once complete — read the result through
the exact same `app.services.gallery_reader` functions the Results Gallery
uses, just pointed at the run's own output directory instead of a curated
`outputs/` entry. It contains no orchestration or validation logic itself.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query
from fastapi.responses import Response

from app.config import get_settings
from app.schemas.gallery import ExperimentDetailResponse, ReplaySliceResponse
from app.schemas.runs import RunLimitsResponse, RunStatusResponse, RunSubmitRequest
from app.services import gallery_reader
from app.services.run_launcher import get_launcher
from app.services.run_registry import RunNotReadyError, RunStatus, get_registry

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("", response_model=RunStatusResponse, status_code=201)
async def submit_run(request: RunSubmitRequest) -> RunStatusResponse:
    settings = get_settings()
    record = await get_launcher().submit(
        network=request.network,
        scenario=request.scenario,
        heuristic_policy=request.heuristic_policy,
        llm_policy=request.llm_policy,
        experiment=request.experiment,
        api_key=request.api_key,
        max_replications=settings.max_sandbox_replications,
    )
    return RunStatusResponse.from_record(record)


@router.get("", response_model=list[RunStatusResponse])
async def list_runs() -> list[RunStatusResponse]:
    records = await get_registry().list_all()
    return [RunStatusResponse.from_record(record) for record in records]


@router.get("/limits", response_model=RunLimitsResponse)
async def get_run_limits() -> RunLimitsResponse:
    """Ahead of `/{run_id}` on purpose — otherwise `run_id="limits"` would match first."""
    settings = get_settings()
    return RunLimitsResponse(
        max_sandbox_replications=settings.max_sandbox_replications,
        max_concurrent_runs=settings.max_concurrent_runs,
    )


@router.get("/{run_id}", response_model=RunStatusResponse)
async def get_run(run_id: str) -> RunStatusResponse:
    record = await get_registry().get(run_id)
    return RunStatusResponse.from_record(record)


@router.delete("/{run_id}", status_code=204)
async def cancel_run(run_id: str) -> Response:
    await get_launcher().cancel(run_id)
    return Response(status_code=204)


@router.get("/{run_id}/detail", response_model=ExperimentDetailResponse)
async def get_run_detail(run_id: str) -> ExperimentDetailResponse:
    record = await get_registry().get(run_id)
    if record.status != RunStatus.COMPLETED or record.result_directory is None:
        raise RunNotReadyError(f"run {run_id} is {record.status.value}, not completed")
    detail = await asyncio.to_thread(
        gallery_reader.get_experiment_detail_at, record.result_directory
    )
    return ExperimentDetailResponse(**detail)


@router.get("/{run_id}/replay", response_model=ReplaySliceResponse)
async def get_run_replay(
    run_id: str,
    replication: int = Query(..., ge=1),
    policy: str = Query(..., min_length=1),
    run_kind: str = Query(..., min_length=1),
) -> ReplaySliceResponse:
    record = await get_registry().get(run_id)
    if record.status != RunStatus.COMPLETED or record.result_directory is None:
        raise RunNotReadyError(f"run {run_id} is {record.status.value}, not completed")
    slice_payload = await asyncio.to_thread(
        gallery_reader.get_replay_slice_at, record.result_directory, replication, policy, run_kind
    )
    return ReplaySliceResponse(**slice_payload)
