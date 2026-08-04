"""Routes for browsing completed experiments (the read-only Results Gallery).

Inside `app.routers`, this module exposes `app.services.gallery_reader` over
HTTP: list what's in `outputs/`, fetch one experiment's manifest/summary/
network/replications table, and fetch a single day-by-day replay slice for
the frontend's scrubber. In the full backend, this is the entire surface the
Gallery pages need — no new simulation runs are triggered here. File reads
run in a thread (`asyncio.to_thread`) so they never block the event loop.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query

from app.schemas.gallery import (
    ExperimentDetailResponse,
    ExperimentListItem,
    ExperimentListResponse,
    ReplaySliceResponse,
)
from app.services import gallery_reader

router = APIRouter(prefix="/gallery", tags=["gallery"])


@router.get("", response_model=ExperimentListResponse)
async def list_experiments() -> ExperimentListResponse:
    items = await asyncio.to_thread(gallery_reader.list_experiments)
    return ExperimentListResponse(experiments=[ExperimentListItem(**item) for item in items])


@router.get("/{directory}", response_model=ExperimentDetailResponse)
async def get_experiment(directory: str) -> ExperimentDetailResponse:
    detail = await asyncio.to_thread(gallery_reader.get_experiment_detail, directory)
    return ExperimentDetailResponse(**detail)


@router.get("/{directory}/replay", response_model=ReplaySliceResponse)
async def get_replay_slice(
    directory: str,
    replication: int = Query(..., ge=1),
    policy: str = Query(..., min_length=1),
    run_kind: str = Query(..., min_length=1),
) -> ReplaySliceResponse:
    slice_payload = await asyncio.to_thread(
        gallery_reader.get_replay_slice, directory, replication, policy, run_kind
    )
    return ReplaySliceResponse(**slice_payload)
