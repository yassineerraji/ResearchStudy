"""Maps core-package exceptions to HTTP responses.

Inside `app.core`, this module is the only place that translates a
`supply_chain_simulator` exception into an HTTP status code and JSON body.
In the full backend, routers and services let these exceptions propagate
rather than catching and re-wrapping them individually, so this mapping stays
in one place as new exception types are wired in across later milestones. It
does not change or suppress the underlying error — only how it's reported.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from supply_chain_simulator.data_io.loaders import ConfigurationError

from app.services.gallery_reader import ExperimentNotFoundError
from app.services.run_registry import RunNotFoundError, RunNotReadyError


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ConfigurationError)
    async def _handle_configuration_error(
        request: Request, exc: ConfigurationError
    ) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(ExperimentNotFoundError)
    async def _handle_experiment_not_found(
        request: Request, exc: ExperimentNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404, content={"detail": f"unknown experiment directory: {exc}"}
        )

    @app.exception_handler(RunNotFoundError)
    async def _handle_run_not_found(request: Request, exc: RunNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": f"unknown run: {exc}"})

    @app.exception_handler(RunNotReadyError)
    async def _handle_run_not_ready(request: Request, exc: RunNotReadyError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})
