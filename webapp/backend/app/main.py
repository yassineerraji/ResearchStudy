"""FastAPI application factory and top-level router registration.

Inside `app`, this module builds the ASGI application: CORS, exception
mapping, and mounting every feature router under `/api/v1`. In the full
backend, it is the only place that assembles the app — routers stay
mountable in isolation for testing, and this file has no business logic of
its own. It does not implement any endpoint handlers directly.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.core.errors import register_exception_handlers


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Supply-Chain Agent Evaluation — Web Backend",
        description=(
            "Read-only results browsing and sandbox-run orchestration in front of "
            "the supply_chain_simulator research package."
        ),
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    from app.routers import configs, gallery, runs

    app.include_router(configs.router, prefix="/api/v1")
    app.include_router(gallery.router, prefix="/api/v1")
    app.include_router(runs.router, prefix="/api/v1")

    return app


app = create_app()
