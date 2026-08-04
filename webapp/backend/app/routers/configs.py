"""Routes for research-config schema, defaults, and validation.

Inside `app.routers`, this module exposes `app.services.config_schema` and
`app.services.config_validate` over HTTP. In the full backend, it is what a
config-editing frontend calls to render a form and to check a draft config
before submission. It contains no validation or file-reading logic itself.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.schemas.configs import (
    ConfigDefaultsResponse,
    ConfigSchemaResponse,
    ConfigValidateRequest,
    ConfigValidateResponse,
)
from app.services import config_schema, config_validate
from app.services.config_schema import ConfigType

router = APIRouter(prefix="/configs", tags=["configs"])

_SCHEMA_NOTE = (
    "This JSON Schema describes field shapes and single-field constraints "
    "only. Cross-field rules (day-range ordering, cross-references between "
    "configs, REPLAY mode requiring a replay_trace_path, etc.) are enforced "
    "only by POST /configs/validate, which resolves the config through the "
    "same validator the research package's CLI uses."
)


@router.get("/schema/{config_type}", response_model=ConfigSchemaResponse)
async def get_config_schema(config_type: ConfigType) -> ConfigSchemaResponse:
    return ConfigSchemaResponse(
        config_type=config_type.value,
        json_schema=config_schema.get_schema(config_type),
        note=_SCHEMA_NOTE,
    )


@router.get("/defaults/{config_type}", response_model=ConfigDefaultsResponse)
async def get_config_defaults(config_type: ConfigType) -> ConfigDefaultsResponse:
    return ConfigDefaultsResponse(
        config_type=config_type.value,
        content=config_schema.get_defaults(config_type),
    )


@router.post("/validate", response_model=ConfigValidateResponse)
async def validate_config(request: ConfigValidateRequest) -> ConfigValidateResponse:
    resolved = config_validate.validate_config_bundle(
        network=request.network,
        scenario=request.scenario,
        heuristic_policy=request.heuristic_policy,
        llm_policy=request.llm_policy,
        experiment=request.experiment,
    )
    return ConfigValidateResponse(valid=True, experiment_id=resolved.experiment.experiment_id)
