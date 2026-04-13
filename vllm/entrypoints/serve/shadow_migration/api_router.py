# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from http import HTTPStatus

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


class MigrateShadowMigrationRequest(BaseModel):
    request_ids: list[str] = Field(min_length=1)
    kvhtc_ipc_path: str = Field(
        min_length=1,
        description="Unix socket path for this pod's shadow KVHTC listener "
        "(must match the path logged by the shadow CPU process).",
    )
    migration_id: int | None = None


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.get("/shadow_migration/requests")
async def list_requests_shadow_migration(
    raw_request: Request,
    include_finished: bool = Query(default=True),
    finished_limit: int = Query(default=1024, ge=0, le=16384),
):
    """List active + recently finished request ids for orchestration.

    Shadow migration operates on scheduler request ids (used by
    ``POST /shadow_migration/migrate``). This endpoint provides a lightweight
    discovery surface for orchestrators to enumerate request ids and basic
    per-request stats.
    """
    client = engine_client(raw_request)
    cfg = client.vllm_config.shadow_migration_config
    if not cfg.enable_shadow_migration:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN,
            detail="Shadow migration is disabled (enable_shadow_migration / "
            "--enable-shadow-migration).",
        )
    try:
        payload = await client.get_shadow_migration_requests(
            include_finished=include_finished, finished_limit=finished_limit
        )
    except Exception as e:
        logger.exception("shadow migration request listing failed")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
    return JSONResponse(content=payload)


@router.post("/shadow_migration/migrate")
async def migrate_shadow_migration(
    body: MigrateShadowMigrationRequest, raw_request: Request
):
    """Queue ``request_ids`` and run hot GPU → shadow CPU KV migration over KVHTC."""
    client = engine_client(raw_request)
    cfg = client.vllm_config.shadow_migration_config
    if not cfg.enable_shadow_migration:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN,
            detail="Shadow migration is disabled (enable_shadow_migration / "
            "--enable-shadow-migration).",
        )
    try:
        migrated = await client.run_shadow_migration_to_kvhtc_ipc(
            body.kvhtc_ipc_path, body.migration_id, body.request_ids
        )
    except Exception as e:
        logger.exception("shadow migration failed")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
    return JSONResponse(content={"migrated_request_ids": migrated})


def attach_router(app: FastAPI):
    args = getattr(app.state, "args", None)
    enabled = bool(getattr(args, "enable_shadow_migration", False))
    if not enabled:
        return
    logger.warning_once(
        "Shadow migration endpoint is enabled. This should ONLY be used when "
        "serverless shadow migration is configured and trusted."
    )
    app.include_router(router)
