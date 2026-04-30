# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import secrets
import time
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
    kvhts_ipc_path: str = Field(
        min_length=1,
        description="Unix socket path for this pod's shadow KVHTS listener "
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
    if not cfg.shadow_sender_enabled:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN,
            detail="Shadow migration is disabled (shadow_sender_enabled / "
            "--shadow-sender-enabled).",
        )
    try:
        payload = await client.shadow_migration_get_requests(
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
    """Queue ``request_ids`` and run hot GPU → shadow CPU KV migration over KVHTS."""
    client = engine_client(raw_request)
    cfg = client.vllm_config.shadow_migration_config
    if not cfg.shadow_sender_enabled:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN,
            detail="Shadow migration is disabled (shadow_sender_enabled / "
            "--shadow-sender-enabled).",
        )
    migration_id = (
        body.migration_id if body.migration_id is not None else time.time_ns()
    )
    try:
        migrated = await client.shadow_migration_migrate(
            body.kvhts_ipc_path, migration_id, body.request_ids
        )
    except Exception as e:
        logger.exception("shadow migration failed")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
    return JSONResponse(
        content={
            "migration_id": migration_id,
            "migrated_request_ids": migrated,
        }
    )


@router.get("/shadow_migration/recv")
async def recv_shadow_migration(
    raw_request: Request,
    migration_id: int = Query(description="Migration ID to receive."),
):
    """Create a new KVSTC receiver session and starts accepting."""
    client = engine_client(raw_request)
    cfg = client.vllm_config.shadow_migration_config
    if not cfg.shadow_receiver_enabled:
        raise HTTPException(
            status_code=HTTPStatus.FORBIDDEN,
            detail="Shadow migration is disabled (shadow_receiver_enabled / "
            "--shadow-receiver-enabled).",
        )
    pfx = cfg.shadow_kvstc_ipc_prefix
    if pfx is None or not str(pfx).strip():
        raise RuntimeError(
            "shadow migration requires shadow_kvstc_ipc_prefix "
            "(--shadow-kvstc-ipc-prefix / VLLM_SHADOW_KVSTC_IPC_PREFIX)"
        )
    kvstc_ipc_path = f"{str(pfx).strip()}-{secrets.token_hex(8)}"
    try:
        await client.shadow_migration_recv(migration_id, kvstc_ipc_path)
    except Exception as e:
        logger.exception("shadow migration recv failed")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
    return JSONResponse(
        content={"migration_id": migration_id, "kvstc_ipc_path": kvstc_ipc_path}
    )


@router.get("/shadow_migration/completed")
async def list_completed_shadow_migration(
    raw_request: Request,
):
    """List completed shadow migration sessions."""
    client = engine_client(raw_request)
    try:
        completed = await client.shadow_migration_completed()
    except Exception as e:
        logger.exception("shadow migration completed failed")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
    return JSONResponse(content=completed)


def attach_router(app: FastAPI):
    args = getattr(app.state, "args", None)
    enabled = bool(getattr(args, "shadow_sender_enabled", False)) or bool(
        getattr(args, "shadow_receiver_enabled", False)
    )
    if not enabled:
        return
    logger.warning_once(
        "Shadow migration endpoint is enabled. This should ONLY be used when "
        "serverless shadow migration is configured and trusted."
    )
    app.include_router(router)
