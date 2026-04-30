# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.config import ShadowMigrationConfig
from vllm.entrypoints.serve.shadow_migration.api_router import attach_router


def test_shadow_migration_config_defaults():
    c = ShadowMigrationConfig()
    assert c.shadow_sender_enabled is False
    assert c.shadow_additional_blocks_per_request == 0
    assert c.shadow_tksth_ipc_prefix is None


@pytest.fixture
def shadow_app_enabled():
    app = FastAPI()
    app.state.args = SimpleNamespace(shadow_sender_enabled=True)
    attach_router(app)
    return app


@pytest.fixture
def shadow_app_recv_enabled():
    app = FastAPI()
    app.state.args = SimpleNamespace(
        shadow_receiver_enabled=True, shadow_sender_enabled=False
    )
    attach_router(app)
    return app


def test_shadow_migration_api_not_registered_when_disabled():
    app = FastAPI()
    app.state.args = SimpleNamespace(shadow_sender_enabled=False)
    attach_router(app)
    tc = TestClient(app)
    r = tc.post("/shadow_migration/migrate", json={"request_ids": ["r1"]})
    assert r.status_code == HTTPStatus.NOT_FOUND


def test_shadow_migration_api_validation_error_without_kvhts_ipc_path(
    shadow_app_enabled,
):
    client = MagicMock()
    client.vllm_config = SimpleNamespace(
        shadow_migration_config=ShadowMigrationConfig(shadow_sender_enabled=True)
    )
    assert client.vllm_config.shadow_migration_config.shadow_sender_enabled is True
    client.shadow_migration_migrate = AsyncMock()
    shadow_app_enabled.state.engine_client = client

    tc = TestClient(shadow_app_enabled)
    r = tc.post("/shadow_migration/migrate", json={"request_ids": ["r1"]})
    assert r.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    client.shadow_migration_migrate.assert_not_called()


def test_shadow_migration_api_success_path(shadow_app_enabled):
    client = MagicMock()
    cfg = SimpleNamespace(shadow_sender_enabled=True)
    client.vllm_config = SimpleNamespace(shadow_migration_config=cfg)
    client.shadow_migration_migrate = AsyncMock(return_value=["r1"])
    shadow_app_enabled.state.engine_client = client

    tc = TestClient(shadow_app_enabled)
    r = tc.post(
        "/shadow_migration/migrate",
        json={"request_ids": ["r1"], "kvhts_ipc_path": "/tmp/x.sock"},
    )
    assert r.status_code == HTTPStatus.OK
    assert r.json() == {"migrated_request_ids": ["r1"]}
    client.shadow_migration_migrate.assert_called_once_with(
        "/tmp/x.sock", time.time_ns(), ["r1"]
    )


def test_shadow_migration_requests_endpoint_success(shadow_app_enabled):
    client = MagicMock()
    cfg = SimpleNamespace(shadow_sender_enabled=True)
    client.vllm_config = SimpleNamespace(shadow_migration_config=cfg)
    payload = {
        "active": [
            {
                "request_id": "i1",
                "status": "RUNNING",
                "arrival_time": 1.25,
                "client_index": 0,
                "num_prompt_tokens": 3,
                "num_output_tokens": 4,
                "num_tokens": 7,
                "num_computed_tokens": 6,
                "num_preemptions": 0,
            }
        ],
        "finished": [
            {
                "request_id": "i0",
                "status": "RequestStatus.FINISHED_ABORTED",
                "client_index": 0,
                "arrival_time": 0.25,
                "num_prompt_tokens": 1,
                "num_output_tokens": 2,
                "num_tokens": 3,
                "num_computed_tokens": 3,
                "num_preemptions": 0,
                "stop_reason": None,
            }
        ],
    }
    client.shadow_migration_get_requests = AsyncMock(return_value=payload)
    shadow_app_enabled.state.engine_client = client

    tc = TestClient(shadow_app_enabled)
    r = tc.get("/shadow_migration/requests")
    assert r.status_code == HTTPStatus.OK
    assert r.json() == payload
    client.shadow_migration_get_requests.assert_called_once_with(
        include_finished=True, finished_limit=1024
    )


def test_shadow_migration_requests_endpoint_forbidden_when_disabled(shadow_app_enabled):
    client = MagicMock()
    cfg = SimpleNamespace(shadow_sender_enabled=False)
    client.vllm_config = SimpleNamespace(shadow_migration_config=cfg)
    client.shadow_migration_get_requests = AsyncMock()
    shadow_app_enabled.state.engine_client = client

    tc = TestClient(shadow_app_enabled)
    r = tc.get("/shadow_migration/requests")
    assert r.status_code == HTTPStatus.FORBIDDEN
    client.shadow_migration_get_requests.assert_not_called()


def test_shadow_migration_recv_success_path(shadow_app_recv_enabled):
    client = MagicMock()
    cfg = SimpleNamespace(
        shadow_receiver_enabled=True,
        shadow_kvstc_ipc_prefix="/tmp/vllm-kvstc",
    )
    client.vllm_config = SimpleNamespace(shadow_migration_config=cfg)
    client.shadow_migration_recv = AsyncMock()
    shadow_app_recv_enabled.state.engine_client = client

    tc = TestClient(shadow_app_recv_enabled)
    r = tc.get("/shadow_migration/recv", params={"migration_id": 7})
    assert r.status_code == HTTPStatus.OK
    body = r.json()
    assert body["migration_id"] == 7
    assert "kvstc_ipc_path" in body
    assert body["kvstc_ipc_path"].startswith("/tmp/vllm-kvstc-")
    client.shadow_migration_recv.assert_called_once()
    call_args = client.shadow_migration_recv.call_args[0]
    assert call_args[0] == 7
    assert call_args[1] == body["kvstc_ipc_path"]
