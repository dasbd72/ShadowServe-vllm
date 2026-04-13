# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
    assert c.enable_shadow_migration is False
    assert c.shadow_additional_blocks_per_request == 0
    assert c.shadow_tkcth_ipc_prefix is None


@pytest.fixture
def shadow_app_enabled():
    app = FastAPI()
    app.state.args = SimpleNamespace(enable_shadow_migration=True)
    attach_router(app)
    return app


def test_shadow_migration_api_not_registered_when_disabled():
    app = FastAPI()
    app.state.args = SimpleNamespace(enable_shadow_migration=False)
    attach_router(app)
    tc = TestClient(app)
    r = tc.post("/shadow_migration/migrate", json={"request_ids": ["r1"]})
    assert r.status_code == HTTPStatus.NOT_FOUND


def test_shadow_migration_api_validation_error_without_kvhtc_ipc_path(
    shadow_app_enabled,
):
    client = MagicMock()
    client.vllm_config = SimpleNamespace(
        shadow_migration_config=ShadowMigrationConfig(enable_shadow_migration=True)
    )
    assert client.vllm_config.shadow_migration_config.enable_shadow_migration is True
    client.run_shadow_migration_to_kvhtc_ipc = AsyncMock()
    shadow_app_enabled.state.engine_client = client

    tc = TestClient(shadow_app_enabled)
    r = tc.post("/shadow_migration/migrate", json={"request_ids": ["r1"]})
    assert r.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    client.run_shadow_migration_to_kvhtc_ipc.assert_not_called()


def test_shadow_migration_api_success_path(shadow_app_enabled):
    client = MagicMock()
    cfg = SimpleNamespace(enable_shadow_migration=True)
    client.vllm_config = SimpleNamespace(shadow_migration_config=cfg)
    client.run_shadow_migration_to_kvhtc_ipc = AsyncMock(return_value=["r1"])
    shadow_app_enabled.state.engine_client = client

    tc = TestClient(shadow_app_enabled)
    r = tc.post(
        "/shadow_migration/migrate",
        json={"request_ids": ["r1"], "kvhtc_ipc_path": "/tmp/x.sock"},
    )
    assert r.status_code == HTTPStatus.OK
    assert r.json() == {"migrated_request_ids": ["r1"]}
    client.run_shadow_migration_to_kvhtc_ipc.assert_called_once_with(
        "/tmp/x.sock", None, ["r1"]
    )


def test_shadow_migration_requests_endpoint_success(shadow_app_enabled):
    client = MagicMock()
    cfg = SimpleNamespace(enable_shadow_migration=True)
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
    client.get_shadow_migration_requests = AsyncMock(return_value=payload)
    shadow_app_enabled.state.engine_client = client

    tc = TestClient(shadow_app_enabled)
    r = tc.get("/shadow_migration/requests")
    assert r.status_code == HTTPStatus.OK
    assert r.json() == payload
    client.get_shadow_migration_requests.assert_called_once_with(
        include_finished=True, finished_limit=1024
    )


def test_shadow_migration_requests_endpoint_forbidden_when_disabled(shadow_app_enabled):
    client = MagicMock()
    cfg = SimpleNamespace(enable_shadow_migration=False)
    client.vllm_config = SimpleNamespace(shadow_migration_config=cfg)
    client.get_shadow_migration_requests = AsyncMock()
    shadow_app_enabled.state.engine_client = client

    tc = TestClient(shadow_app_enabled)
    r = tc.get("/shadow_migration/requests")
    assert r.status_code == HTTPStatus.FORBIDDEN
    client.get_shadow_migration_requests.assert_not_called()
