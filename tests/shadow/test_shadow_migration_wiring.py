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
        json={
            "migration_id": 7,
            "request_ids": ["r1"],
            "kvhts_ipc_path": "/tmp/x.sock",
        },
    )
    assert r.status_code == HTTPStatus.OK
    assert r.json() == {"migration_id": 7, "migrated_request_ids": ["r1"]}
    client.shadow_migration_migrate.assert_called_once_with("/tmp/x.sock", 7, ["r1"])


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


@pytest.mark.asyncio
async def test_async_mp_client_shadow_migration_async_delegation():
    from vllm.v1.engine.core_client import AsyncMPClient

    mock_util = AsyncMock(
        side_effect=[
            ["r1", "r2"],
            {"active": {"r1": {}}, "finished": []},
            None,
            {7: [1, 2, 3]},
        ]
    )
    client = object.__new__(AsyncMPClient)
    client.call_utility_async = mock_util

    migrated = await client.shadow_migration_migrate_async("/sock", 7, ["r1", "r2"])
    assert migrated == ["r1", "r2"]
    mock_util.assert_any_call("shadow_migration_migrate", "/sock", 7, ["r1", "r2"])

    listed = await client.shadow_migration_get_requests_async(
        include_finished=False, finished_limit=10
    )
    assert listed == {"active": {"r1": {}}, "finished": []}
    mock_util.assert_any_call("shadow_migration_get_requests", False, 10)

    await client.shadow_migration_recv_async(7, "/tmp/kvstc.sock")
    mock_util.assert_any_call("shadow_migration_recv", 7, "/tmp/kvstc.sock")

    completed = await client.shadow_migration_completed_async()
    assert completed == {7: [1, 2, 3]}
    mock_util.assert_any_call("shadow_migration_completed")


def test_inproc_client_shadow_migration_delegation():
    from vllm.v1.engine.core_client import InprocClient

    client = object.__new__(InprocClient)
    client.engine_core = MagicMock()
    client.engine_core.shadow_migration_migrate.return_value = ["r1"]
    client.engine_core.shadow_migration_get_requests.return_value = {"active": {}}
    client.engine_core.shadow_migration_recv.return_value = None
    client.engine_core.shadow_migration_completed.return_value = {1: [2]}

    assert client.shadow_migration_migrate("/sock", 3, ["r1"]) == ["r1"]
    client.engine_core.shadow_migration_migrate.assert_called_once_with(
        "/sock", 3, ["r1"]
    )

    assert client.shadow_migration_get_requests(
        include_finished=False, finished_limit=5
    ) == {"active": {}}
    client.engine_core.shadow_migration_get_requests.assert_called_once_with(False, 5)

    assert client.shadow_migration_recv(9, "/kvstc") is None
    client.engine_core.shadow_migration_recv.assert_called_once_with(9, "/kvstc")

    assert client.shadow_migration_completed() == {1: [2]}
    client.engine_core.shadow_migration_completed.assert_called_once()


def test_sync_mp_client_shadow_migration_delegation():
    from vllm.v1.engine.core_client import SyncMPClient

    client = object.__new__(SyncMPClient)
    client.call_utility = MagicMock(return_value={"active": {"r1": {}}})

    payload = client.shadow_migration_get_requests(
        include_finished=True, finished_limit=1024
    )
    assert payload == {"active": {"r1": {}}}
    client.call_utility.assert_called_once_with(
        "shadow_migration_get_requests", True, 1024
    )
