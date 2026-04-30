# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import urllib.error
import urllib.request
from http.client import HTTPResponse

import pytest

from vllm.shadow.http.server import KvhtsHttpState, start_shadow_http_server


def _read_json(resp: HTTPResponse) -> tuple[int, dict]:
    body = resp.read().decode("utf-8")
    data = json.loads(body) if body else {}
    return resp.status, data


def _get_json(url: str, timeout: float = 2.0) -> tuple[int, dict]:
    """GET URL and return (status, body). Treats 4xx/5xx as status, not exception."""
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return _read_json(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        data = json.loads(body) if body else {}
        return e.code, data


def test_shadow_http_health_and_recv_ready():
    state = KvhtsHttpState("/tmp/test-kvhts.sock")
    host = "127.0.0.1"
    httpd, thr = start_shadow_http_server(host, 0, state)
    try:
        port = httpd.server_port
        base = f"http://{host}:{port}"

        with urllib.request.urlopen(f"{base}/health", timeout=2.0) as r:
            status, data = _read_json(r)
        assert status == 200
        assert data == {"status": "ok"}

        status, data = _get_json(f"{base}/shadow_migration/recv")
        assert status == 503
        assert "detail" in data

        state.set_listening()
        status, data = _get_json(f"{base}/shadow_migration/recv")
        assert status == 200
        assert data == {"kvhts_ipc_path": "/tmp/test-kvhts.sock"}
    finally:
        httpd.shutdown()
        thr.join(timeout=5.0)


def test_shadow_http_unknown_path():
    state = KvhtsHttpState("/x")
    httpd, thr = start_shadow_http_server("127.0.0.1", 0, state)
    try:
        port = httpd.server_port
        req = urllib.request.Request(f"http://127.0.0.1:{port}/nope")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=2.0)
        assert exc_info.value.code == 404
    finally:
        httpd.shutdown()
        thr.join(timeout=5.0)


def test_shadow_http_post_not_allowed():
    state = KvhtsHttpState("/x")
    httpd, thr = start_shadow_http_server("127.0.0.1", 0, state)
    try:
        port = httpd.server_port
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/health",
            data=b"{}",
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=2.0)
        assert exc_info.value.code == 405
    finally:
        httpd.shutdown()
        thr.join(timeout=5.0)
