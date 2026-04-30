# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Threaded stdlib HTTP server for shadow orchestration ."""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

logger = logging.getLogger("vllm.shadow.http.server")

__all__ = ("KvhtsHttpState", "start_shadow_http_server")


class KvhtsHttpState:
    """Thread-safe readiness for GET /shadow_migration/recv."""

    __slots__ = ("_lock", "_ipc_path", "_listening")

    def __init__(self, ipc_path: str) -> None:
        self._lock = threading.Lock()
        self._ipc_path = ipc_path
        self._listening = False

    def set_listening(self) -> None:
        with self._lock:
            self._listening = True

    def recv_response(self) -> tuple[int, bytes]:
        """Return HTTP status and JSON body for /shadow_migration/recv."""
        with self._lock:
            if not self._listening:
                body = {"detail": "KVHTS listener not ready"}
                return (503, json.dumps(body).encode("utf-8"))
            body = {"kvhts_ipc_path": self._ipc_path}
            return (200, json.dumps(body).encode("utf-8"))


class _ShadowThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        kvhts_state: KvhtsHttpState,
        bind_and_activate: bool = True,
    ) -> None:
        self.kvhts_state = kvhts_state
        super().__init__(
            server_address,
            _ShadowHttpRequestHandler,
            bind_and_activate=bind_and_activate,
        )


class _ShadowHttpRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:  # noqa: N802 — stdlib name
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/health":
            body = json.dumps({"status": "ok"}).encode("utf-8")
            self._send_json(200, body)
            return
        if path == "/shadow_migration/recv":
            srv: _ShadowThreadingHTTPServer = self.server  # type: ignore[assignment]
            status, body = srv.kvhts_state.recv_response()
            self._send_json(status, body)
            return
        self.send_error(404, "Not Found")

    def _send_json(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        self.send_error(405, "Method Not Allowed")


def start_shadow_http_server(
    host: str,
    port: int,
    kvhts_state: KvhtsHttpState,
) -> tuple[_ShadowThreadingHTTPServer, threading.Thread]:
    """Bind and serve in a daemon thread. Caller must call ``shutdown()`` on stop."""
    httpd = _ShadowThreadingHTTPServer((host, port), kvhts_state)
    thread = threading.Thread(
        target=httpd.serve_forever,
        name="vllm-shadow-http",
        daemon=True,
    )
    thread.start()
    logger.info(
        "Shadow HTTP listening on http://%s:%s",
        host,
        httpd.server_port,
    )
    return httpd, thread
