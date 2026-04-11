# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TKCTH UDS helpers: framed JSON message send/recv over a stream socket.

TKCTH uses the same framing as :func:`vllm.shadow.transfer.uds.send_bytes`:
an 8-byte big-endian length prefix followed by a UTF-8 JSON payload.
"""

from __future__ import annotations

import os
import socket
from contextlib import suppress

from vllm.shadow.transfer import uds
from vllm.shadow.transfer.tkcth_protocol import (
    TkcthMessage,
    tkcth_message_from_bytes,
)

__all__ = (
    "UdsTkcthSenderTransport",
    "UdsTkcthReceiverTransport",
)


class UdsTkcthSenderTransport:
    """Sender-side TKCTH over ``AF_UNIX`` ``SOCK_STREAM`` with framed JSON."""

    __slots__ = ("_conn",)

    def __init__(self) -> None:
        self._conn: socket.socket | None = None

    def connect(self, tkcth_ipc_path: str) -> None:
        self.close()
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.connect(tkcth_ipc_path)
        self._conn = conn

    def send(self, msg: TkcthMessage) -> None:
        uds.send_bytes(self._require_conn(), msg.to_bytes())

    def close(self) -> None:
        if self._conn is not None:
            with suppress(OSError):
                self._conn.close()
            self._conn = None

    def _require_conn(self) -> socket.socket:
        if self._conn is None:
            raise RuntimeError("UdsTkcthSenderTransport is not connected")
        return self._conn


class UdsTkcthReceiverTransport:
    """Receiver-side TKCTH over ``AF_UNIX`` ``SOCK_STREAM`` with framed JSON.

    This transport owns the server socket (from :meth:`prepare` / :meth:`listen`)
    and the accepted connection socket (from :meth:`accept_once` or :meth:`listen`).
    """

    __slots__ = ("_conn", "_server")

    def __init__(self) -> None:
        self._conn: socket.socket | None = None
        self._server: socket.socket | None = None

    def prepare(self, tkcth_ipc_path: str, *, backlog: int = 1) -> None:
        """Bind and listen; call :meth:`accept_once` before :meth:`recv`."""
        if self._server is not None or self._conn is not None:
            self.close()

        with suppress(FileNotFoundError):
            os.unlink(tkcth_ipc_path)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(tkcth_ipc_path)
            server.listen(backlog)
        except BaseException:
            with suppress(OSError):
                server.close()
            raise
        self._server = server

    def accept_once(self) -> None:
        if self._server is None:
            raise RuntimeError(
                "UdsTkcthReceiverTransport.prepare() must be called before "
                "accept_once()"
            )
        if self._conn is not None:
            raise RuntimeError("UdsTkcthReceiverTransport already has a connection")
        conn, _addr = self._server.accept()
        self._conn = conn

    def listen(self, tkcth_ipc_path: str) -> None:
        self.prepare(tkcth_ipc_path)
        self.accept_once()

    def recv(self) -> TkcthMessage:
        return tkcth_message_from_bytes(uds.recv_bytes(self._require_conn()))

    def close(self) -> None:
        if self._conn is not None:
            with suppress(OSError):
                self._conn.close()
            self._conn = None
        if self._server is not None:
            with suppress(OSError):
                self._server.close()
            self._server = None

    def _require_conn(self) -> socket.socket:
        if self._conn is None:
            raise RuntimeError("UdsTkcthReceiverTransport is not attached")
        return self._conn
