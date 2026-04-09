# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``vllm.shadow.transfer.uds`` (length-prefixed I/O, SCM_RIGHTS)."""

from __future__ import annotations

import os
import shutil
import socket
import tempfile
import threading
import time
from contextlib import suppress

import pytest

from vllm.shadow.transfer.uds import (
    _BYTES_LEN_STRUCT,
    DEFAULT_RECV_BUFSIZE,
    UdsTransport,
)

unix_only = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"),
    reason="AF_UNIX required",
)


def _pair() -> tuple[socket.socket, socket.socket]:
    a, b = socket.socketpair()
    return a, b


# ---------------------------------------------------------------------------
# In-memory connected sockets (no filesystem AF_UNIX)
# ---------------------------------------------------------------------------


def test_default_recv_bufsize() -> None:
    assert DEFAULT_RECV_BUFSIZE == 1024 * 1024


def test_recv_exact_zero_returns_empty() -> None:
    a, b = _pair()
    try:
        assert UdsTransport._recv_exact(a, 0) == b""
    finally:
        a.close()
        b.close()


def test_recv_exact_negative_raises() -> None:
    a, b = _pair()
    try:
        with pytest.raises(ValueError, match="non-negative"):
            UdsTransport._recv_exact(a, -1)
    finally:
        a.close()
        b.close()


def test_recv_exact_connection_closed() -> None:
    a, b = _pair()
    b.close()
    try:
        with pytest.raises(ConnectionError, match="connection closed"):
            UdsTransport._recv_exact(a, 1)
    finally:
        a.close()


def test_require_conn_raises_when_not_attached() -> None:
    t = UdsTransport()
    with pytest.raises(RuntimeError, match="is not connected"):
        t._send_bytes(b"hi")


def test_send_recv_bytes_round_trip_with_attach() -> None:
    a, b = _pair()
    t_send = UdsTransport()
    t_recv = UdsTransport()
    try:
        t_send.attach(a)
        t_recv.attach(b)
        payload = b"hello" * 200
        t_send._send_bytes(payload)
        assert t_recv._recv_bytes() == payload
    finally:
        t_send.close()
        t_recv.close()


def test_send_recv_bytes_empty_payload() -> None:
    a, b = _pair()
    t_send = UdsTransport()
    t_recv = UdsTransport()
    try:
        t_send.attach(a)
        t_recv.attach(b)
        t_send._send_bytes(b"")
        assert t_recv._recv_bytes() == b""
    finally:
        t_send.close()
        t_recv.close()


def test_recv_bytes_rejects_huge_length_header() -> None:
    a, b = _pair()
    t_recv = UdsTransport()
    try:
        t_recv.attach(b)
        bad = _BYTES_LEN_STRUCT.pack((1 << 31) + 1)
        a.sendall(bad)
        with pytest.raises(ValueError, match="unreasonable payload length"):
            t_recv._recv_bytes()
    finally:
        t_recv.close()
        a.close()


def test_send_fds_rejects_empty_data() -> None:
    a, b = _pair()
    t = UdsTransport()
    try:
        t.attach(a)
        with pytest.raises(ValueError, match="non-empty"):
            t._send_fds([], b"")
    finally:
        t.close()
        b.close()


def test_send_fds_no_fds_sendall_recv() -> None:
    a, b = _pair()
    t_send = UdsTransport()
    t_recv = UdsTransport()
    try:
        t_send.attach(a)
        t_recv.attach(b)
        t_send._send_fds([], b"plain")
        data, fds = t_recv._recv_fds(0)
        assert data == b"plain"
        assert fds == []
    finally:
        t_send.close()
        t_recv.close()


def test_send_fds_with_pipe_read_fd_round_trip() -> None:
    a, b = _pair()
    t_send = UdsTransport()
    t_recv = UdsTransport()
    r, w = os.pipe()
    try:
        t_send.attach(a)
        t_recv.attach(b)
        t_send._send_fds([r], b"pl")
        os.close(r)
        data, fds = t_recv._recv_fds(4)
        assert data == b"pl"
        assert len(fds) == 1
        new_r = fds[0]
        try:
            os.write(w, b"Z")
            assert os.read(new_r, 1) == b"Z"
        finally:
            os.close(new_r)
    finally:
        t_send.close()
        t_recv.close()
        with suppress(OSError):
            os.close(w)


def test_recv_fds_rejects_negative_maxfds() -> None:
    a, b = _pair()
    t = UdsTransport()
    try:
        t.attach(a)
        with pytest.raises(ValueError, match="non-negative"):
            t._recv_fds(-1)
    finally:
        t.close()
        b.close()


def test_send_ack_recv_ack() -> None:
    a, b = _pair()
    t1 = UdsTransport()
    t2 = UdsTransport()
    try:
        t1.attach(a)
        t2.attach(b)
        t1._send_ack()
        t2._recv_ack()
    finally:
        t1.close()
        t2.close()


def test_recv_ack_invalid_payload() -> None:
    a, b = _pair()
    t = UdsTransport()
    try:
        t.attach(b)
        a.sendall(b"\x00")
        with pytest.raises(ValueError, match="invalid ACK"):
            t._recv_ack()
    finally:
        t.close()
        a.close()


def test_close_is_idempotent() -> None:
    t = UdsTransport()
    t.close()
    t.close()


# ---------------------------------------------------------------------------
# AF_UNIX listen / connect / accept
# ---------------------------------------------------------------------------


def _wait_for_unix_bound_path(sock_path: str, *, timeout_sec: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if os.path.exists(sock_path):
            return
        time.sleep(0.001)
    pytest.fail(f"timed out waiting for Unix socket path {sock_path!r}")


@unix_only
def test_prepare_accept_connect_round_trip_bytes() -> None:
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "uds.sock")
    out: list[bytes] = []
    err: list[BaseException] = []

    def _serve() -> None:
        srv = UdsTransport()
        try:
            srv.prepare(path)
            srv.accept_once()
            out.append(srv._recv_bytes())
        except BaseException as e:  # pragma: no cover
            err.append(e)
        finally:
            srv.close()

    th = threading.Thread(target=_serve)
    th.start()
    try:
        _wait_for_unix_bound_path(path)
        cli = UdsTransport()
        try:
            cli.connect(path)
            cli._send_bytes(b"round-trip")
        finally:
            cli.close()
    finally:
        th.join()
        with suppress(FileNotFoundError, OSError):
            os.unlink(path)
        shutil.rmtree(tmp, ignore_errors=True)

    if err:
        raise err[0]
    assert out == [b"round-trip"]


@unix_only
def test_accept_once_without_prepare_raises() -> None:
    t = UdsTransport()
    with pytest.raises(RuntimeError, match="prepare()"):
        t.accept_once()
    t.close()


@unix_only
def test_second_accept_once_raises() -> None:
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "uds2.sock")
    ev = threading.Event()
    client_err: list[BaseException] = []

    def _client() -> None:
        c = UdsTransport()
        try:
            c.connect(path)
            ev.wait(timeout=5.0)
        except BaseException as e:  # pragma: no cover
            client_err.append(e)
        finally:
            c.close()

    t = UdsTransport()
    try:
        t.prepare(path)
        _wait_for_unix_bound_path(path)
        th = threading.Thread(target=_client)
        th.start()
        t.accept_once()
        with pytest.raises(RuntimeError, match="already has a connection"):
            t.accept_once()
        ev.set()
        th.join()
    finally:
        t.close()
        with suppress(FileNotFoundError, OSError):
            os.unlink(path)
        shutil.rmtree(tmp, ignore_errors=True)

    if client_err:
        raise client_err[0]
