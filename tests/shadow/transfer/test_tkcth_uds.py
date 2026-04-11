# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``vllm.shadow.transfer.tkcth_uds`` (framed JSON over Unix stream)."""

from __future__ import annotations

import os
import shutil
import socket
import tempfile
import threading
import time
from contextlib import suppress

import pytest

from vllm.shadow.transfer.tkcth_protocol import TkcthTokenDelta
from vllm.shadow.transfer.tkcth_uds import (
    UdsTkcthReceiverTransport,
    UdsTkcthSenderTransport,
)

unix_only = pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"), reason="AF_UNIX required"
)


def _wait_for_unix_bound_path(sock_path: str, *, timeout_sec: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if os.path.exists(sock_path):
            return
        time.sleep(0.001)
    pytest.fail(f"timed out waiting for Unix socket path {sock_path!r}")


def _round_trip_once(msg: TkcthTokenDelta) -> TkcthTokenDelta:
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "tkcth.sock")
    recv_result: list[TkcthTokenDelta] = []
    recv_err: list[BaseException] = []

    def _recv() -> None:
        receiver = UdsTkcthReceiverTransport()
        try:
            receiver.prepare(path)
            receiver.accept_once()
            out = receiver.recv()
            assert isinstance(out, TkcthTokenDelta)
            recv_result.append(out)
        except BaseException as e:  # pragma: no cover
            recv_err.append(e)
        finally:
            receiver.close()

    t = threading.Thread(target=_recv)
    t.start()
    try:
        _wait_for_unix_bound_path(path)
        sender = UdsTkcthSenderTransport()
        try:
            sender.connect(path)
            sender.send(msg)
        finally:
            sender.close()
    finally:
        t.join()

    try:
        if recv_err:
            raise recv_err[0]
        assert len(recv_result) == 1
        return recv_result[0]
    finally:
        with suppress(FileNotFoundError, OSError):
            os.unlink(path)
        shutil.rmtree(tmp, ignore_errors=True)


@unix_only
def test_prepare_accept_connect_round_trip() -> None:
    msg = TkcthTokenDelta(
        migration_id=9,
        request_id="req-1",
        token_ids=[1, 2, 3],
    )
    got = _round_trip_once(msg)
    assert got.migration_id == msg.migration_id
    assert got.request_id == msg.request_id
    assert got.token_ids == msg.token_ids


@unix_only
def test_sequential_messages_same_connection() -> None:
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "tkcth-seq.sock")
    messages = [
        TkcthTokenDelta(migration_id=0, request_id="a", token_ids=[10]),
        TkcthTokenDelta(migration_id=0, request_id="a", token_ids=[11]),
        TkcthTokenDelta(migration_id=0, request_id="a", token_ids=[12]),
    ]
    received: list[object] = []
    recv_err: list[BaseException] = []

    def _recv_all() -> None:
        receiver = UdsTkcthReceiverTransport()
        try:
            receiver.prepare(path)
            receiver.accept_once()
            try:
                for _ in messages:
                    received.append(receiver.recv())
            except BaseException as e:  # pragma: no cover
                recv_err.append(e)
        except BaseException as e:  # pragma: no cover
            recv_err.append(e)
        finally:
            receiver.close()

    t = threading.Thread(target=_recv_all)
    t.start()
    try:
        _wait_for_unix_bound_path(path)
        sender = UdsTkcthSenderTransport()
        try:
            sender.connect(path)
            for m in messages:
                sender.send(m)
        finally:
            sender.close()
    finally:
        t.join()

    try:
        if recv_err:
            raise recv_err[0]
        assert len(received) == 3
        assert [getattr(x, "token_ids", None) for x in received] == [[10], [11], [12]]
    finally:
        with suppress(FileNotFoundError, OSError):
            os.unlink(path)
        shutil.rmtree(tmp, ignore_errors=True)


def test_unconnected_sender_receiver_errors() -> None:
    sender = UdsTkcthSenderTransport()
    msg = TkcthTokenDelta(migration_id=1, request_id="x", token_ids=[1])
    with pytest.raises(RuntimeError, match="not connected"):
        sender.send(msg)

    receiver = UdsTkcthReceiverTransport()
    with pytest.raises(RuntimeError, match="not attached"):
        receiver.recv()

    r = UdsTkcthReceiverTransport()
    with pytest.raises(RuntimeError, match="prepare"):
        r.accept_once()
