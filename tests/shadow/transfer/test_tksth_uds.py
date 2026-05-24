# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``vllm.shadow.transfer.tksth_uds`` (framed JSON over Unix stream)."""

from __future__ import annotations

import socket
import threading
from collections.abc import Callable

import pytest

from vllm.shadow.transfer.tksth_protocol import TksthTokenDelta
from vllm.shadow.transfer.tksth_uds import (
    UdsTksthReceiverTransport,
    UdsTksthSenderTransport,
)


@pytest.fixture
def tksth_ipc_path(tmp_path) -> str:
    return str(tmp_path / "tksth.sock")


def _threaded_tksth(
    ipc_path: str,
    run_receiver: Callable[[UdsTksthReceiverTransport], None],
    run_sender: Callable[[UdsTksthSenderTransport], None],
) -> None:
    err: list[BaseException] = []
    ready = threading.Event()
    r = UdsTksthReceiverTransport()

    def _recv() -> None:
        try:
            r.prepare(ipc_path)
            ready.set()
            r.accept_once()
            run_receiver(r)
        except BaseException as e:  # pragma: no cover
            err.append(e)
        finally:
            r.close()

    t = threading.Thread(target=_recv)
    t.start()
    if not ready.wait(timeout=5.0):
        pytest.fail("receiver prepare() did not complete in time")
    try:
        s = UdsTksthSenderTransport()
        try:
            s.connect(ipc_path)
            run_sender(s)
        finally:
            s.close()
    finally:
        t.join()
    if err:
        raise err[0]


@pytest.mark.skipif(
    not hasattr(socket, "AF_UNIX"),
    reason="AF_UNIX required",
)
def test_tksth_uds_framing_one_then_sequential(tksth_ipc_path: str) -> None:
    """Single :class:`TksthTokenDelta` round-trip, then more on the same connection."""
    first = TksthTokenDelta(
        migration_id=9,
        request_id="req-1",
        token_ids=[1, 2, 3],
    )
    rest = [
        TksthTokenDelta(migration_id=0, request_id="a", token_ids=[10]),
        TksthTokenDelta(migration_id=0, request_id="a", token_ids=[11]),
        TksthTokenDelta(migration_id=0, request_id="a", token_ids=[12]),
    ]
    got: list[TksthTokenDelta] = []

    def recv(rx: UdsTksthReceiverTransport) -> None:
        a = rx.recv()
        assert isinstance(a, TksthTokenDelta)
        got.append(a)
        for _ in rest:
            b = rx.recv()
            assert isinstance(b, TksthTokenDelta)
            got.append(b)

    def send(sx: UdsTksthSenderTransport) -> None:
        sx.send(first)
        for m in rest:
            sx.send(m)

    _threaded_tksth(tksth_ipc_path, recv, send)

    g0 = got[0]
    assert g0.migration_id == first.migration_id
    assert g0.request_id == first.request_id
    assert g0.token_ids == first.token_ids
    assert [x.token_ids for x in got[1:]] == [[10], [11], [12]]


def test_unconnected_sender_receiver_errors() -> None:
    sender = UdsTksthSenderTransport()
    msg = TksthTokenDelta(migration_id=1, request_id="x", token_ids=[1])
    with pytest.raises(RuntimeError, match="not connected"):
        sender.send(msg)

    receiver = UdsTksthReceiverTransport()
    with pytest.raises(RuntimeError, match="not connected"):
        receiver.recv()

    r = UdsTksthReceiverTransport()
    with pytest.raises(RuntimeError, match="prepare"):
        r.accept_once()
