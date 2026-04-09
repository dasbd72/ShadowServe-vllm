# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``uds`` — UDS ``SCM_RIGHTS`` FD passing + ACK.

These helpers back Phase 2 hot→cold KV migration: ``send_fds`` per layer-batch
(memfd FD + metadata), ``recv_fds`` on the shadow side, then ``send_ack`` /
``recv_ack`` per the plan's one-ACK-per-layer framing.

Requires a Unix-like socket API with ``SCM_RIGHTS`` (not Windows).
"""

from __future__ import annotations

import os
import socket
from collections.abc import Generator

import pytest

from vllm.shadow.transfer import uds

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HAS_SCM_RIGHTS = hasattr(socket, "SCM_RIGHTS")


def _require_scm_rights() -> None:
    if not _HAS_SCM_RIGHTS:
        pytest.skip("SCM_RIGHTS not available on this platform")


def _socketpair() -> tuple[socket.socket, socket.socket]:
    return socket.socketpair()


@pytest.fixture
def pair() -> Generator[tuple[socket.socket, socket.socket], None, None]:
    a, b = _socketpair()
    try:
        yield a, b
    finally:
        a.close()
        b.close()


# ---------------------------------------------------------------------------
# send_fds / recv_fds
# ---------------------------------------------------------------------------


def test_send_recv_data_only_no_fds(pair: tuple[socket.socket, socket.socket]) -> None:
    a, b = pair
    payload = b"layer-meta-or-inline-bytes"
    uds.send_fds(a, [], payload)
    data, fds = uds.recv_fds(b, 0)
    assert data == payload
    assert fds == []


def test_send_fds_empty_data_raises(pair: tuple[socket.socket, socket.socket]) -> None:
    a, _b = pair
    with pytest.raises(ValueError, match="non-empty"):
        uds.send_fds(a, [], b"")


def test_send_recv_minimal_payload_with_fds(
    pair: tuple[socket.socket, socket.socket],
) -> None:
    """Memfd path still sends at least one payload byte.

    Wire format requires non-empty ``data``.
    """
    _require_scm_rights()
    a, b = pair
    r, w = os.pipe()
    try:
        inline = b"\x00"
        uds.send_fds(a, [w], inline)
        data, fds = uds.recv_fds(b, 1)
        assert data == inline
        assert len(fds) == 1
        os.write(fds[0], b"q")
        assert os.read(r, 1) == b"q"
        os.close(fds[0])
    finally:
        os.close(r)
        os.close(w)


def test_send_recv_single_fd_round_trip(
    pair: tuple[socket.socket, socket.socket],
) -> None:
    _require_scm_rights()
    a, b = pair
    r, w = os.pipe()
    try:
        uds.send_fds(a, [w], b"hello")
        data, fds = uds.recv_fds(b, 1)
        assert data == b"hello"
        assert len(fds) == 1
        os.write(fds[0], b"z")
        assert os.read(r, 1) == b"z"
        os.close(fds[0])
    finally:
        os.close(r)
        os.close(w)


def test_send_recv_multiple_fds_round_trip(
    pair: tuple[socket.socket, socket.socket],
) -> None:
    _require_scm_rights()
    a, b = pair
    r1, w1 = os.pipe()
    r2, w2 = os.pipe()
    try:
        uds.send_fds(a, [w1, w2], b"batch")
        data, fds = uds.recv_fds(b, 2)
        assert data == b"batch"
        assert len(fds) == 2
        os.write(fds[0], b"a")
        os.write(fds[1], b"b")
        assert os.read(r1, 1) == b"a"
        assert os.read(r2, 1) == b"b"
        os.close(fds[0])
        os.close(fds[1])
    finally:
        os.close(r1)
        os.close(w1)
        os.close(r2)
        os.close(w2)


def test_recv_fds_negative_maxfds_raises(
    pair: tuple[socket.socket, socket.socket],
) -> None:
    _, b = pair
    with pytest.raises(ValueError, match="non-negative"):
        uds.recv_fds(b, -1)


def test_recv_fds_maxfds_zero_drops_fds_with_ctrunc(
    pair: tuple[socket.socket, socket.socket],
) -> None:
    """Receiver must size ancillary buffer for FDs.

    ``maxfds=0`` cannot receive ``SCM_RIGHTS``.
    """
    _require_scm_rights()
    a, b = pair
    r, w = os.pipe()
    try:
        uds.send_fds(a, [w], b"oops")
        with pytest.raises(OSError, match="ancillary data truncated"):
            uds.recv_fds(b, 0)
    finally:
        os.close(r)
        os.close(w)


def test_recv_fds_ancillary_buffer_too_small(
    pair: tuple[socket.socket, socket.socket],
) -> None:
    """If the ancillary buffer cannot hold all passed FDs, ``MSG_CTRUNC`` is raised.

    On Linux, ``CMSG_SPACE(sizeof(int))`` and ``CMSG_SPACE(2 * sizeof(int))`` are
    often equal (padding), so two FDs in one ``SCM_RIGHTS`` may still fit when
    ``maxfds == 1``.
    Use three FDs so the control message needs a strictly larger buffer than
    ``CMSG_SPACE(1 * sizeof(int))``.
    """
    _require_scm_rights()
    a, b = pair
    pipes = [os.pipe() for _ in range(3)]
    readers, writers = zip(*pipes, strict=True)
    writers_list = list(writers)
    try:
        uds.send_fds(a, writers_list, b"x")
        with pytest.raises(OSError, match="ancillary data truncated"):
            uds.recv_fds(b, 1)
    finally:
        for fd in readers:
            os.close(fd)
        for fd in writers_list:
            os.close(fd)


# ---------------------------------------------------------------------------
# send_bytes / recv_bytes
# ---------------------------------------------------------------------------


def test_send_recv_bytes_round_trip(pair: tuple[socket.socket, socket.socket]) -> None:
    a, b = pair
    payload = b"hello-bytes"
    uds.send_bytes(a, payload)
    assert uds.recv_bytes(b) == payload


def test_send_recv_bytes_empty_ok(pair: tuple[socket.socket, socket.socket]) -> None:
    a, b = pair
    uds.send_bytes(a, b"")
    assert uds.recv_bytes(b) == b""


def test_send_recv_bytes_multiple_frames(
    pair: tuple[socket.socket, socket.socket],
) -> None:
    a, b = pair
    frames = [b"a", b"", b"0123456789", b"x" * 4096]
    for f in frames:
        uds.send_bytes(a, f)
    for f in frames:
        assert uds.recv_bytes(b) == f


def test_recv_bytes_eof_raises(pair: tuple[socket.socket, socket.socket]) -> None:
    a, b = pair
    a.close()
    with pytest.raises(ConnectionError, match="closed"):
        uds.recv_bytes(b)


# ---------------------------------------------------------------------------
# send_ack / recv_ack
# ---------------------------------------------------------------------------


def test_ack_round_trip(pair: tuple[socket.socket, socket.socket]) -> None:
    a, b = pair
    uds.send_ack(a)
    uds.recv_ack(b)


def test_recv_ack_eof_raises(pair: tuple[socket.socket, socket.socket]) -> None:
    a, b = pair
    a.close()
    with pytest.raises(ConnectionError, match="closed before ACK"):
        uds.recv_ack(b)


def test_recv_ack_wrong_payload_raises(
    pair: tuple[socket.socket, socket.socket],
) -> None:
    a, b = pair
    a.sendall(b"\x02")
    with pytest.raises(ValueError, match="invalid ACK"):
        uds.recv_ack(b)


# ---------------------------------------------------------------------------
# Plan-shaped sequence: recv_fds then ACK back (one layer-batch step)
# ---------------------------------------------------------------------------


def test_layer_batch_send_recv_then_ack(
    pair: tuple[socket.socket, socket.socket],
) -> None:
    """Mirror one step: hot sends memfd + envelope bytes; cold ACKs after ingest."""
    a, b = pair
    r, w = os.pipe()
    try:
        envelope_and_json = b'{"batch_size":1}'
        uds.send_fds(a, [w], envelope_and_json)
        data, fds = uds.recv_fds(b, 1)
        assert data == envelope_and_json
        assert len(fds) == 1
        uds.send_ack(b)
        uds.recv_ack(a)
        os.close(fds[0])
    finally:
        os.close(r)
        os.close(w)
