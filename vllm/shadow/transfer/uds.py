# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVHTC UDS helpers: pass file descriptors via ``SCM_RIGHTS`` over Unix sockets.

Uses ``sendmsg`` / ``recvmsg`` for ancillary data.

Intended for ``AF_UNIX`` ``SOCK_STREAM`` connections (e.g. hot pod → cold shadow).
One ``sendmsg`` / ``recvmsg`` pair carries one payload buffer plus zero or more
attached FDs atomically (per POSIX semantics for local sockets).

Other KVHTC transports may not use this module. This module intentionally avoids
importing vLLM or heavy dependencies.
"""

from __future__ import annotations

import array
import socket
import struct
from typing import Final

__all__ = (
    "DEFAULT_RECV_BUFSIZE",
    "send_bytes",
    "recv_bytes",
    "send_ack",
    "recv_ack",
    "send_fds",
    "recv_fds",
)

# Maximum bytes read in one ``recvmsg`` for the primary buffer. Payloads larger
# than this require a different framing strategy (not handled here).
DEFAULT_RECV_BUFSIZE: Final[int] = 1024 * 1024

_ACK_PAYLOAD: Final[bytes] = b"\x01"
_BYTES_LEN_STRUCT: Final[struct.Struct] = struct.Struct("!Q")  # uint64 big-endian


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return b""
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed before receiving expected bytes")
        buf.extend(chunk)
    return bytes(buf)


def send_bytes(conn: socket.socket, data: bytes) -> None:
    """Send a length-prefixed bytes payload over a stream socket.

    Frame format: 8-byte big-endian unsigned length prefix, followed by payload.
    Supports empty payloads.
    """
    conn.sendall(_BYTES_LEN_STRUCT.pack(len(data)))
    if len(data) > 0:
        conn.sendall(data)


def recv_bytes(conn: socket.socket) -> bytes:
    """Receive one length-prefixed bytes payload sent by :func:`send_bytes`."""
    header = _recv_exact(conn, _BYTES_LEN_STRUCT.size)
    (n,) = _BYTES_LEN_STRUCT.unpack(header)
    if n > (1 << 31):
        raise ValueError(f"recv_bytes: unreasonable payload length: {n}")
    return _recv_exact(conn, int(n))


def send_fds(conn: socket.socket, fds: list[int], data: bytes) -> None:
    """Send ``data`` on ``conn``, optionally attaching ``fds`` via ``SCM_RIGHTS``.

    If ``fds`` is empty, sends ``data`` with ``sendall`` (no ancillary data).
    Otherwise uses ``sendmsg`` with a single ``SCM_RIGHTS`` control message
    containing all integers in ``fds``.

    Args:
        conn: Connected stream socket (typically ``AF_UNIX``).
        fds: File descriptors to pass; the receiver gets duplicate FDs referring to the
            same open file descriptions.
        data: Payload bytes for the normal message buffer. Must be non-empty.
    """
    if not data:
        raise ValueError("data must be non-empty")
    if not fds:
        conn.sendall(data)
        return
    fds_arr = array.array("i", fds)
    conn.sendmsg([data], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds_arr)])


def recv_fds(conn: socket.socket, maxfds: int) -> tuple[bytes, list[int]]:
    """Receive one message: payload bytes plus any FDs attached with ``SCM_RIGHTS``.

    Uses a single ``recvmsg`` with an ancillary buffer sized for up to ``maxfds`` file
    descriptors. If the sender passed more than ``maxfds`` FDs, raises ``OSError`` with
    truncated control data (``MSG_CTRUNC``).

    Args:
        conn: Connected stream socket (typically ``AF_UNIX``).
        maxfds: Maximum number of file descriptors to accept; must be
            non-negative. Use ``0`` to receive only ``data`` (no FDs); if the
            sender still attaches FDs, the control message is truncated and this
            function raises.

    Returns:
        ``(data, fds)`` where ``fds`` is a list of integer FDs duplicated in this
        process.
    """
    if maxfds < 0:
        raise ValueError("maxfds must be non-negative")
    fdsize = struct.calcsize("i")
    ancbufsize = socket.CMSG_SPACE(fdsize * maxfds) if maxfds > 0 else 0
    data, ancdata, msg_flags, _addr = conn.recvmsg(
        DEFAULT_RECV_BUFSIZE,
        ancbufsize,
    )
    if msg_flags & socket.MSG_TRUNC:
        raise OSError(
            "recv_fds: payload exceeded DEFAULT_RECV_BUFSIZE; "
            "increase DEFAULT_RECV_BUFSIZE or use smaller messages",
        )
    if msg_flags & socket.MSG_CTRUNC:
        raise OSError(
            "recv_fds: ancillary data truncated; increase maxfds or receive fewer FDs",
        )
    fds: list[int] = []
    for level, typ, cdata in ancdata:
        if level == socket.SOL_SOCKET and typ == socket.SCM_RIGHTS:
            arr = array.array("i")
            arr.frombytes(cdata)
            fds.extend(arr.tolist())
    return data, fds


def send_ack(conn: socket.socket) -> None:
    """Send a small fixed payload for handshakes (e.g. per-layer ACK).

    Typically sent after ``recv_fds``.
    """
    conn.sendall(_ACK_PAYLOAD)


def recv_ack(conn: socket.socket) -> None:
    """Receive exactly the payload from :func:`send_ack`.

    Raises on mismatch or EOF.
    """
    expected = _ACK_PAYLOAD
    buf = bytearray()
    while len(buf) < len(expected):
        chunk = conn.recv(len(expected) - len(buf))
        if not chunk:
            raise ConnectionError("connection closed before ACK")
        buf.extend(chunk)
    if bytes(buf) != expected:
        raise ValueError(f"invalid ACK payload: {bytes(buf)!r}")
