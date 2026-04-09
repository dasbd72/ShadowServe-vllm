# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UDS helpers: pass file descriptors via ``SCM_RIGHTS`` over Unix sockets.

Uses ``sendmsg`` / ``recvmsg`` for ancillary data.

Intended for ``AF_UNIX`` ``SOCK_STREAM`` connections (e.g. hot pod → shadow).
One ``sendmsg`` / ``recvmsg`` pair carries one payload buffer plus zero or more
attached FDs atomically (per POSIX semantics for local sockets).

Framing and FD passing live on :class:`UdsTransport` so transports call
``self._send_bytes`` / ``self._recv_fds`` / etc. after ``connect``,
``prepare``+``accept_once``, or :meth:`UdsTransport.attach`.

This module intentionally avoids importing vLLM or heavy dependencies.
"""

from __future__ import annotations

import array
import os
import socket
import struct
from contextlib import suppress
from typing import Final

__all__ = (
    "DEFAULT_RECV_BUFSIZE",
    "UdsTransport",
)

# Maximum bytes read in one ``recvmsg`` for the primary buffer. Payloads larger
# than this require a different framing strategy (not handled here).
DEFAULT_RECV_BUFSIZE: Final[int] = 1024 * 1024

_ACK_PAYLOAD: Final[bytes] = b"\x01"
_BYTES_LEN_STRUCT: Final[struct.Struct] = struct.Struct("!Q")  # uint64 big-endian


class UdsTransport:
    """AF_UNIX stream transport: connect/listen, length-prefixed I/O, FDs, ACK.

    **Server path:** :meth:`prepare` (bind and listen) + :meth:`accept_once`.

    **Client path:** :meth:`connect` or :meth:`attach` with an existing socket.

    **Common:** framed :meth:`_send_bytes` / :meth:`_recv_bytes`, :meth:`_send_fds` /
    :meth:`_recv_fds`, :meth:`_send_ack` / :meth:`_recv_ack`, :meth:`close`.
    """

    __slots__ = ("_conn", "_server")

    def __init__(self) -> None:
        self._conn: socket.socket | None = None
        self._server: socket.socket | None = None

    # ---------- Connection management ----------

    def attach(self, conn: socket.socket) -> None:
        self.close()
        self._conn = conn

    def prepare(
        self,
        ipc_path: str,
        *,
        backlog: int = 1,
        accept_timeout: float | None = None,
    ) -> None:
        """Bind and listen; call :meth:`accept_once` before receiving."""
        if self._server is not None or self._conn is not None:
            self.close()

        with suppress(FileNotFoundError):
            os.unlink(ipc_path)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(ipc_path)
            server.listen(backlog)
            if accept_timeout is not None:
                server.settimeout(accept_timeout)
        except BaseException:
            with suppress(OSError):
                server.close()
            raise
        self._server = server

    def accept_once(self) -> None:
        if self._server is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.prepare() must be called before "
                "accept_once()",
            )
        if self._conn is not None:
            raise RuntimeError(
                f"{self.__class__.__name__} already has a connection",
            )
        conn, _addr = self._server.accept()
        self._conn = conn

    def connect(self, ipc_path: str) -> None:
        """Connect to a server socket at the given IPC path."""
        self.close()
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.connect(ipc_path)
        self._conn = conn

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
            raise RuntimeError(f"{self.__class__.__name__} is not connected")
        return self._conn

    @staticmethod
    def _recv_exact(conn: socket.socket, n: int) -> bytes:
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return b""
        buf = bytearray()
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError(
                    "connection closed before receiving expected bytes"
                )
            buf.extend(chunk)
        return bytes(buf)

    def _send_bytes(self, data: bytes) -> None:
        """Send a length-prefixed bytes payload (8-byte big-endian length + payload)."""
        conn = self._require_conn()
        conn.sendall(_BYTES_LEN_STRUCT.pack(len(data)))
        if len(data) > 0:
            conn.sendall(data)

    def _recv_bytes(self) -> bytes:
        """Receive one length-prefixed bytes payload from :meth:`_send_bytes`."""
        conn = self._require_conn()
        header = self._recv_exact(conn, _BYTES_LEN_STRUCT.size)
        (n,) = _BYTES_LEN_STRUCT.unpack(header)
        if n > (1 << 31):
            raise ValueError(f"_recv_bytes: unreasonable payload length: {n}")
        return self._recv_exact(conn, int(n))

    def _send_fds(self, fds: list[int], data: bytes) -> None:
        """Send ``data``, optionally attaching ``fds`` via ``SCM_RIGHTS``.

        If ``fds`` is empty, sends ``data`` with ``sendall`` (no ancillary data).
        Otherwise uses ``sendmsg`` with a single ``SCM_RIGHTS`` control message.
        ``data`` must always be non-empty.
        """
        if not data:
            raise ValueError("data must be non-empty")
        conn = self._require_conn()
        if not fds:
            conn.sendall(data)
            return
        fds_arr = array.array("i", fds)
        conn.sendmsg([data], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds_arr)])

    def _recv_fds(self, maxfds: int) -> tuple[bytes, list[int]]:
        """Receive one message: payload bytes plus FDs."""
        if maxfds < 0:
            raise ValueError("maxfds must be non-negative")
        conn = self._require_conn()
        fdsize = struct.calcsize("i")
        ancbufsize = socket.CMSG_SPACE(fdsize * maxfds) if maxfds > 0 else 0
        data, ancdata, msg_flags, _addr = conn.recvmsg(
            DEFAULT_RECV_BUFSIZE,
            ancbufsize,
        )
        if msg_flags & socket.MSG_TRUNC:
            raise OSError(
                "_recv_fds: payload exceeded DEFAULT_RECV_BUFSIZE; "
                "increase DEFAULT_RECV_BUFSIZE or use smaller messages",
            )
        if msg_flags & socket.MSG_CTRUNC:
            raise OSError(
                "_recv_fds: ancillary data truncated; "
                "increase maxfds or receive fewer FDs",
            )
        fds: list[int] = []
        for level, typ, cdata in ancdata:
            if level == socket.SOL_SOCKET and typ == socket.SCM_RIGHTS:
                arr = array.array("i")
                arr.frombytes(cdata)
                fds.extend(arr.tolist())
        return data, fds

    def _send_ack(self) -> None:
        """Send a small fixed payload for handshakes (e.g. per-layer ACK)."""
        self._require_conn().sendall(_ACK_PAYLOAD)

    def _recv_ack(self) -> None:
        """Receive exactly the payload from :meth:`_send_ack`."""
        expected = _ACK_PAYLOAD
        conn = self._require_conn()
        buf = bytearray()
        while len(buf) < len(expected):
            chunk = conn.recv(len(expected) - len(buf))
            if not chunk:
                raise ConnectionError("connection closed before ACK")
            buf.extend(chunk)
        if bytes(buf) != expected:
            raise ValueError(f"invalid ACK payload: {bytes(buf)!r}")
