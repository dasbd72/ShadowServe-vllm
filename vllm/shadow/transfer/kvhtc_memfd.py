# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVHTC send/receive over Unix socket + ``memfd`` + ``SCM_RIGHTS``.

**Handoff (once per batch):** framed bytes (``uds.send_bytes``) carrying migration
JSON, then ``recv_ack``.

**Layer batch (per layer):** ``SharedLayerKV`` in a memfd;
``send_fds`` (FD passing) with inline payload of ``MigrationLayerEnvelope``,
then ``recv_ack``.

Avoids importing vLLM; uses ``torch`` only for ``pinned_kv`` typing and dtype.
"""

from __future__ import annotations

import mmap
import os
import socket
from contextlib import suppress
from math import prod

import torch

from vllm.shadow.transfer import uds
from vllm.shadow.transfer.kvhtc_protocol import (
    MigrationHandoff,
    MigrationLayerEnvelope,
    shared_kv_num_bytes,
    torch_dtype_from_str,
)

__all__ = (
    "MemfdTensor",
    "UdsMemfdKvHtcReceiverTransport",
    "UdsMemfdKvHtcSenderTransport",
)


class MemfdTensor:
    """Received memfd: ``mmap`` + owning fd + tensor view over the mapping.

    Call :meth:`close` to ``munmap`` and :func:`os.close` the descriptor.
    """

    __slots__ = ("_fd", "_mm", "_tensor", "_closed")

    def __init__(self, mm: mmap.mmap, fd: int, tensor: torch.Tensor) -> None:
        self._mm = mm
        self._fd = fd
        self._tensor = tensor
        self._closed = False

    @property
    def fd(self) -> int:
        return self._fd

    @property
    def mm(self) -> mmap.mmap:
        return self._mm

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    def __del__(self) -> None:
        with suppress(OSError):
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        del self._tensor
        self._mm.close()
        os.close(self._fd)
        self._closed = True


def _materialize_layer_kv(mm: mmap.mmap, handoff: MigrationHandoff) -> torch.Tensor:
    """Materialize the layer KV from the memfd mapping using handoff layout fields."""
    dt = torch_dtype_from_str(handoff.dtype)
    shape = (
        2,
        handoff.shadow_num_blocks,
        handoff.num_kv_heads,
        handoff.block_size,
        handoff.head_dim,
    )
    return torch.frombuffer(
        mm,
        dtype=dt,
        count=prod(shape),
    ).reshape(shape)


class UdsMemfdKvHtcSenderTransport:
    """Sender-side KVHTC over ``AF_UNIX`` + ``memfd`` + ``SCM_RIGHTS``."""

    __slots__ = ("_conn",)

    def __init__(self) -> None:
        self._conn: socket.socket | None = None

    def attach(self, conn: socket.socket) -> None:
        self.close()
        self._conn = conn

    def connect(self, kvhtc_ipc_path: str) -> None:
        self.close()
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.connect(kvhtc_ipc_path)
        self._conn = conn

    def send_handoff(self, handoff: MigrationHandoff) -> None:
        """Send migration handoff as framed bytes, then ``recv_ack``."""
        conn = self._require_conn()
        uds.send_bytes(conn, handoff.to_bytes())
        uds.recv_ack(conn)

    def send_layer(
        self,
        handoff: MigrationHandoff,
        layer_envelope: MigrationLayerEnvelope,
        pinned_kv: torch.Tensor,
    ) -> None:
        """Pack layer envelope + shared KV into a memfd.

        Then ``send_fds`` followed by ``recv_ack``.
        """
        conn = self._require_conn()
        if handoff.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if pinned_kv.device.type != "cpu":
            raise ValueError("pinned_kv must be a CPU tensor (pinned host memory)")
        pv = pinned_kv.detach()
        if pv.dim() != 5:
            raise ValueError(f"pinned_kv must be 5D, got shape {tuple(pv.shape)}")
        if not pv.is_contiguous():
            raise ValueError("pinned_kv must be contiguous")
        total_bytes = shared_kv_num_bytes(handoff)

        mfd_flags = getattr(os, "MFD_CLOEXEC", 0)
        fd = os.memfd_create("vllm_shadow_kv_layer", mfd_flags)
        mm: mmap.mmap | None = None
        try:
            os.ftruncate(fd, total_bytes)
            mm = mmap.mmap(fd, total_bytes, mmap.MAP_SHARED, mmap.PROT_WRITE)
            out = torch.frombuffer(
                mm,
                dtype=pv.dtype,
                count=pv.numel(),
            ).reshape(pv.shape)
            out.copy_(pv, non_blocking=True)

            uds.send_fds(conn, [fd], layer_envelope.to_bytes())
            uds.recv_ack(conn)
        finally:
            if mm is not None:
                mm.close()
            os.close(fd)

    def close(self) -> None:
        if self._conn is not None:
            with suppress(OSError):
                self._conn.close()
            self._conn = None

    def _require_conn(self) -> socket.socket:
        if self._conn is None:
            raise RuntimeError("UdsMemfdKvHtcSenderTransport is not connected")
        return self._conn


class UdsMemfdKvHtcReceiverTransport:
    """Receiver-side KVHTC over ``AF_UNIX`` + ``memfd`` + ``SCM_RIGHTS``.

    This transport owns the server socket (from :meth:`prepare`) and the accepted
    connection socket (from :meth:`accept_once`).
    """

    __slots__ = ("_conn", "_server")

    def __init__(self) -> None:
        self._conn: socket.socket | None = None
        self._server: socket.socket | None = None

    def prepare(
        self,
        kvhtc_ipc_path: str,
        *,
        backlog: int = 1,
        accept_timeout: float | None = None,
    ) -> None:
        """Bind and listen; call :meth:`accept_once` before :meth:`recv_handoff`."""
        if self._server is not None or self._conn is not None:
            self.close()

        with suppress(FileNotFoundError):
            os.unlink(kvhtc_ipc_path)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(kvhtc_ipc_path)
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
                "UdsMemfdKvHtcReceiverTransport.prepare() must be called before "
                "accept_once()"
            )
        if self._conn is not None:
            raise RuntimeError(
                "UdsMemfdKvHtcReceiverTransport already has a connection"
            )
        conn, _addr = self._server.accept()
        self._conn = conn

    def recv_handoff(self) -> MigrationHandoff:
        """``recv_bytes`` → parse envelope + JSON → ACK → return handoff."""
        conn = self._require_conn()
        payload = uds.recv_bytes(conn)
        handoff = MigrationHandoff.from_bytes(payload)
        uds.send_ack(conn)
        return handoff

    def recv_layer(
        self, handoff: MigrationHandoff
    ) -> tuple[MigrationLayerEnvelope, MemfdTensor]:
        """``recv_fds`` → parse envelope → ACK.

        Returns envelope + :class:`MemfdTensor` (KV view in :attr:`MemfdTensor.tensor`).
        """
        conn = self._require_conn()
        data, fds = uds.recv_fds(conn, 1)
        envelope = MigrationLayerEnvelope.from_bytes(data)
        if len(fds) != 1:
            raise ValueError(f"expected exactly 1 memfd, got {len(fds)} fds")

        fd = fds[0]
        mm: mmap.mmap | None = None
        try:
            st = os.fstat(fd)
            size = int(st.st_size)
            if size <= 0:
                raise ValueError("memfd has empty size")
            mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_WRITE)
            expected_bytes = shared_kv_num_bytes(handoff)
            if size != expected_bytes:
                raise ValueError(
                    f"layer memfd size {size} != expected {expected_bytes} "
                    f"(envelope + SharedLayerKV for handoff layout)"
                )
            uds.send_ack(conn)
        except BaseException as e:
            if mm is not None:
                mm.close()
            os.close(fd)
            raise e

        layer_kv = _materialize_layer_kv(mm, handoff)
        tensor = MemfdTensor(mm, fd, layer_kv)

        return envelope, tensor

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
            raise RuntimeError(
                "UdsMemfdKvHtcReceiverTransport has no connection "
                "(call prepare() then accept_once() before recv)"
            )
        return self._conn
