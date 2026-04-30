# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVHTS send/receive over Unix socket + ``memfd`` + ``SCM_RIGHTS``."""

from __future__ import annotations

from vllm.shadow.transfer.kv_transport_common import (
    LAYER_MEMFD_INLINE_PAYLOAD,
    MemfdTensor,
    torch_dtype_from_str,
)
from vllm.shadow.transfer.kvhts_protocol import KvhtsHandoff
from vllm.shadow.transfer.uds import UdsTransport

__all__ = (
    "MemfdTensor",
    "UdsMemfdKvhtsReceiverTransport",
    "UdsMemfdKvhtsSenderTransport",
)


class UdsMemfdKvhtsSenderTransport(UdsTransport):
    """Sender-side KVHTS over ``AF_UNIX`` + ``memfd`` + ``SCM_RIGHTS``."""

    def send_handoff(self, handoff: KvhtsHandoff) -> None:
        self._send_bytes(handoff.to_bytes())
        self._recv_ack()

    def send_layer(self, tensor: MemfdTensor) -> None:
        self._send_fds([tensor.fd], LAYER_MEMFD_INLINE_PAYLOAD)
        self._recv_ack()
        tensor.close()


class UdsMemfdKvhtsReceiverTransport(UdsTransport):
    """Receiver-side KVHTS over ``AF_UNIX`` + ``memfd`` + ``SCM_RIGHTS``.

    This transport owns the server socket (from :meth:`prepare`) and the accepted
    connection socket (from :meth:`accept_once`).
    """

    def recv_handoff(self) -> KvhtsHandoff:
        """``_recv_bytes`` → parse envelope + JSON → ACK → return handoff."""
        payload = self._recv_bytes()
        handoff = KvhtsHandoff.from_bytes(payload)
        self._send_ack()
        return handoff

    def recv_layer(self, handoff: KvhtsHandoff) -> MemfdTensor:
        """``_recv_fds`` → validate inline payload → ACK → return memfd tensor."""
        data, fds = self._recv_fds(1)
        if data != LAYER_MEMFD_INLINE_PAYLOAD:
            raise ValueError(
                "unexpected layer inline payload: "
                f"{data!r} (expected {LAYER_MEMFD_INLINE_PAYLOAD!r})"
            )
        if len(fds) != 1:
            raise ValueError(f"expected exactly 1 memfd, got {len(fds)} fds")
        self._send_ack()

        fd = fds[0]
        dtype = torch_dtype_from_str(handoff.dtype)
        shape = (
            2,
            handoff.shadow_num_blocks,
            handoff.num_kv_heads,
            handoff.block_size,
            handoff.head_dim,
        )
        tensor = MemfdTensor.from_memfd(fd, dtype, shape)
        return tensor
