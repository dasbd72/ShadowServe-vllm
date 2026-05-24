# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVSTC send/receive over Unix socket + ``memfd`` + ``SCM_RIGHTS``."""

from __future__ import annotations

from typing import Final

from vllm.shadow.transfer.kv_transport_common import MemfdTensor, torch_dtype_from_str
from vllm.shadow.transfer.kvstc_protocol import KvstcHandoff
from vllm.shadow.transfer.uds import UdsTransport

__all__ = (
    "MemfdTensor",
    "UdsMemfdKvstcReceiverTransport",
    "UdsMemfdKvstcSenderTransport",
)

_LAYER_MEMFD_INLINE_PAYLOAD: Final[bytes] = b"\x02"


class UdsMemfdKvstcSenderTransport(UdsTransport):
    """Sender-side KVSTC over ``AF_UNIX`` + ``memfd`` + ``SCM_RIGHTS``."""

    def send_handoff(self, handoff: KvstcHandoff) -> None:
        self._send_bytes(handoff.to_bytes())
        self._recv_ack()

    def send_layer(self, tensor: MemfdTensor) -> None:
        self._send_fds([tensor.fd], _LAYER_MEMFD_INLINE_PAYLOAD)
        self._recv_ack()
        tensor.close()


class UdsMemfdKvstcReceiverTransport(UdsTransport):
    """Receiver-side KVSTC over ``AF_UNIX`` + ``memfd`` + ``SCM_RIGHTS``."""

    def recv_handoff(self) -> KvstcHandoff:
        payload = self._recv_bytes()
        handoff = KvstcHandoff.from_bytes(payload)
        self._send_ack()
        return handoff

    def recv_layer(self, handoff: KvstcHandoff) -> MemfdTensor:
        """``_recv_fds`` → validate inline payload → ACK → return memfd tensor."""
        data, fds = self._recv_fds(1)
        if data != _LAYER_MEMFD_INLINE_PAYLOAD:
            raise ValueError(
                "unexpected layer inline payload: "
                f"{data!r} (expected {_LAYER_MEMFD_INLINE_PAYLOAD!r})"
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
