# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TKSTH UDS helpers: framed JSON message send/recv over a stream socket.

TKSTH uses the same framing as :class:`~vllm.shadow.transfer.uds.UdsTransport`
:meth:`~vllm.shadow.transfer.uds.UdsTransport._send_bytes` / ``_recv_bytes``:
an 8-byte big-endian length prefix followed by a UTF-8 JSON payload.
"""

from __future__ import annotations

from vllm.shadow.transfer.tksth_protocol import (
    TksthMessage,
    tksth_message_from_bytes,
)
from vllm.shadow.transfer.uds import UdsTransport

__all__ = (
    "UdsTksthSenderTransport",
    "UdsTksthReceiverTransport",
)


class UdsTksthSenderTransport(UdsTransport):
    """Sender-side TKSTH over ``AF_UNIX`` ``SOCK_STREAM`` with framed JSON."""

    def send(self, msg: TksthMessage) -> None:
        self._send_bytes(msg.to_bytes())


class UdsTksthReceiverTransport(UdsTransport):
    """Receiver-side TKSTH over ``AF_UNIX`` ``SOCK_STREAM`` with framed JSON."""

    def recv(self) -> TksthMessage:
        return tksth_message_from_bytes(self._recv_bytes())
