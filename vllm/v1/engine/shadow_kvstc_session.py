# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Event-driven KVSTC receiver session for shadow CPU → cold GPU handoff.

This mirrors the threading model used by :mod:`vllm.v1.engine.shadow_tksth_session`:

- The recv thread performs *only* KVSTC transport I/O (UDS + memfd framing reused
  from the KVKTS transport) and pushes events into a queue.
- The EngineCore main loop drains the queue and performs scheduler mutations and
  GPU KV scatters, keeping all engine/scheduler state single-threaded.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import queue
import threading
import time
from contextlib import suppress
from typing import TypeAlias

import torch

from vllm.shadow.transfer.kv_transport_common import MemfdTensor, torch_dtype_from_str
from vllm.shadow.transfer.kvstc_memfd import UdsMemfdKvstcReceiverTransport
from vllm.shadow.transfer.kvstc_protocol import KvstcHandoff
from vllm.v1.core.kv_cache_manager import KVCacheBlock

logger = logging.getLogger(__name__)

__all__ = (
    "ShadowKvstcSession",
    "KvstcLayerReceived",
    "KvstcSessionDone",
    "KvstcSessionEvent",
)


@dataclasses.dataclass(slots=True)
class KvstcLayerReceived:
    """One KV layer payload received from the transport."""

    layer_idx: int
    tensor: MemfdTensor


@dataclasses.dataclass(slots=True)
class KvstcSessionDone:
    """Pushed when the recv thread exits (normally or on error)."""

    # Best-effort error string for diagnostics; transport exceptions are not raised
    # across threads.
    error: str | None = None


KvstcSessionEvent: TypeAlias = KvstcHandoff | KvstcLayerReceived | KvstcSessionDone


class ShadowKvstcSession:
    """Manages one KVSTC recv session (one connection, one handoff, N layers).

    The recv thread does only transport I/O and pushes events into ``events``.
    The EngineCore main loop calls :meth:`drain_events` and performs scheduler /
    GPU mutations — keeping all engine state single-threaded.
    """

    def __init__(
        self,
        *,
        migration_id: int,
        kvstc_ipc_path: str,
        prepare_timeout_s: float = 5.0,
    ) -> None:
        self.migration_id = migration_id
        self.kvstc_ipc_path = kvstc_ipc_path
        self.events: queue.Queue[KvstcSessionEvent] = queue.Queue()

        # Managed by the EngineCore main loop
        self.handoff: KvstcHandoff | None = None
        self.pinned_kv: torch.Tensor | None = None
        self.allocated_by_request: dict[str, list[KVCacheBlock]] = {}

        self._receiver_box: list[UdsMemfdKvstcReceiverTransport | None] = [None]
        self._prepare_evt = threading.Event()
        self._done = False

        self._thread = threading.Thread(
            target=self._thread_main,
            name="vllm-shadow-kvstc",
            daemon=True,
        )
        self._thread.start()

        if not self._prepare_evt.wait(timeout=prepare_timeout_s):
            self.cancel()
            self._thread.join(timeout=5.0)
            raise TimeoutError(f"kvstc prepare timed out for path {kvstc_ipc_path!r}")

    @property
    def is_done(self) -> bool:
        return self._done

    def cancel(self) -> None:
        """Close the listener/connection from another thread."""
        r = self._receiver_box[0]
        self._receiver_box[0] = None
        if r is not None:
            with suppress(OSError):
                r.close()

    def join(self, timeout: float = 5.0) -> None:
        self._thread.join(timeout=timeout)

    def drain_events(self) -> list[KvstcSessionEvent]:
        """Non-blocking drain of all queued events."""
        out: list[KvstcSessionEvent] = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                break
        for ev in out:
            if isinstance(ev, KvstcSessionDone):
                self._done = True
        return out

    # ------------------------------------------------------------------ #
    # Recv thread — pure I/O, no engine/scheduler access
    # ------------------------------------------------------------------ #

    def _thread_main(self) -> None:
        recv = UdsMemfdKvstcReceiverTransport()
        self._receiver_box[0] = recv
        put = self.events.put_nowait
        t_start = time.perf_counter()
        n_layers = 0
        error: str | None = None

        try:
            recv.prepare(self.kvstc_ipc_path)
            self._prepare_evt.set()
            recv.accept_once()
            logger.info("kvstc accepted path=%s", self.kvstc_ipc_path)

            handoff = recv.recv_handoff()
            self.handoff = handoff
            put(handoff)

            self.pinned_kv = torch.empty(
                2,
                handoff.shadow_num_blocks,
                handoff.num_kv_heads,
                handoff.block_size,
                handoff.head_dim,
                dtype=torch_dtype_from_str(handoff.dtype),
                device="cpu",
                pin_memory=True,
            )

            for layer_idx in range(int(handoff.num_layers)):
                tensor = recv.recv_layer(handoff)
                n_layers += 1
                put(KvstcLayerReceived(layer_idx=layer_idx, tensor=tensor))

        except (OSError, EOFError, BrokenPipeError, ValueError) as e:
            error = f"{type(e).__name__}: {e}"
            logger.warning("kvstc recv error path=%s: %s", self.kvstc_ipc_path, error)
        except BaseException as e:
            error = f"{type(e).__name__}: {e}"
            logger.exception("kvstc fatal error path=%s", self.kvstc_ipc_path)
        finally:
            self._receiver_box[0] = None
            recv.close()
            with suppress(FileNotFoundError, OSError):
                os.unlink(self.kvstc_ipc_path)
            put(KvstcSessionDone(error=error))
            elapsed = time.perf_counter() - t_start
            logger.info(
                "kvstc session end path=%s layers=%d time_s=%.3f",
                self.kvstc_ipc_path,
                n_layers,
                elapsed,
            )
