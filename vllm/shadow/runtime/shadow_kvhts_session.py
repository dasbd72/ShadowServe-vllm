# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVHTS receiver session for hot GPU → shadow CPU handoff."""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import suppress

from vllm.shadow.transfer.kv_transport_common import MemfdTensor
from vllm.shadow.transfer.kvhts_memfd import UdsMemfdKvhtsReceiverTransport
from vllm.shadow.transfer.kvhts_protocol import KvhtsHandoff

logger = logging.getLogger(__name__)

__all__ = ("ShadowKvhtsSession",)


class ShadowKvhtsSession:
    """Manages one KVHTS recv session (one connection, one handoff, N layers)."""

    def __init__(
        self,
        *,
        migration_id: int,
        kvhts_ipc_path: str,
        prepare_timeout_s: float = 5.0,
    ) -> None:
        self.migration_id = migration_id
        self.kvhts_ipc_path = kvhts_ipc_path

        self.handoff: KvhtsHandoff | None = None
        self.layers: list[MemfdTensor] | None = None

        self._receiver_box: list[UdsMemfdKvhtsReceiverTransport | None] = [None]
        self._prepare_evt = threading.Event()
        self._done = False

        self._thread = threading.Thread(
            target=self._thread_main,
            name="vllm-shadow-kvhts",
            daemon=True,
        )
        self._thread.start()

        if not self._prepare_evt.wait(timeout=prepare_timeout_s):
            self.cancel()
            self._thread.join(timeout=5.0)
            raise TimeoutError(f"kvhts prepare timed out for path {kvhts_ipc_path!r}")

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

    def _thread_main(self) -> None:
        recv = UdsMemfdKvhtsReceiverTransport()
        self._receiver_box[0] = recv
        t_start = time.perf_counter()
        n_layers = 0
        error: str | None = None

        try:
            recv.prepare(self.kvhts_ipc_path)
            self._prepare_evt.set()
            recv.accept_once()
            logger.info("kvhts accepted path=%s", self.kvhts_ipc_path)

            handoff = recv.recv_handoff()
            assert handoff.migration_id == self.migration_id
            layers = []
            for _ in range(int(handoff.num_layers)):
                tensor = recv.recv_layer(handoff)
                layers.append(tensor)
                n_layers += 1

            self.handoff = handoff
            self.layers = layers
        except (OSError, EOFError, BrokenPipeError, ValueError) as e:
            error = f"{type(e).__name__}: {e}"
            logger.warning("kvhts recv error path=%s: %s", self.kvhts_ipc_path, error)
        except BaseException as e:
            error = f"{type(e).__name__}: {e}"
            logger.exception("kvhts fatal error path=%s", self.kvhts_ipc_path)
        finally:
            self._receiver_box[0] = None
            recv.close()
            with suppress(FileNotFoundError, OSError):
                os.unlink(self.kvhts_ipc_path)
            logger.info(
                "kvhts session end path=%s layers=%d time_s=%.3f",
                self.kvhts_ipc_path,
                n_layers,
                time.perf_counter() - t_start,
            )
            self._done = True
