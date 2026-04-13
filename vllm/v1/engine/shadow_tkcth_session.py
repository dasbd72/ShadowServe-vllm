# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Event-driven TKCTH bridge for shadow token streaming.

The recv thread only performs I/O and pushes wire-protocol messages
(from :mod:`vllm.shadow.transfer.tkcth_protocol`) plus one sentinel
into a queue.  The EngineCore main loop drains the queue and performs
scheduler / output mutations — keeping all scheduler access
single-threaded.

No intermediate "bridge event" types are created; the wire types
(``TkcthTokenDelta``, ``TkcthFinish``, ``TkcthError``) flow directly to the consumer.
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

from vllm.shadow.transfer.tkcth_protocol import (
    TkcthError,
    TkcthFinish,
    TkcthMessage,
    TkcthTokenDelta,
)
from vllm.shadow.transfer.tkcth_uds import UdsTkcthReceiverTransport
from vllm.v1.engine import FinishReason
from vllm.v1.request import RequestStatus

logger = logging.getLogger(__name__)

__all__ = (
    "ShadowTkcthSession",
    "ShadowSessionDone",
    "shadow_finish_to_engine_and_status",
)


# ---------------------------------------------------------------------- #
# Sentinel event (no wire equivalent)
# ---------------------------------------------------------------------- #


@dataclasses.dataclass(slots=True)
class ShadowSessionDone:
    """Pushed when the recv thread exits (normally or on error).

    ``orphaned_request_ids`` contains requests that the shadow never
    sent a ``TkcthFinish`` or ``TkcthError`` for before the transport closed.
    """

    orphaned_request_ids: list[str]


ShadowSessionEvent: TypeAlias = TkcthMessage | ShadowSessionDone
"""Union of wire-protocol messages and the session-done sentinel."""


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def shadow_finish_to_engine_and_status(
    shadow_reason: str | None,
) -> tuple[FinishReason, RequestStatus]:
    if shadow_reason is None:
        return FinishReason.STOP, RequestStatus.FINISHED_STOPPED
    s = shadow_reason.strip().lower()
    if s == "length":
        return FinishReason.LENGTH, RequestStatus.FINISHED_LENGTH_CAPPED
    if s == "error":
        return FinishReason.ERROR, RequestStatus.FINISHED_ERROR
    if s == "abort":
        return FinishReason.ABORT, RequestStatus.FINISHED_ABORTED
    if s in ("stop", "migrated", ""):
        return FinishReason.STOP, RequestStatus.FINISHED_STOPPED
    return FinishReason.STOP, RequestStatus.FINISHED_STOPPED


# ---------------------------------------------------------------------- #
# Session
# ---------------------------------------------------------------------- #


class ShadowTkcthSession:
    """Manages one TKCTH recv session.

    The recv thread only does transport I/O and pushes wire-protocol
    messages (plus :class:`ShadowSessionDone`) into ``events``.
    The EngineCore main loop calls :meth:`drain_events` and performs
    scheduler / output mutations — keeping all scheduler access
    single-threaded.
    """

    def __init__(
        self,
        *,
        tkcth_ipc_path: str,
        migration_id: int,
        request_ids: list[str],
        request_id_to_client_index: dict[str, int],
        recv_ready_event: threading.Event,
    ) -> None:
        self.migration_id = migration_id
        self.tkcth_ipc_path = tkcth_ipc_path
        self.request_id_to_client_index = request_id_to_client_index
        self.events: queue.Queue[ShadowSessionEvent] = queue.Queue()

        self._recv_ready_event = recv_ready_event
        self._initial_request_ids: set[str] = set(request_ids)
        self._receiver_box: list[UdsTkcthReceiverTransport | None] = [None]
        self._prepare_evt = threading.Event()
        self._done = False

        self._thread = threading.Thread(
            target=self._thread_main,
            name=f"vllm-shadow-tkcth-{migration_id}",
            daemon=True,
        )
        self._thread.start()

        if not self._prepare_evt.wait(timeout=120.0):
            self.cancel()
            self._thread.join(timeout=5.0)
            raise TimeoutError(
                f"tkcth prepare timed out for path {tkcth_ipc_path!r} "
                f"(migration_id={migration_id})"
            )

    @property
    def is_done(self) -> bool:
        return self._done

    def cancel(self) -> None:
        """Close the listener/connection from another thread."""
        self._recv_ready_event.set()
        r = self._receiver_box[0]
        if r is not None:
            r.close()

    def join(self, timeout: float = 5.0) -> None:
        self._thread.join(timeout=timeout)

    def drain_events(self) -> list[ShadowSessionEvent]:
        """Non-blocking drain of all queued events.

        Called from the EngineCore main loop.  If a
        :class:`ShadowSessionDone` event is found, this session is marked
        as done.
        """
        out: list[ShadowSessionEvent] = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                break
        for ev in out:
            if isinstance(ev, ShadowSessionDone):
                self._done = True
        return out

    # ------------------------------------------------------------------ #
    # Recv thread — pure I/O, no scheduler access
    # ------------------------------------------------------------------ #

    def _thread_main(self) -> None:
        recv = UdsTkcthReceiverTransport()
        self._receiver_box[0] = recv
        pending: set[str] = set(self._initial_request_ids)
        t_start = time.perf_counter()
        n_frames = 0
        mid = self.migration_id
        put = self.events.put_nowait

        try:
            recv.prepare(self.tkcth_ipc_path)
            self._prepare_evt.set()
            recv.accept_once()
            logger.info(
                "tkcth accepted migration_id=%s path=%s pending_requests=%d",
                mid,
                self.tkcth_ipc_path,
                len(pending),
            )

            if not self._recv_ready_event.wait(timeout=300.0):
                logger.error("tkcth recv_ready_event timeout migration_id=%s", mid)

            while pending:
                try:
                    msg = recv.recv()
                except (OSError, EOFError, BrokenPipeError) as e:
                    logger.warning(
                        "tkcth recv transport error migration_id=%s: %s",
                        mid,
                        e,
                    )
                    break
                except ValueError:
                    logger.exception(
                        "tkcth bad frame migration_id=%s",
                        mid,
                    )
                    break

                n_frames += 1

                if getattr(msg, "migration_id", None) != mid:
                    logger.warning(
                        "tkcth migration_id mismatch got=%s expected=%s",
                        getattr(msg, "migration_id", None),
                        mid,
                    )
                    continue

                if isinstance(msg, TkcthTokenDelta):
                    if msg.request_id in pending:
                        put(msg)

                elif isinstance(msg, TkcthFinish):
                    if msg.request_id in pending:
                        put(msg)
                        pending.discard(msg.request_id)

                elif isinstance(msg, TkcthError):
                    put(msg)
                    if msg.request_id is not None:
                        pending.discard(msg.request_id)
                    else:
                        pending.clear()

                else:
                    logger.warning(
                        "tkcth unexpected message type migration_id=%s: %s",
                        mid,
                        type(msg).__name__,
                    )

            put(ShadowSessionDone(orphaned_request_ids=list(pending)))
        finally:
            self._receiver_box[0] = None
            recv.close()
            with suppress(FileNotFoundError, OSError):
                os.unlink(self.tkcth_ipc_path)
            elapsed = time.perf_counter() - t_start
            logger.info(
                "tkcth session end migration_id=%s frames=%d time_s=%.3f",
                mid,
                n_frames,
                elapsed,
            )
