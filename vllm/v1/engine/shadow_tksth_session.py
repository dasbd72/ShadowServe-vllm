# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Event-driven TKSTH bridge for shadow token streaming.

The recv thread only performs I/O and pushes wire-protocol messages
(from :mod:`vllm.shadow.transfer.tksth_protocol`) plus one sentinel
into a queue.  The EngineCore main loop drains the queue and performs
scheduler / output mutations — keeping all scheduler access
single-threaded.

No intermediate "bridge event" types are created; the wire types
(``TksthTokenDelta``, ``TksthFinish``, ``TksthError``) flow directly to the consumer.
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

from vllm.shadow.transfer.tksth_protocol import (
    TksthError,
    TksthFinish,
    TksthFinishReason,
    TksthMessage,
    TksthTokenDelta,
)
from vllm.shadow.transfer.tksth_uds import UdsTksthReceiverTransport
from vllm.v1.engine import FinishReason
from vllm.v1.request import Request, RequestStatus

logger = logging.getLogger(__name__)

__all__ = (
    "ShadowTksthSession",
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
    sent a ``TksthFinish`` or ``TksthError`` for before the transport closed.
    """

    orphaned_request_ids: list[str]


ShadowSessionEvent: TypeAlias = TksthMessage | ShadowSessionDone
"""Union of wire-protocol messages and the session-done sentinel."""


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def shadow_finish_to_engine_and_status(
    shadow_reason: TksthFinishReason,
) -> tuple[FinishReason, RequestStatus]:
    if shadow_reason == "stop":
        return FinishReason.STOP, RequestStatus.FINISHED_STOPPED
    if shadow_reason == "length":
        return FinishReason.LENGTH, RequestStatus.FINISHED_LENGTH_CAPPED
    if shadow_reason == "abort":
        return FinishReason.ABORT, RequestStatus.FINISHED_ABORTED
    if shadow_reason == "error":
        return FinishReason.ERROR, RequestStatus.FINISHED_ERROR
    if shadow_reason == "migrated":
        return FinishReason.MIGRATED, RequestStatus.FINISHED_MIGRATED
    raise ValueError(f"unknown shadow finish reason: {shadow_reason}")


# ---------------------------------------------------------------------- #
# Session
# ---------------------------------------------------------------------- #


class ShadowTksthSession:
    """Manages one TKSTH recv session.

    The recv thread only does transport I/O and pushes wire-protocol
    messages (plus :class:`ShadowSessionDone`) into ``events``.
    The EngineCore main loop calls :meth:`drain_events` and performs
    scheduler / output mutations — keeping all scheduler access
    single-threaded.
    """

    def __init__(
        self,
        *,
        tksth_ipc_path: str,
        migration_id: int,
        requests: list[Request],
        prepare_timeout_s: float = 10.0,
    ) -> None:
        self.migration_id = migration_id
        self.tksth_ipc_path = tksth_ipc_path
        self.requests: dict[str, Request] = {r.request_id: r for r in requests}
        self.events: queue.Queue[ShadowSessionEvent] = queue.Queue()

        self._receiver_box: list[UdsTksthReceiverTransport | None] = [None]
        self._prepare_evt = threading.Event()
        self._done = False

        self._thread = threading.Thread(
            target=self._thread_main,
            name=f"vllm-shadow-tksth-{migration_id}",
            daemon=True,
        )
        self._thread.start()

        if not self._prepare_evt.wait(timeout=prepare_timeout_s):
            self.cancel()
            self._thread.join(timeout=5.0)
            raise TimeoutError(
                f"tksth prepare timed out for path {tksth_ipc_path!r} "
                f"(migration_id={migration_id})"
            )

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
        recv = UdsTksthReceiverTransport()
        self._receiver_box[0] = recv
        pending: set[str] = set(self.requests.keys())
        t_start = time.perf_counter()
        n_frames = 0
        mid = self.migration_id
        put = self.events.put_nowait

        try:
            recv.prepare(self.tksth_ipc_path)
            self._prepare_evt.set()
            recv.accept_once()
            logger.info(
                "tksth accepted migration_id=%s path=%s pending_requests=%d",
                mid,
                self.tksth_ipc_path,
                len(pending),
            )

            while pending:
                try:
                    msg = recv.recv()
                except (OSError, EOFError, BrokenPipeError) as e:
                    logger.warning(
                        "tksth recv transport error migration_id=%s: %s",
                        mid,
                        e,
                    )
                    break
                except ValueError:
                    logger.exception(
                        "tksth bad frame migration_id=%s",
                        mid,
                    )
                    break

                n_frames += 1

                if getattr(msg, "migration_id", None) != mid:
                    logger.warning(
                        "tksth migration_id mismatch got=%s expected=%s",
                        getattr(msg, "migration_id", None),
                        mid,
                    )
                    continue

                if isinstance(msg, TksthTokenDelta):
                    if msg.request_id in pending:
                        put(msg)

                elif isinstance(msg, TksthFinish):
                    if msg.request_id in pending:
                        put(msg)
                        pending.discard(msg.request_id)

                elif isinstance(msg, TksthError):
                    put(msg)
                    if msg.request_id is not None:
                        pending.discard(msg.request_id)
                    else:
                        pending.clear()

                else:
                    logger.warning(
                        "tksth unexpected message type migration_id=%s: %s",
                        mid,
                        type(msg).__name__,
                    )

            put(ShadowSessionDone(orphaned_request_ids=list(pending)))
        finally:
            self._receiver_box[0] = None
            recv.close()
            with suppress(FileNotFoundError, OSError):
                os.unlink(self.tksth_ipc_path)
            elapsed = time.perf_counter() - t_start
            logger.info(
                "tksth session end migration_id=%s frames=%d time_s=%.3f",
                mid,
                n_frames,
                elapsed,
            )
