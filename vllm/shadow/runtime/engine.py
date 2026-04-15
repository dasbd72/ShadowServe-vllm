# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shadow CPU engine: KVHTC receive loop and shadow worker (decode + TKCTH)."""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from contextlib import suppress

from vllm.shadow.models.registry import load_shadow_model_from_config
from vllm.shadow.runtime.config import ShadowConfig
from vllm.shadow.runtime.executor import ShadowExecutor
from vllm.shadow.transfer.kvhtc_memfd import (
    MemfdTensor,
    UdsMemfdKvHtcReceiverTransport,
)
from vllm.shadow.transfer.kvhtc_protocol import MigrationHandoff

logger = logging.getLogger("vllm.shadow.runtime.engine")

_ACCEPT_POLL_S = 1.0
_ACCEPT_BACKLOG = 128


__all__ = ("ShadowEngine",)


class ShadowEngine:
    """Shadow CPU process with KVHTC migration receive loop.

    Model weights are loaded eagerly at startup so they are ready before the
    first migration arrives. The engine thread runs one accept → receive →
    decode session, then signals shutdown so the process can exit.
    """

    def __init__(
        self,
        ready_file: str,
        shadow_kvhtc_ipc_prefix: str,
        shadow_config: ShadowConfig,
    ) -> None:
        self._ready_file: str = ready_file
        self._shadow_kvhtc_ipc_path: str = (
            f"{shadow_kvhtc_ipc_prefix}-{secrets.token_hex(8)}"
        )
        self._shadow_config: ShadowConfig = shadow_config

        self._kvhtc_receiver: UdsMemfdKvHtcReceiverTransport | None = None
        self._executor: ShadowExecutor | None = None

        t0 = time.perf_counter()
        self._model = load_shadow_model_from_config(self._shadow_config)
        logger.info(
            "Shadow CPU weights ready (load_wall_ms=%.1f).",
            (time.perf_counter() - t0) * 1000.0,
        )

        self._engine_thread: threading.Thread | None = None
        self._engine_error: BaseException | None = None
        self._stop_event = threading.Event()

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the shadow engine in a background thread."""
        if self._engine_thread is not None:
            raise RuntimeError("ShadowEngine.start() already called")
        self._engine_error = None
        self._stop_event.clear()
        self._engine_thread = threading.Thread(
            target=self._run_engine,
            name="shadow-cpu-engine",
            daemon=False,
        )
        self._engine_thread.start()

        logger.info(
            "Shadow CPU process running (PID %d). Waiting for shutdown signal...",
            os.getpid(),
        )

    def stop(self, *, reason: str | None = None) -> None:
        """Request shutdown and release blocking resources.

        This method is safe to call multiple times.
        """
        if reason:
            logger.info("Shadow CPU engine stopping: %s", reason)
        else:
            logger.info("Shadow CPU engine stopping.")
        self._stop_event.set()
        # Ensure accept() unblocks promptly.
        self._close_kvhtc_receiver()
        if self._executor is not None:
            self._executor.stop()

    def join(self, timeout: float | None = None) -> None:
        """Wait for engine shutdown and re-raise any background exception."""
        if self._engine_thread is None:
            raise RuntimeError("ShadowEngine.start() was not called")
        self._engine_thread.join(timeout=timeout)
        if timeout is not None and self._engine_thread.is_alive():
            raise TimeoutError("shadow engine did not finish within timeout")
        err = self._engine_error
        if err is not None:
            raise err

    # ── Engine ──────────────────────────────────────────────────────

    def _run_engine(self) -> None:
        try:
            self._initialize()
            self._accept_once_loop()
            if self._stop_requested():
                return
            self._run_migration_session()
        except BaseException as e:
            self._engine_error = e
        finally:
            self._teardown()

    def _initialize(self) -> None:
        device = os.environ.get("VLLM_TARGET_DEVICE", "")
        if device and device != "cpu":
            logger.warning(
                "VLLM_TARGET_DEVICE is '%s', expected 'cpu'. "
                "The shadow process is designed for CPU-only operation.",
                device,
            )

        self._prepare_kvhtc_receiver()
        self._emit_ready()

    def _teardown(self) -> None:
        self._close_kvhtc_receiver()
        if os.path.exists(self._ready_file):
            with suppress(OSError):
                os.unlink(self._ready_file)
        ipc = self._shadow_kvhtc_ipc_path
        if ipc and os.path.exists(ipc):
            with suppress(OSError):
                os.unlink(ipc)

    def _prepare_kvhtc_receiver(self) -> None:
        transport = UdsMemfdKvHtcReceiverTransport()
        transport.prepare(
            self._shadow_kvhtc_ipc_path,
            backlog=_ACCEPT_BACKLOG,
            accept_timeout=_ACCEPT_POLL_S,
        )
        self._kvhtc_receiver = transport
        logger.info(
            "Listening for kvhtc migration on IPC %s.",
            self._shadow_kvhtc_ipc_path,
        )

    def _close_kvhtc_receiver(self) -> None:
        if self._kvhtc_receiver is not None:
            with suppress(OSError):
                self._kvhtc_receiver.close()
            self._kvhtc_receiver = None

    def _emit_ready(self) -> None:
        """Write a ready-file so the entrypoint (or tests) can detect startup.

        Format: line 1 = PID, line 2 = KVHTC IPC socket path (no trailing newline).
        """
        try:
            with open(self._ready_file, "w") as f:
                f.write(str(os.getpid()))
                f.write("\n")
                f.write(self._shadow_kvhtc_ipc_path)
            logger.info("Ready signal written to %s", self._ready_file)
        except OSError:
            logger.warning(
                "Could not write ready-file %s", self._ready_file, exc_info=True
            )

    # ── Migration session (engine thread: accept → receive → decode) ──

    def _stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def _accept_once_loop(self) -> None:
        """Wait for one KVHTC connection, run migration + decode, then exit."""
        assert self._kvhtc_receiver is not None
        receiver = self._kvhtc_receiver

        while not self._stop_requested():
            try:
                receiver.accept_once()
                break
            except TimeoutError:
                continue
            except OSError:
                if self._stop_requested():
                    return
                logger.warning("accept failed", exc_info=True)
                continue

        if self._stop_requested():
            return

    def _run_migration_session(self) -> None:
        assert self._kvhtc_receiver is not None
        receiver = self._kvhtc_receiver

        handoff = receiver.recv_handoff()
        logger.info(
            "Received migration handoff: migration_id=%s num_layers=%s batch_size=%s",
            handoff.migration_id,
            handoff.num_layers,
            handoff.batch_size,
        )

        layer_tensors = self._recv_all_layers(receiver, handoff)
        self._executor = ShadowExecutor(
            handoff, model=self._model, layer_kv_tensors=layer_tensors
        )
        try:
            self._executor.start()
            self._executor.join()
        finally:
            self._executor = None

    @staticmethod
    def _recv_all_layers(
        receiver: UdsMemfdKvHtcReceiverTransport,
        handoff: MigrationHandoff,
    ) -> list[MemfdTensor]:
        out: list[MemfdTensor] = []
        for layer_idx in range(handoff.num_layers):
            envelope, tensor = receiver.recv_layer(handoff)
            if layer_idx != envelope.layer_idx:
                raise ValueError(
                    f"layer index mismatch: {layer_idx} != {envelope.layer_idx}"
                )
            out.append(tensor)
            logger.info(
                "Received layer %d: migration_id=%s",
                layer_idx,
                handoff.migration_id,
            )
        return out
