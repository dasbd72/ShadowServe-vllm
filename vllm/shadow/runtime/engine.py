# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shadow CPU engine: one KVHTS migration, CPU decode, TKSTH token output."""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from typing import Any

from vllm.shadow.http.server import KvhtsHttpState, start_shadow_http_server
from vllm.shadow.models.kv_state import ShadowKvState
from vllm.shadow.models.llama import LlamaForCausalLM
from vllm.shadow.models.registry import load_shadow_model_from_config
from vllm.shadow.models.sampling import sampling_params_from_wire_dict
from vllm.shadow.runtime.config import ShadowConfig
from vllm.shadow.runtime.executor import ShadowExecutor
from vllm.shadow.transfer.kv_transport_common import MemfdTensor
from vllm.shadow.transfer.kvhts_memfd import UdsMemfdKvhtsReceiverTransport
from vllm.shadow.transfer.kvhts_protocol import KvhtsHandoff

logger = logging.getLogger("vllm.shadow.runtime.engine")

_ACCEPT_POLL_S = 1.0
_ACCEPT_BACKLOG = 128


__all__ = ("ShadowEngine",)


class _ModelLoader:
    def __init__(self, shadow_config: ShadowConfig) -> None:
        self._shadow_config: ShadowConfig = shadow_config

    def run(self) -> LlamaForCausalLM:
        t0 = time.perf_counter()
        model = load_shadow_model_from_config(self._shadow_config)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "Shadow model loaded model_type=%s dtype=%s layers=%d time_ms=%.1f",
            model.hf.model_type,
            self._shadow_config.dtype,
            model.hf.num_hidden_layers,
            elapsed_ms,
        )
        return model


@dataclass(slots=True)
class _MigrationSessionResults:
    handoff: KvhtsHandoff
    kv_state: ShadowKvState


class _MigrationSession:
    def __init__(
        self,
        shadow_kvhts_ipc_path: str,
        listener_ready_callback: Callable[[], Any],
    ) -> None:
        self._shadow_kvhts_ipc_path: str = shadow_kvhts_ipc_path
        self._listener_ready_callback: Callable[[], Any] = listener_ready_callback

        self._stop_event = threading.Event()

        self._kvhts_receiver: UdsMemfdKvhtsReceiverTransport | None = None

    def stop(self) -> None:
        self._stop_event.set()
        self._close_kvhts_receiver()

    def run(self) -> _MigrationSessionResults | None:
        try:
            self._prepare_kvhts_receiver()
            if self._stop_event.is_set():
                return None
            self._accept_once_loop()
            if self._stop_event.is_set():
                return None
            return self._run_migration_session()
        finally:
            self._teardown()

    def _teardown(self) -> None:
        self._close_kvhts_receiver()
        ipc = self._shadow_kvhts_ipc_path
        if ipc and os.path.exists(ipc):
            with suppress(OSError):
                os.unlink(ipc)

    def _prepare_kvhts_receiver(self) -> None:
        transport = UdsMemfdKvhtsReceiverTransport()
        transport.prepare(
            self._shadow_kvhts_ipc_path,
            backlog=_ACCEPT_BACKLOG,
            accept_timeout=_ACCEPT_POLL_S,
        )
        self._kvhts_receiver = transport
        logger.info(
            "Listening for kvhts migration on IPC %s.",
            self._shadow_kvhts_ipc_path,
        )
        self._listener_ready_callback()

    def _close_kvhts_receiver(self) -> None:
        if self._kvhts_receiver is not None:
            with suppress(OSError):
                self._kvhts_receiver.close()
            self._kvhts_receiver = None

    def _accept_once_loop(self) -> None:
        """Wait for one KVHTS connection."""
        assert self._kvhts_receiver is not None
        receiver = self._kvhts_receiver

        while not self._stop_event.is_set():
            try:
                receiver.accept_once()
                break
            except TimeoutError:
                continue
            except OSError:
                if self._stop_event.is_set():
                    return
                logger.warning("accept failed", exc_info=True)
                continue

    def _run_migration_session(self) -> _MigrationSessionResults:
        assert self._kvhts_receiver is not None
        receiver = self._kvhts_receiver

        handoff = receiver.recv_handoff()
        logger.info(
            "Received migration handoff: migration_id=%s num_layers=%s batch_size=%s",
            handoff.migration_id,
            handoff.num_layers,
            handoff.batch_size,
        )

        layer_tensors: list[MemfdTensor] = []
        for layer_idx in range(handoff.num_layers):
            tensor = receiver.recv_layer(handoff)
            layer_tensors.append(tensor)
            logger.info(
                "Received layer %d: migration_id=%s",
                layer_idx,
                handoff.migration_id,
            )

        if not handoff.requests:
            raise ValueError("migration handoff has no requests")

        kv_state = ShadowKvState(
            num_layers=handoff.num_layers,
            num_blocks=handoff.shadow_num_blocks,
            num_kv_heads=handoff.num_kv_heads,
            head_dim=handoff.head_dim,
            block_size=handoff.block_size,
            dtype=handoff.dtype,
        )
        for layer_idx, t in enumerate(layer_tensors):
            kv_state.register_layer(layer_idx, t)
        for req in handoff.requests:
            kv_state.add_request(
                request_id=req.request_id,
                prompt_token_ids=req.prompt_token_ids,
                output_token_ids=req.output_token_ids,
                num_computed_tokens=req.num_computed_tokens,
                block_table=req.block_table,
                sampling_params=sampling_params_from_wire_dict(req.sampling_params),
            )
        return _MigrationSessionResults(
            handoff=handoff,
            kv_state=kv_state,
        )


class ShadowEngine:
    """Engine to load model in parallel with KVHTS, then run ShadowExecutor.

    When the KVHTS listener is bound, ``GET /shadow_migration/recv`` on the stdlib
    HTTP server begins returning the socket path (503 until then).
    Tears down after decode finishes.
    """

    def __init__(
        self,
        shadow_kvhts_ipc_prefix: str,
        shadow_config: ShadowConfig,
        *,
        shadow_http_host: str = "127.0.0.1",
        shadow_http_port: int = 8004,
    ) -> None:
        self._shadow_kvhts_ipc_path: str = (
            f"{shadow_kvhts_ipc_prefix}-{secrets.token_hex(8)}"
        )
        self._shadow_config: ShadowConfig = shadow_config
        self._shadow_http_host: str = shadow_http_host
        self._shadow_http_port: int = shadow_http_port

        self._stop_event = threading.Event()

        self._http_kvhts_state = KvhtsHttpState(self._shadow_kvhts_ipc_path)
        self._http_server: ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None

        self._migration_session = _MigrationSession(
            listener_ready_callback=self._on_kvhts_listener_ready,
            shadow_kvhts_ipc_path=self._shadow_kvhts_ipc_path,
        )
        self._model_loader = _ModelLoader(shadow_config)
        self._executor: ShadowExecutor | None = None

    def stop(self) -> None:
        self._stop_event.set()
        self._shutdown_shadow_http()
        self._migration_session.stop()
        if self._executor is not None:
            self._executor.stop()

    def _shutdown_shadow_http(self) -> None:
        srv = self._http_server
        if srv is None:
            return
        self._http_server = None
        with suppress(Exception):
            srv.shutdown()
        thr = self._http_thread
        self._http_thread = None
        if thr is not None and thr.is_alive():
            thr.join(timeout=5.0)

    def run(self) -> None:
        try:
            self._http_server, self._http_thread = start_shadow_http_server(
                self._shadow_http_host,
                self._shadow_http_port,
                self._http_kvhts_state,
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                ms_future = executor.submit(self._migration_session.run)
                model_future = executor.submit(self._model_loader.run)

            if self._stop_event.is_set():
                return

            ms_results = ms_future.result()
            model = model_future.result()
            if self._stop_event.is_set():
                return

            assert ms_results is not None
            handoff, kv_state = (
                ms_results.handoff,
                ms_results.kv_state,
            )
            self._executor = ShadowExecutor(
                migration_id=handoff.migration_id,
                tksth_ipc_path=handoff.tksth_ipc_path,
                model=model,
                kv_state=kv_state,
            )
            logger.info(
                "Running shadow executor: migration_id=%s", handoff.migration_id
            )
            self._executor.run()
        finally:
            self._teardown()

    def _teardown(self) -> None:
        self._shutdown_shadow_http()

    def _on_kvhts_listener_ready(self) -> None:
        """Mark shadow HTTP recv as ready (KVHTS UDS is listening)."""
        self._http_kvhts_state.set_listening()
        logger.info(
            "KVHTS listener ready; GET /shadow_migration/recv returns path %s",
            self._shadow_kvhts_ipc_path,
        )
