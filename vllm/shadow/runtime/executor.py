# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU shadow executor: autoregressive decode + TKCTH."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import suppress

from vllm.shadow.models.kv_manager import ShadowKvManager
from vllm.shadow.models.llama import LlamaLikeShadowModel
from vllm.shadow.models.sampling import (
    ShadowSamplingParams,
    sample_from_logits,
    sampling_params_from_wire_dict,
    should_stop,
)
from vllm.shadow.transfer.kvhtc_memfd import MemfdTensor
from vllm.shadow.transfer.kvhtc_protocol import MigrationHandoff
from vllm.shadow.transfer.tkcth_protocol import TkcthFinish, TkcthTokenDelta
from vllm.shadow.transfer.tkcth_uds import UdsTkcthSenderTransport

logger = logging.getLogger("vllm.shadow.runtime.executor")

_DECODE_JOIN_TIMEOUT_S = 300.0

__all__ = ("ShadowExecutor",)


class ShadowExecutor:
    """CPU-side shadow executor."""

    def __init__(
        self,
        handoff: MigrationHandoff,
        model: LlamaLikeShadowModel,
        layer_kv_tensors: list[MemfdTensor],
    ) -> None:
        self._handoff = handoff
        if not handoff.requests:
            raise ValueError("migration handoff has no requests")
        if len(layer_kv_tensors) != handoff.num_layers:
            raise ValueError(
                f"expected {handoff.num_layers} layer KV tensors, got "
                f"{len(layer_kv_tensors)}"
            )

        self._sampling_params: list[ShadowSamplingParams] = [
            sampling_params_from_wire_dict(r.sampling_params) for r in handoff.requests
        ]
        self._kv_manager = ShadowKvManager(handoff)
        self._model = model
        for layer_idx, t in enumerate(layer_kv_tensors):
            self._kv_manager.register_layer(layer_idx, t)

        self._tkcth_sender: UdsTkcthSenderTransport | None = None

        self._executor_thread: threading.Thread | None = None
        self._executor_error: BaseException | None = None
        self._stop_event = threading.Event()

    # ── Lifecycle ─────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the decode thread."""
        if self._executor_thread is not None:
            raise RuntimeError("ShadowExecutor.start() already called")
        self._executor_error = None
        self._stop_event.clear()
        self._executor_thread = threading.Thread(
            target=self._run_executor,
            name="shadow-cpu-decode",
            daemon=False,
        )
        self._executor_thread.start()

    def stop(self) -> None:
        """Request shutdown and release blocking resources."""
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        """Wait for executor shutdown and re-raise any background exception."""
        if self._executor_thread is None:
            raise RuntimeError("ShadowExecutor.start() was not called")
        self._executor_thread.join(timeout=timeout)
        if timeout is not None and self._executor_thread.is_alive():
            raise TimeoutError("decode thread did not finish within timeout")
        err = self._executor_error
        if err is not None:
            raise err

    # ── Executor ───────────────────────────────────────────────────────

    def _run_executor(self) -> None:
        try:
            self._initialize()
            self._decode_loop()
            if self._stop_requested():
                return
            self._emit_finishes()
        except BaseException as e:
            self._executor_error = e
        finally:
            self._teardown()

    def _initialize(self) -> None:
        transport = UdsTkcthSenderTransport()
        transport.connect(self._handoff.tkcth_ipc_path)
        self._tkcth_sender = transport

    def _teardown(self) -> None:
        if self._tkcth_sender is not None:
            with suppress(OSError):
                self._tkcth_sender.close()
            self._tkcth_sender = None

    def _max_decode_iterations(self) -> int:
        m = 1
        for sp in self._sampling_params:
            m = max(m, max(0, sp.max_tokens))
        return m

    def _stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def _decode_loop(self) -> None:
        if self._stop_requested():
            return
        num_requests = len(self._handoff.requests)
        request_still_active = [True] * num_requests
        max_steps = self._max_decode_iterations()

        for decode_step in range(max_steps):
            if self._stop_requested():
                return
            if not any(request_still_active):
                break

            request_indices: list[int] = []
            query_token_ids: list[list[int]] = []
            for req_idx in range(num_requests):
                if not request_still_active[req_idx]:
                    continue
                uncomputed_token_ids = self._kv_manager.uncomputed_token_ids(req_idx)
                if not uncomputed_token_ids:
                    raise RuntimeError(
                        "shadow decode: active request has no uncomputed tokens"
                    )
                request_indices.append(req_idx)
                query_token_ids.append(uncomputed_token_ids)
                self._kv_manager.prepare_blocks_for_query(
                    req_idx, len(uncomputed_token_ids)
                )
            batch = self._kv_manager.build_attention_batch(
                request_indices,
                query_token_ids,
            )

            t_forward = time.perf_counter()
            logits = self._model.forward_logits(batch)
            forward_elapsed_ms = (time.perf_counter() - t_forward) * 1000.0

            eos = self._model.eos_token_id
            step_output_token_ids: list[int] = []
            for row_idx, req_idx in enumerate(request_indices):
                sp = self._sampling_params[req_idx]
                tid = sample_from_logits(logits[row_idx], sp)
                out_after = self._kv_manager.num_output_tokens(req_idx) + 1
                stopped = should_stop(
                    tid,
                    sp,
                    eos,
                    num_output_tokens_after_append=out_after,
                )
                step_output_token_ids.append(tid)
                qlen = len(query_token_ids[row_idx])
                handoff_req = self._handoff.requests[req_idx]
                self._kv_manager.append_decoded_token(req_idx, tid)
                self._emit_single_token_delta(handoff_req.request_id, tid)
                self._kv_manager.advance_num_computed(req_idx, qlen)
                if stopped:
                    request_still_active[req_idx] = False

            logger.info(
                "shadow_cpu_decode migration_id=%s output_token_ids=%s "
                "step=%s batch_size=%d forward_elapsed_ms=%.2f",
                self._handoff.migration_id,
                step_output_token_ids,
                decode_step,
                len(request_indices),
                forward_elapsed_ms,
            )

    # ---- TKCTH emission ----

    def _emit_single_token_delta(
        self,
        request_id: str,
        token_id: int,
    ) -> None:
        assert self._tkcth_sender is not None
        self._tkcth_sender.send(
            TkcthTokenDelta(
                migration_id=int(self._handoff.migration_id),
                request_id=request_id,
                token_ids=[token_id],
            )
        )

    def _emit_finishes(self) -> None:
        assert self._tkcth_sender is not None
        mid = int(self._handoff.migration_id)
        for hr in self._handoff.requests:
            self._tkcth_sender.send(
                TkcthFinish(
                    migration_id=mid,
                    request_id=hr.request_id,
                    finish_reason="stop",
                )
            )
