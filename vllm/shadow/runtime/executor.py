# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU shadow executor: autoregressive decode + TKCTH."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import suppress

from vllm.shadow.models.kv_state import ShadowKvState
from vllm.shadow.models.llama import LlamaLikeShadowModel
from vllm.shadow.models.sampling import (
    ShadowSamplingParams,
    sample_from_logits,
    sampling_params_from_wire_dict,
    stop_finish_reason,
)
from vllm.shadow.transfer.kvhtc_memfd import MemfdTensor
from vllm.shadow.transfer.kvhtc_protocol import MigrationHandoff
from vllm.shadow.transfer.tkcth_protocol import TkcthFinish, TkcthTokenDelta
from vllm.shadow.transfer.tkcth_uds import UdsTkcthSenderTransport

logger = logging.getLogger("vllm.shadow.runtime.executor")

__all__ = ("ShadowExecutor",)


class ShadowExecutor:
    """CPU-side shadow executor."""

    def __init__(
        self,
        handoff: MigrationHandoff,
        model: LlamaLikeShadowModel,
        layer_kv_tensors: list[MemfdTensor],
    ) -> None:
        migration_id = handoff.migration_id
        num_layers = handoff.num_layers
        num_blocks = handoff.shadow_num_blocks
        num_kv_heads = handoff.num_kv_heads
        head_dim = handoff.head_dim
        block_size = handoff.block_size
        dtype = handoff.dtype
        requests = handoff.requests
        tkcth_ipc_path = handoff.tkcth_ipc_path

        self._migration_id = migration_id
        self._tkcth_ipc_path = tkcth_ipc_path

        if not requests:
            raise ValueError("migration handoff has no requests")
        if len(layer_kv_tensors) != num_layers:
            raise ValueError(
                f"expected {num_layers} layer KV tensors, got {len(layer_kv_tensors)}"
            )

        self._model = model

        kv_state = ShadowKvState(
            num_layers=num_layers,
            num_blocks=num_blocks,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=dtype,
        )
        for layer_idx, t in enumerate(layer_kv_tensors):
            kv_state.register_layer(layer_idx, t)
        for req in requests:
            kv_state.add_request(
                request_id=req.request_id,
                prompt_token_ids=req.prompt_token_ids,
                output_token_ids=req.output_token_ids,
                num_computed_tokens=req.num_computed_tokens,
                block_table=req.dst_block_table,
                sampling_params=sampling_params_from_wire_dict(req.sampling_params),
            )
        self._kv_state = kv_state

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
        except BaseException as e:
            self._executor_error = e
        finally:
            self._teardown()

    def _initialize(self) -> None:
        transport = UdsTkcthSenderTransport()
        transport.connect(self._tkcth_ipc_path)
        self._tkcth_sender = transport

    def _teardown(self) -> None:
        if self._tkcth_sender is not None:
            with suppress(OSError):
                self._tkcth_sender.close()
            self._tkcth_sender = None

    def _stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def _decode_loop(self) -> None:
        if self._stop_requested():
            self._abort_remaining_requests()
            return

        decode_step = 0
        while self._kv_state.has_requests():
            if self._stop_requested():
                self._abort_remaining_requests()
                return

            t_preprocess = time.perf_counter()
            request_ids = self._kv_state.request_ids()
            query_token_ids: list[list[int]] = []
            num_computed: list[int] = []
            sampling_params: list[ShadowSamplingParams] = []
            for rid in request_ids:
                results = self._kv_state.prepare_request(rid)
                if results is None:
                    self._finish_request(rid, "error")
                    logger.warning(
                        "no free shadow KV slots left for block table extension "
                        "(request_id=%r)",
                        rid,
                    )
                    continue
                qtids, nc, sp = results
                if not qtids:
                    self._finish_request(rid, "error")
                    logger.error(
                        "request has no uncomputed tokens (request_id=%r)", rid
                    )
                    continue
                query_token_ids.append(qtids)
                num_computed.append(nc)
                sampling_params.append(sp)
            batch = self._kv_state.build_attention_batch(
                request_ids,
                query_token_ids,
            )
            preprocess_elapsed_ms = (time.perf_counter() - t_preprocess) * 1000.0

            t_forward = time.perf_counter()
            logits = self._model.forward_logits(batch)
            forward_elapsed_ms = (time.perf_counter() - t_forward) * 1000.0

            t_postprocess = time.perf_counter()
            step_output_token_ids: list[int] = []
            for row_idx, rid in enumerate(request_ids):
                qtids = query_token_ids[row_idx]
                nc = num_computed[row_idx]
                sp = sampling_params[row_idx]
                tid = sample_from_logits(logits[row_idx], sp)
                self._kv_state.advance_decoded_token(rid, tid)
                self._emit_single_token_delta(rid, tid)
                step_output_token_ids.append(tid)
                # When finish_reason is not None, the request is finished.
                finish_reason = stop_finish_reason(
                    tid,
                    sp,
                    self._model.eos_token_id,
                    num_output_tokens_after_append=nc + len(qtids),
                )
                if finish_reason is not None:
                    self._finish_request(rid, finish_reason)
            postprocess_elapsed_ms = (time.perf_counter() - t_postprocess) * 1000.0

            logger.info(
                "shadow_cpu_decode migration_id=%s output_token_ids=%s "
                "step=%s batch_size=%d preprocess_elapsed_ms=%.2f "
                "forward_elapsed_ms=%.2f postprocess_elapsed_ms=%.2f",
                self._migration_id,
                step_output_token_ids,
                decode_step,
                len(request_ids),
                preprocess_elapsed_ms,
                forward_elapsed_ms,
                postprocess_elapsed_ms,
            )
            decode_step += 1

    # ---- TKCTH emission ----

    def _emit_single_token_delta(
        self,
        request_id: str,
        token_id: int,
    ) -> None:
        assert self._tkcth_sender is not None
        self._tkcth_sender.send(
            TkcthTokenDelta(
                migration_id=int(self._migration_id),
                request_id=request_id,
                token_ids=[token_id],
            )
        )

    def _finish_request(self, request_id: str, finish_reason: str) -> None:
        assert self._tkcth_sender is not None
        self._tkcth_sender.send(
            TkcthFinish(
                migration_id=int(self._migration_id),
                request_id=request_id,
                finish_reason=finish_reason,
            )
        )
        self._kv_state.free_request(request_id)

    def _abort_remaining_requests(self) -> None:
        """Emit one ``TkcthFinish`` per request still in ``_kv_state``, then clear."""
        for rid in list(self._kv_state.request_ids()):
            self._finish_request(rid, "abort")
