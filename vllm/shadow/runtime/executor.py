# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU shadow executor: autoregressive decode + TKSTH."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import suppress

from vllm.shadow.models.kv_state import ShadowKvState
from vllm.shadow.models.llama import LlamaForCausalLM
from vllm.shadow.models.sampling import (
    Sampler,
    ShadowSamplingParams,
    shadow_sampling_metadata,
    stop_finish_reason,
)
from vllm.shadow.transfer.tksth_protocol import (
    TksthFinish,
    TksthFinishReason,
    TksthTokenDelta,
)
from vllm.shadow.transfer.tksth_uds import UdsTksthSenderTransport

logger = logging.getLogger("vllm.shadow.runtime.executor")

__all__ = ("ShadowExecutor",)


class ShadowExecutor:
    """CPU-side shadow executor."""

    def __init__(
        self,
        migration_id: int,
        tksth_ipc_path: str,
        model: LlamaForCausalLM,
        kv_state: ShadowKvState,
    ) -> None:
        self._migration_id = migration_id
        self._tksth_ipc_path = tksth_ipc_path

        self._model = model
        self._kv_state = kv_state
        self._sampler = Sampler()

        self._tksth_sender: UdsTksthSenderTransport | None = None

        self._stop_event = threading.Event()
        self._migrate_event = threading.Event()

    def stop(self) -> None:
        """Request shutdown and release blocking resources."""
        self._stop_event.set()

    def migrate(self) -> None:
        """Migrate the requests to the shadow executor."""
        self._migrate_event.set()

    def run(self) -> None:
        try:
            self._initialize()
            self._decode_loop()
        finally:
            self._teardown()

    def _initialize(self) -> None:
        transport = UdsTksthSenderTransport()
        transport.connect(self._tksth_ipc_path)
        self._tksth_sender = transport

    def _teardown(self) -> None:
        if self._tksth_sender is not None:
            with suppress(OSError):
                self._tksth_sender.close()
            self._tksth_sender = None

    def _decode_loop(self) -> None:
        decode_step = 0
        while True:
            if self._stop_event.is_set():
                self._abort_requests()
                return
            if self._migrate_event.is_set():
                self._migrate_requests()
                return
            if not self._kv_state.has_requests():
                logger.warning("no requests to process, stopping shadow executor.")
                return
            if not self._kv_state.has_free_blocks():
                logger.warning(
                    "no free shadow KV slots left for block table extension, "
                    "stopping shadow executor."
                )
                self._migrate_requests()
                return

            t_preprocess = time.perf_counter()
            all_request_ids = self._kv_state.request_ids()
            request_ids: list[str] = []
            query_token_ids: list[list[int]] = []
            num_computed: list[int] = []
            sampling_params: list[ShadowSamplingParams] = []
            for rid in all_request_ids:
                tids, nc, sp, alloc = self._kv_state.prepare_request(rid)
                if not tids:
                    raise RuntimeError(f"request {rid!r} has no uncomputed tokens")
                if not alloc:
                    logger.warning(
                        "no free shadow KV slots left for block table extension."
                    )
                    return
                request_ids.append(rid)
                query_token_ids.append(tids)
                num_computed.append(nc)
                sampling_params.append(sp)
            if not request_ids:
                logger.warning("no requests to process")
                continue
            batch = self._kv_state.build_attention_batch(
                request_ids,
                query_token_ids,
            )
            preprocess_elapsed_ms = (time.perf_counter() - t_preprocess) * 1000.0

            t_forward = time.perf_counter()
            logits = self._model(batch)
            forward_elapsed_ms = (time.perf_counter() - t_forward) * 1000.0

            t_postprocess = time.perf_counter()
            meta = shadow_sampling_metadata(
                sampling_params,
                vocab=logits.shape[-1],
                device=logits.device,
                generators={},
            )
            step_output_token_ids = self._sampler(logits, meta).tolist()
            for row_idx, rid in enumerate(request_ids):
                qtids = query_token_ids[row_idx]
                nc = num_computed[row_idx]
                sp = sampling_params[row_idx]
                tid = step_output_token_ids[row_idx]
                self._kv_state.advance_decoded_token(rid, tid)
                self._emit_single_token_delta(rid, tid)
                # When finish_reason is not None, the request is finished.
                finish_reason = stop_finish_reason(
                    tid,
                    sp,
                    self._model.eos_token_id,
                    num_output_tokens_after_append=nc + len(qtids),
                )
                if finish_reason is not None:
                    self._finish_request(rid, finish_reason, free_request=True)
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

    # ---- TKSTH emission ----

    def _emit_single_token_delta(
        self,
        request_id: str,
        token_id: int,
    ) -> None:
        assert self._tksth_sender is not None
        self._tksth_sender.send(
            TksthTokenDelta(
                migration_id=int(self._migration_id),
                request_id=request_id,
                token_ids=[token_id],
            )
        )

    def _finish_request(
        self,
        request_id: str,
        finish_reason: TksthFinishReason,
        free_request: bool = False,
    ) -> None:
        assert self._tksth_sender is not None
        self._tksth_sender.send(
            TksthFinish(
                migration_id=int(self._migration_id),
                request_id=request_id,
                finish_reason=finish_reason,
            )
        )
        if free_request:
            self._kv_state.free_request(request_id)

    def _abort_requests(self) -> None:
        """Emit one ``TksthFinish`` per request still in ``_kv_state``, then clear."""
        for rid in list(self._kv_state.request_ids()):
            self._finish_request(rid, "abort", free_request=True)

    def _migrate_requests(self) -> None:
        """Migrate the requests to the shadow executor.

        The requests are not freed from the KV state, so they can be resumed later.
        """
        for rid in list(self._kv_state.request_ids()):
            self._finish_request(rid, "migrated", free_request=False)
