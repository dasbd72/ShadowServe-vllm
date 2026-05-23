# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import enum
import logging
import math
import os
import queue
import signal
import threading
import time
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import msgspec
import torch
import zmq

from vllm.shadow.models.kv_state import ShadowKvState
from vllm.shadow.models.registry import load_shadow_model_from_config
from vllm.shadow.models.sampling import (
    Sampler,
    ShadowSamplingParams,
    sampling_params_from_wire_dict,
    shadow_sampling_metadata,
    stop_finish_reason,
)
from vllm.shadow.runtime.arg_utils import ShadowEngineArgs
from vllm.shadow.runtime.shadow_kvhts_session import ShadowKvhtsSession
from vllm.shadow.runtime.utils import (
    HANDSHAKE_TIMEOUT_MINS,
    ShadowEngineHandshakeMetadata,
    ShadowEngineZmqAddresses,
    decode_msgpack_zmq_frames,
    make_zmq_socket,
)
from vllm.shadow.transfer.kv_transport_common import (
    MemfdTensor,
    dtype_str_from_torch,
)
from vllm.shadow.transfer.kvstc_memfd import UdsMemfdKvstcSenderTransport
from vllm.shadow.transfer.kvstc_protocol import KvstcHandoff, KvstcRequest
from vllm.shadow.transfer.tksth_protocol import (
    TksthError,
    TksthFinish,
    TksthTokenDelta,
)
from vllm.shadow.transfer.tksth_uds import UdsTksthSenderTransport

logger = logging.getLogger("vllm.shadow.runtime.core")

ENGINE_CORE_IDENTITY = (0).to_bytes(2, "little")

# JIT-compile CPU attention + sampling on startup (no live migration state).
_WARMUP_STEPS = 10
_WARMUP_PROMPT_LEN = 2


class ShadowEngineCoreRequestType(enum.Enum):
    UTILITY = b"\x00"
    SHUTDOWN = b"\x01"


class ShadowEngineCoreOutput:
    def __init__(
        self,
        request_id: str,
        token_ids: list[int],
        finish_reason: str | None = None,
        error_message: str | None = None,
    ):
        self.request_id = request_id
        self.token_ids = token_ids
        self.finish_reason = finish_reason
        self.error_message = error_message


class ShadowEngineCoreOutputs:
    def __init__(
        self,
        outputs: list[ShadowEngineCoreOutput],
        request_ids: list[str],
        kv_cache_usage: float,
        num_generation_tokens: int,
        preprocess_elapsed: float,
        forward_elapsed: float,
        postprocess_elapsed: float,
        utility_output: dict[str, Any] | None = None,
    ):
        self.outputs = outputs
        self.request_ids = request_ids
        self.kv_cache_usage = kv_cache_usage
        self.num_generation_tokens = num_generation_tokens
        self.preprocess_elapsed = preprocess_elapsed
        self.forward_elapsed = forward_elapsed
        self.postprocess_elapsed = postprocess_elapsed
        self.utility_output = utility_output

    def encode(self) -> list[bytes]:
        payload = {
            "outputs": [
                {
                    "request_id": o.request_id,
                    "token_ids": o.token_ids,
                    "finish_reason": o.finish_reason,
                    "error_message": o.error_message,
                }
                for o in self.outputs
            ],
            "request_ids": self.request_ids,
            "kv_cache_usage": self.kv_cache_usage,
            "num_generation_tokens": self.num_generation_tokens,
            "preprocess_elapsed": self.preprocess_elapsed,
            "forward_elapsed": self.forward_elapsed,
            "postprocess_elapsed": self.postprocess_elapsed,
            "utility_output": self.utility_output,
        }
        return [msgspec.msgpack.encode(payload)]

    @classmethod
    def decode(
        cls, frames: Sequence[bytes | memoryview | zmq.Frame]
    ) -> ShadowEngineCoreOutputs:
        data = frames[0]
        if isinstance(data, zmq.Frame):
            data = data.buffer
        payload = msgspec.msgpack.decode(data)
        if payload.get("utility_output") is not None:
            return cls(
                outputs=[],
                request_ids=[],
                kv_cache_usage=0.0,
                num_generation_tokens=0,
                preprocess_elapsed=0.0,
                forward_elapsed=0.0,
                postprocess_elapsed=0.0,
                utility_output=payload["utility_output"],
            )
        return cls(
            outputs=[
                ShadowEngineCoreOutput(
                    request_id=o["request_id"],
                    token_ids=o["token_ids"],
                    finish_reason=o.get("finish_reason"),
                    error_message=o.get("error_message"),
                )
                for o in payload["outputs"]
            ],
            request_ids=payload["request_ids"],
            kv_cache_usage=payload["kv_cache_usage"],
            num_generation_tokens=payload["num_generation_tokens"],
            preprocess_elapsed=payload["preprocess_elapsed"],
            forward_elapsed=payload["forward_elapsed"],
            postprocess_elapsed=payload["postprocess_elapsed"],
        )


@dataclass(slots=True)
class MigrationMeta:
    migration_id: int
    tksth_sender: UdsTksthSenderTransport


@dataclass(slots=True)
class ExecutionMeta:
    migration_id: int
    kv_state: ShadowKvState
    tksth_sender: UdsTksthSenderTransport
    waiting_for_migration: bool = False


class ShadowEngineCore:
    """Shadow engine core that handles the model execution and KV state management."""

    def __init__(self, engine_args: ShadowEngineArgs):
        omp_num_threads = os.environ.get("OMP_NUM_THREADS")
        if omp_num_threads is not None:
            logger.info("Setting torch threads to %d", int(omp_num_threads))
            try:
                torch.set_num_threads(int(omp_num_threads))
            except ValueError:
                logger.warning(
                    "Ignoring invalid OMP_NUM_THREADS=%r for torch.set_num_threads",
                    omp_num_threads,
                )

        self.engine_args = engine_args

        self.model = load_shadow_model_from_config(self.engine_args.get_model_config())
        self.sampler = Sampler()

        self._shadow_kvhts_sessions: list[ShadowKvhtsSession] = []
        self._completed_migration_ids: list[int] = []
        self._execution_meta: ExecutionMeta | None = None

        self._warmup_on_init()

    @classmethod
    def from_engine_args(cls, engine_args: ShadowEngineArgs) -> ShadowEngineCore:
        """Construct a core from parsed CLI/launcher arguments."""
        return cls(engine_args=engine_args)

    @torch.inference_mode()
    def _warmup_on_init(self) -> None:
        """Run dummy forwards to JIT-compile CPU kernels before KVHTS traffic."""
        model_cfg = self.engine_args.get_model_config()
        block_size = model_cfg.block_size
        dtype = model_cfg.dtype
        assert isinstance(dtype, torch.dtype)
        dtype_str = dtype_str_from_torch(dtype)

        hf = self.model.hf
        num_layers = int(hf.num_hidden_layers)
        num_kv_heads = int(hf.num_key_value_heads)
        head_dim = int(hf.head_dim)

        total_tokens = _WARMUP_PROMPT_LEN + _WARMUP_STEPS
        blocks_per_req = math.ceil(total_tokens / block_size)
        shadow_num_blocks = blocks_per_req

        kv_state = ShadowKvState(
            num_layers=num_layers,
            num_blocks=shadow_num_blocks,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=dtype_str,
        )

        layer_shape = (2, shadow_num_blocks, num_kv_heads, block_size, head_dim)
        layer_tensors: list[MemfdTensor] = []
        request_id = "_warmup_0"
        try:
            for layer_idx in range(num_layers):
                layer_tensor = torch.zeros(layer_shape, dtype=dtype)
                memfd = MemfdTensor.from_tensor(layer_tensor)
                layer_tensors.append(memfd)
                kv_state.register_layer(layer_idx, memfd)

            warmup_sampling = ShadowSamplingParams(
                temperature=0.9,
                top_k=50,
                top_p=0.9,
                stop_token_ids=[],
                max_tokens=total_tokens,
            )
            kv_state.add_request(
                request_id=request_id,
                prompt_token_ids=list(range(_WARMUP_PROMPT_LEN)),
                output_token_ids=[],
                num_computed_tokens=0,
                block_table=list(range(blocks_per_req)),
                sampling_params=warmup_sampling,
            )

            t0 = time.perf_counter()
            for _ in range(_WARMUP_STEPS):
                token_ids, _, sampling_params, allocated = kv_state.prepare_request(
                    request_id
                )
                if not token_ids or not allocated:
                    break
                batch = kv_state.build_attention_batch(
                    [request_id],
                    [token_ids],
                )
                logits = self.model(batch)
                meta = shadow_sampling_metadata(
                    [sampling_params],
                    vocab=logits.shape[-1],
                    device=logits.device,
                    generators={},
                )
                next_token = self.sampler(logits, meta).tolist()[0]
                del meta
                kv_state.advance_decoded_token(request_id, next_token)

            logger.info(
                "shadow engine warmup completed in %.3fs (%d steps)",
                time.perf_counter() - t0,
                _WARMUP_STEPS,
            )
        finally:
            kv_state.remove_request(request_id)
            for memfd in layer_tensors:
                memfd.close()

    @torch.inference_mode()
    def step(self) -> ShadowEngineCoreOutputs | None:
        if not self._execution_meta or self._execution_meta.waiting_for_migration:
            return None

        kv_state = self._execution_meta.kv_state

        t_preprocess = time.perf_counter()
        errors: list[ShadowEngineCoreOutput] = []
        request_ids: list[str] = []
        query_token_ids: list[list[int]] = []
        num_computed: list[int] = []
        sampling_params: list[ShadowSamplingParams] = []
        for rid in kv_state.requests:
            tids, nc, sp, alloc = kv_state.prepare_request(rid)
            if not tids:
                errors.append(
                    ShadowEngineCoreOutput(
                        request_id=rid,
                        token_ids=[],
                        error_message="no uncomputed tokens",
                    )
                )
                continue
            if not alloc:
                self._execution_meta.waiting_for_migration = True
                logger.warning("no free KV slots left for block table extension.")
                break
            request_ids.append(rid)
            query_token_ids.append(tids)
            num_computed.append(nc)
            sampling_params.append(sp)

        if not request_ids:
            return ShadowEngineCoreOutputs(
                outputs=errors,
                request_ids=request_ids,
                kv_cache_usage=0.0,
                num_generation_tokens=0,
                preprocess_elapsed=0.0,
                forward_elapsed=0.0,
                postprocess_elapsed=0.0,
            )

        batch = kv_state.build_attention_batch(
            request_ids,
            query_token_ids,
        )
        preprocess_elapsed = time.perf_counter() - t_preprocess

        t_forward = time.perf_counter()
        logits = self.model(batch)
        forward_elapsed = time.perf_counter() - t_forward

        t_postprocess = time.perf_counter()
        outputs: list[ShadowEngineCoreOutput] = []
        meta = shadow_sampling_metadata(
            sampling_params,
            vocab=logits.shape[-1],
            device=logits.device,
            generators={},
        )
        step_output_token_ids = self.sampler(logits, meta).tolist()
        for row_idx, rid in enumerate(request_ids):
            qtids = query_token_ids[row_idx]
            nc = num_computed[row_idx]
            sp = sampling_params[row_idx]
            tid = step_output_token_ids[row_idx]
            kv_state.advance_decoded_token(rid, tid)
            finish_reason = stop_finish_reason(
                tid,
                sp,
                self.model.eos_token_id,
                num_output_tokens_after_append=nc + len(qtids),
            )
            outputs.append(
                ShadowEngineCoreOutput(
                    request_id=rid, token_ids=[tid], finish_reason=finish_reason
                )
            )
        postprocess_elapsed = time.perf_counter() - t_postprocess

        return ShadowEngineCoreOutputs(
            outputs=outputs + errors,
            request_ids=request_ids,
            kv_cache_usage=kv_state.kv_cache_usage(),
            num_generation_tokens=len(step_output_token_ids),
            preprocess_elapsed=preprocess_elapsed,
            forward_elapsed=forward_elapsed,
            postprocess_elapsed=postprocess_elapsed,
        )

    def shutdown(self):
        for session in self._shadow_kvhts_sessions:
            session.cancel()
        for session in self._shadow_kvhts_sessions:
            session.join(timeout=2.0)
        self._shadow_kvhts_sessions.clear()
        self._completed_migration_ids.clear()
        if self._execution_meta is not None:
            self._execution_meta.tksth_sender.close()
            self._execution_meta = None

    def shadow_migration_recv(self, migration_id: int, kvhts_ipc_path: str) -> None:
        """Create a KVHTS receiver session and start accepting on ``kvhts_ipc_path``."""
        if any(
            session.migration_id == migration_id
            for session in self._shadow_kvhts_sessions
        ) or (
            self._execution_meta is not None
            and self._execution_meta.migration_id == migration_id
        ):
            logger.warning(
                "kvhts receiver: migration_id=%d already exists",
                migration_id,
            )
            return
        logger.info(
            "kvhts receiver starting migration_id=%d path=%s",
            migration_id,
            kvhts_ipc_path,
        )
        session = ShadowKvhtsSession(
            migration_id=migration_id,
            kvhts_ipc_path=kvhts_ipc_path,
        )
        self._shadow_kvhts_sessions.append(session)

    def shadow_migration_migrate(
        self, migration_id: int, kvstc_ipc_path: str
    ) -> list[str]:
        """Migrate requests to cold GPU."""

        if (
            self._execution_meta is None
            or self._execution_meta.migration_id != migration_id
        ):
            logger.warning(
                "kvstc migration: migration_id=%d not found",
                migration_id,
            )
            return []

        kv_state = self._execution_meta.kv_state

        logger.info(
            "Running kvstc migration: migration_id=%s kvstc_ipc_path=%s",
            migration_id,
            kvstc_ipc_path,
        )

        t0 = time.perf_counter()
        sender = UdsMemfdKvstcSenderTransport()
        sender.connect(kvstc_ipc_path)
        request_ids = [key for key in kv_state.requests]
        requests = [
            KvstcRequest(
                request_id=request_id,
                token_ids=kv_state.requests[request_id].prompt_token_ids
                + kv_state.requests[request_id].output_token_ids,
                block_table=kv_state.requests[request_id].block_table,
            )
            for request_id in request_ids
        ]
        handoff = KvstcHandoff(
            migration_id=migration_id,
            num_layers=kv_state.num_layers,
            batch_size=len(request_ids),
            shadow_num_blocks=kv_state.num_blocks,
            num_kv_heads=self.model.hf.num_key_value_heads,
            head_dim=self.model.hf.head_dim,
            block_size=kv_state.block_size,
            dtype=dtype_str_from_torch(kv_state.expected_dtype),
            requests=requests,
        )
        sender.send_handoff(handoff)
        for layer in kv_state.layers:
            assert layer is not None
            sender.send_layer(layer)
        sender.close()

        logger.info(
            "kvstc migration session done: migration_id=%s requests=%d "
            "layers=%d shadow_blocks=%d bytes≈%d time_s=%.3f",
            migration_id,
            len(requests),
            handoff.num_layers,
            handoff.shadow_num_blocks,
            handoff.num_layers
            * handoff.shadow_num_blocks
            * handoff.num_kv_heads
            * handoff.block_size
            * handoff.head_dim,
            time.perf_counter() - t0,
        )

        for request_id in request_ids:
            self._execution_meta.tksth_sender.send(
                TksthFinish(
                    migration_id=migration_id,
                    request_id=request_id,
                    finish_reason="migrated",
                )
            )
        self._execution_meta.tksth_sender.close()
        self._execution_meta = None

        return request_ids

    def shadow_migration_completed(self) -> dict[str, list[int]]:
        return {
            "completed_kvhts_sessions": self._completed_migration_ids,
        }

    def _process_shadow_kvhts_results(self) -> None:
        still_active: list[ShadowKvhtsSession] = []
        for session in self._shadow_kvhts_sessions:
            if not session.is_done or self._execution_meta is not None:
                still_active.append(session)
                continue
            if session.handoff is None or session.layers is None:
                logger.warning(
                    "kvhts session migration_id=%d finished without handoff",
                    session.migration_id,
                )
                self._completed_migration_ids.append(session.migration_id)
                continue
            migration_id = session.migration_id
            kv_state: ShadowKvState | None = None
            try:
                kv_state = ShadowKvState(
                    num_layers=session.handoff.num_layers,
                    num_blocks=session.handoff.shadow_num_blocks,
                    num_kv_heads=session.handoff.num_kv_heads,
                    head_dim=session.handoff.head_dim,
                    block_size=session.handoff.block_size,
                    dtype=session.handoff.dtype,
                )
                for layer_idx, tensor in enumerate(session.layers):
                    kv_state.register_layer(layer_idx=layer_idx, tensor=tensor)
                for req in session.handoff.requests:
                    kv_state.add_request(
                        request_id=req.request_id,
                        prompt_token_ids=req.prompt_token_ids,
                        output_token_ids=req.output_token_ids,
                        num_computed_tokens=req.num_computed_tokens,
                        block_table=req.block_table,
                        sampling_params=sampling_params_from_wire_dict(
                            req.sampling_params
                        ),
                    )
                tksth_sender = UdsTksthSenderTransport()
                tksth_sender.connect(session.handoff.tksth_ipc_path)
                logger.info(
                    "kvhts session migration_id=%d ingested %d requests",
                    migration_id,
                    len(session.handoff.requests),
                )

                self._execution_meta = ExecutionMeta(
                    migration_id=migration_id,
                    kv_state=kv_state,
                    tksth_sender=tksth_sender,
                )
                self._completed_migration_ids.append(session.migration_id)
            except Exception:
                logger.exception(
                    "failed to ingest kvhts session migration_id=%d",
                    migration_id,
                )
        self._shadow_kvhts_sessions = still_active

    def _shadow_tksth_send_outputs(self, outputs: ShadowEngineCoreOutputs) -> None:
        for output in outputs.outputs:
            assert self._execution_meta is not None
            migration_id = self._execution_meta.migration_id
            kv_state = self._execution_meta.kv_state
            tksth_sender = self._execution_meta.tksth_sender
            if output.error_message is not None:
                tksth_sender.send(
                    TksthError(
                        migration_id=migration_id,
                        message=output.error_message,
                        request_id=output.request_id,
                    )
                )
            else:
                tksth_sender.send(
                    TksthTokenDelta(
                        migration_id=migration_id,
                        request_id=output.request_id,
                        token_ids=output.token_ids,
                    )
                )
            if output.finish_reason is not None:
                tksth_sender.send(
                    TksthFinish(
                        migration_id=migration_id,
                        request_id=output.request_id,
                        finish_reason=output.finish_reason,
                    )
                )
            if output.finish_reason is not None or output.error_message is not None:
                kv_state.remove_request(output.request_id)
                if not kv_state.requests:
                    self._execution_meta.tksth_sender.close()
                    self._execution_meta = None


class ShadowEngineCoreProc(ShadowEngineCore):
    """ZMQ-wrapper for running ShadowEngineCore in background process."""

    ENGINE_CORE_DEAD = b"SHADOW_ENGINE_CORE_DEAD"

    def __init__(
        self,
        engine_args: ShadowEngineArgs,
        handshake_address: str,
    ):
        self.input_queue = queue.Queue[tuple[ShadowEngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[tuple[int, ShadowEngineCoreOutputs] | bytes]()

        self.process_input_queue_block = True
        self.addresses = self._startup_handshake(handshake_address)
        super().__init__(engine_args)

        ready_event = threading.Event()
        self.input_thread = threading.Thread(
            target=self.process_input_socket,
            args=(self.addresses.input_address, ready_event),
            daemon=True,
        )
        self.input_thread.start()

        self.output_thread = threading.Thread(
            target=self.process_output_socket,
            args=(self.addresses.output_address,),
            daemon=True,
        )
        self.output_thread.start()

        while not ready_event.wait(timeout=10):
            if not self.input_thread.is_alive():
                raise RuntimeError(
                    "Shadow engine input socket thread died during startup"
                )

    @staticmethod
    def _startup_handshake(handshake_address: str) -> ShadowEngineZmqAddresses:
        with ExitStack() as stack, zmq.Context() as ctx:
            handshake_socket = stack.enter_context(
                make_zmq_socket(
                    ctx,
                    handshake_address,
                    zmq.DEALER,
                    identity=ENGINE_CORE_IDENTITY,
                    linger=5000,
                    bind=False,
                )
            )
            handshake_socket.send(msgspec.msgpack.encode({"status": "HELLO"}))
            if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):
                raise RuntimeError(
                    "Did not receive init message from shadow engine client "
                    f"within {HANDSHAKE_TIMEOUT_MINS} minutes"
                )
            init_bytes = handshake_socket.recv()
            init_message = msgspec.msgpack.decode(
                init_bytes, type=ShadowEngineHandshakeMetadata
            )
            addresses = init_message.addresses
            handshake_socket.send(msgspec.msgpack.encode({"status": "READY"}))
            return addresses

    @staticmethod
    def run_engine_core(*args, **kwargs):
        """Launch ShadowEngineCore busy loop in background process."""
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core: ShadowEngineCoreProc | None = None
        try:
            engine_core = ShadowEngineCoreProc(*args, **kwargs)
            engine_core.run_busy_loop()
        except SystemExit:
            logger.debug("ShadowEngineCore exiting.")
            raise
        except Exception:
            if engine_core is None:
                logger.exception("ShadowEngineCore failed to start.")
            else:
                logger.exception("ShadowEngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise
        finally:
            if engine_core is not None:
                engine_core.shutdown()

    def has_work(self) -> bool:
        return bool(self._shadow_kvhts_sessions) or (
            self._execution_meta is not None
            and not self._execution_meta.waiting_for_migration
        )

    def run_busy_loop(self):
        """Core busy loop of the ShadowEngineCore."""
        while True:
            self._process_input_queue()
            self._process_engine_step()

    def _process_input_queue(self):
        while not self.has_work():
            if self.input_queue.empty() and logger.isEnabledFor(logging.DEBUG):
                logger.debug("ShadowEngineCore waiting for work.")
            block = self.process_input_queue_block
            try:
                req = self.input_queue.get(block=block)
                self._handle_client_request(*req)
            except queue.Empty:
                break
            if not block:
                break

        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _process_engine_step(self):
        self._process_shadow_kvhts_results()
        outputs = self.step()
        if outputs is not None:
            self.output_queue.put_nowait((0, outputs))
            self._shadow_tksth_send_outputs(outputs)

    def _handle_client_request(
        self, request_type: ShadowEngineCoreRequestType, request: Any
    ) -> None:
        if request_type == ShadowEngineCoreRequestType.SHUTDOWN:
            raise SystemExit()
        if request_type == ShadowEngineCoreRequestType.UTILITY:
            call_id, method_name, args = request
            output: dict[str, Any] = {"call_id": call_id}
            try:
                method = getattr(self, method_name)
                output["result"] = method(*args)
            except Exception as e:
                logger.exception("Invocation of %s method failed", method_name)
                output["failure_message"] = f"Call to {method_name} method failed: {e}"
            self.output_queue.put_nowait(
                (
                    0,
                    ShadowEngineCoreOutputs(
                        outputs=[],
                        request_ids=[],
                        kv_cache_usage=0.0,
                        num_generation_tokens=0,
                        preprocess_elapsed=0.0,
                        forward_elapsed=0.0,
                        postprocess_elapsed=0.0,
                        utility_output=output,
                    ),
                )
            )
            return
        logger.error("Unrecognized input request type encountered: %s", request_type)

    def _send_engine_dead(self):
        self.output_queue.put_nowait(self.ENGINE_CORE_DEAD)
        self.output_thread.join(timeout=5.0)

    def process_input_socket(self, input_address: str, ready_event: threading.Event):
        generic_decoder = msgspec.msgpack.Decoder()
        with ExitStack() as stack, zmq.Context() as ctx:
            input_socket = stack.enter_context(
                make_zmq_socket(
                    ctx,
                    input_address,
                    zmq.DEALER,
                    identity=ENGINE_CORE_IDENTITY,
                    bind=False,
                )
            )
            input_socket.send(b"")
            ready_event.set()
            while True:
                type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                request_type = ShadowEngineCoreRequestType(bytes(type_frame.buffer))
                if request_type == ShadowEngineCoreRequestType.UTILITY:
                    request = decode_msgpack_zmq_frames(generic_decoder, data_frames)
                else:
                    request = None
                self.input_queue.put_nowait((request_type, request))

    def process_output_socket(self, output_address: str):
        with ExitStack() as stack, zmq.Context() as ctx:
            output_socket = stack.enter_context(
                make_zmq_socket(ctx, output_address, zmq.PUSH, linger=4000)
            )
            while True:
                output = self.output_queue.get()
                if output == self.ENGINE_CORE_DEAD:
                    output_socket.send(output)
                    break
                assert not isinstance(output, bytes)
                _client_index, outputs = output
                output_socket.send_multipart(outputs.encode(), copy=False)
