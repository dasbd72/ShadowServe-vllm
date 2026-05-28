# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import threading
import time
from contextlib import suppress

import numpy as np
import ray
import torch
from ray.actor import ActorHandle

from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.shadow.models.loader import ShadowModelLoader
from vllm.shadow.runtime.arg_utils import ShadowEngineArgs
from vllm.shadow.runtime.engine import AsyncShadow
from vllm.shadow.transfer.kv_transport_common import MemfdTensor
from vllm.shadow.transfer.kvhts_memfd import UdsMemfdKvhtsSenderTransport
from vllm.shadow.transfer.kvhts_protocol import KvhtsHandoff, KvhtsRequest
from vllm.shadow.transfer.tksth_protocol import (
    TksthError,
    TksthFinish,
    TksthMessage,
)
from vllm.shadow.transfer.tksth_uds import UdsTksthReceiverTransport

logger = logging.getLogger("scripts/cold_start_interference/run.py")


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_LOAD_FORMAT = "serverless_llm"


def _resolve_model(model: str, load_format: str, model_path: str | None) -> str:
    if model_path is not None:
        return model_path
    if load_format == "serverless_llm":
        storage_path = os.getenv("STORAGE_PATH", os.path.expanduser("~/models"))
        return os.path.join(storage_path, "vllm", model)
    return model


# Fixed DRAM working set (not the experiment knob). Large enough to miss LLC.
_MEM_BW_WORKSET_BYTES = 512 * 1024 * 1024
_MEM_BW_COPY_CHUNK_BYTES = 16 * 1024 * 1024


class MemoryBandwidthWorkload:
    """Synthetic DRAM copy load to verify cold-init interference without shadow.

    ``num_threads`` (from ``--workload-cpus``) controls aggregate memory pressure.
    """

    def __init__(self, num_threads: int) -> None:
        if num_threads <= 0:
            raise ValueError(f"num_threads must be positive, got {num_threads}")

        self.num_threads = num_threads

        self._buffers: list[np.ndarray] = []
        self._chunk_bytes = 0
        self._stop = threading.Event()
        self._workers_thread: threading.Thread | None = None
        self._elapsed = 0.0

    def init(self) -> None:
        num_chunks = max(2, math.ceil(_MEM_BW_WORKSET_BYTES / _MEM_BW_COPY_CHUNK_BYTES))
        self._chunk_bytes = math.ceil(_MEM_BW_WORKSET_BYTES / num_chunks)
        self._buffers = [
            np.empty(self._chunk_bytes, dtype=np.uint8) for _ in range(num_chunks)
        ]
        for buf in self._buffers:
            buf.fill(1)
        workset_gb = sum(buf.nbytes for buf in self._buffers) / (1024**3)
        logger.info(
            "MemoryBandwidthWorkload ready workset_gb=%.2f chunk_mb=%.0f threads=%d",
            workset_gb,
            self._chunk_bytes / (1024**2),
            self.num_threads,
        )

    def start(self) -> None:
        """Spawn copy threads and return (Ray actor stays free for ``stop()``)."""
        if self._workers_thread is not None and self._workers_thread.is_alive():
            raise RuntimeError("MemoryBandwidthWorkload already running")
        if not self._buffers:
            raise RuntimeError("call init() before start()")

        self._stop.clear()
        self._workers_thread = threading.Thread(
            target=self._run_workers,
            name="membw-workers",
            daemon=True,
        )
        self._workers_thread.start()
        logger.info("MemoryBandwidthWorkload.start")

    def stop(self) -> float:
        """Stop copy threads and return elapsed seconds since ``start()``."""
        if self._workers_thread is None:
            return self._elapsed

        self._stop.set()
        self._workers_thread.join()
        self._workers_thread = None
        logger.info("MemoryBandwidthWorkload.stop %.3fs", self._elapsed)
        return self._elapsed

    def _run_workers(self) -> None:
        bytes_moved = 0
        bytes_lock = threading.Lock()

        def worker(tid: int) -> None:
            nonlocal bytes_moved
            n_bufs = len(self._buffers)
            i = tid % n_bufs
            local_bytes = 0
            while not self._stop.is_set():
                src = self._buffers[i % n_bufs]
                dst = self._buffers[(i + 1) % n_bufs]
                np.copyto(dst, src)
                local_bytes += self._chunk_bytes
                i += 1
            with bytes_lock:
                bytes_moved += local_bytes

        t0 = time.perf_counter()
        threads = [
            threading.Thread(target=worker, args=(tid,), name=f"membw-{tid}")
            for tid in range(self.num_threads)
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self._elapsed = time.perf_counter() - t0
        achieved_gibps = (
            (bytes_moved / self._elapsed) / (1024**3) if self._elapsed > 0 else 0.0
        )
        logger.info(
            "MemoryBandwidthWorkload finished %.3fs achieved_gibps=%.2f "
            "moved_gib=%.2f threads=%d",
            self._elapsed,
            achieved_gibps,
            bytes_moved / (1024**3),
            self.num_threads,
        )


class ShadowBackend:
    def __init__(
        self,
        model: str,
        load_format: str,
        model_path: str | None,
        num_requests: int = 4,
        input_len: int = 3,
    ) -> None:
        self.num_requests = num_requests
        self.input_len = input_len

        self.block_size = 16
        self.dtype = "bfloat16"

        resolved_model = _resolve_model(model, load_format, model_path)
        self.engine_args = ShadowEngineArgs(
            model=resolved_model,
            dtype=self.dtype,
            block_size=self.block_size,
            load_format=load_format,
        )
        hf_config = ShadowModelLoader(self.engine_args.get_model_config()).hf_config
        self.num_layers = hf_config.num_hidden_layers
        self.num_kv_heads = hf_config.num_key_value_heads
        self.head_dim = hf_config.head_dim

        self.engine: AsyncShadow | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.start_running = asyncio.Event()

    async def init_backend(self) -> None:
        self.engine = AsyncShadow.from_engine_args(self.engine_args)
        self.loop = asyncio.get_running_loop()

    async def run(self) -> None:
        """KVHTS recv + hot send; shadow decode continues until stopped."""
        assert self.engine is not None

        migration_id = 2
        kvhts_ipc_path = "/tmp/kvhts.sock"
        tksth_ipc_path = "/tmp/tksth.sock"

        self._migration_id = migration_id
        self._kvhts_path = kvhts_ipc_path

        t0 = time.perf_counter()
        await self.engine.shadow_migration_recv(migration_id, kvhts_ipc_path)
        await asyncio.to_thread(self._hot_mock, tksth_ipc_path)
        elapsed = time.perf_counter() - t0
        logger.info("ShadowBackend.run %.3fs", elapsed)
        await self.start_running.wait()

    async def shutdown(self) -> None:
        self.start_running.set()
        engine = self.engine
        self.engine = None
        if engine is not None:
            engine.shutdown()
            logger.info("ShadowBackend.shutdown")

    def _hot_mock(self, tksth_ipc_path: str) -> None:
        kvhts_transport = UdsMemfdKvhtsSenderTransport()
        try:
            kvhts_transport.connect(self._kvhts_path)

            blocks_per_req = math.ceil(self.input_len / self.block_size)
            shadow_num_blocks = (blocks_per_req + 125) * self.num_requests
            prompt_token_ids = [(i % 10_000) + 1 for i in range(self.input_len)]
            requests = []
            for idx in range(self.num_requests):
                block_table = list(
                    range(idx * blocks_per_req, (idx + 1) * blocks_per_req)
                )
                requests.append(
                    KvhtsRequest(
                        request_id=f"req-{idx}",
                        prompt_token_ids=prompt_token_ids,
                        output_token_ids=[self.input_len + 1],
                        num_computed_tokens=self.input_len,
                        block_table=block_table,
                        sampling_params={
                            "temperature": 1.0,
                            "max_tokens": 100000,
                        },
                    )
                )
            handoff = KvhtsHandoff(
                migration_id=self._migration_id,
                num_layers=self.num_layers,
                batch_size=len(requests),
                shadow_num_blocks=shadow_num_blocks,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                block_size=self.block_size,
                dtype=self.dtype,
                requests=requests,
                tksth_ipc_path=tksth_ipc_path,
            )
            kvhts_transport.send_handoff(handoff)

            tksth_transport = UdsTksthReceiverTransport()
            tksth_transport.prepare(tksth_ipc_path)

            for _ in range(int(self.num_layers)):
                layer = MemfdTensor.from_tensor(
                    torch.randn(
                        (
                            2,
                            shadow_num_blocks,
                            self.num_kv_heads,
                            self.block_size,
                            self.head_dim,
                        ),
                        dtype=torch.bfloat16,
                    )
                )
                kvhts_transport.send_layer(layer)
                layer.close()

            try:
                t0 = time.perf_counter()
                tksth_transport.accept_once()
                logger.info(
                    "TKSTH connected migration_id=%d (decode until stop)",
                    self._migration_id,
                )

                pending = {req.request_id for req in requests}
                tokens = 0
                messages: list[TksthMessage] = []
                with suppress(ConnectionError):
                    while pending:
                        msg = tksth_transport.recv()
                        if not self.start_running.is_set():
                            self.loop.call_soon_threadsafe(self.start_running.set)
                        messages.append(msg)
                        if isinstance(msg, TksthFinish):
                            pending.discard(msg.request_id)
                        elif isinstance(msg, TksthError):
                            raise ValueError(f"TKSTH error: {msg}")
                        else:
                            tokens += len(msg.token_ids)
                logger.info(
                    "TKSTH completed migration_id=%d tokens=%d tps=%.3f",
                    self._migration_id,
                    tokens,
                    tokens / (time.perf_counter() - t0),
                )
            finally:
                tksth_transport.close()
        finally:
            kvhts_transport.close()


class VllmBackend:
    """Cold GPU receiver (``AsyncLLMEngine`` + KVSTC recv).

    Ray-wrapped like ``VllmBackend`` in ServerlessLLM.
    """

    def __init__(
        self,
        model: str,
        load_format: str,
        model_path: str | None,
    ) -> None:
        resolved_model = _resolve_model(model, load_format, model_path)
        self.engine_args = AsyncEngineArgs(
            model=resolved_model,
            dtype="bfloat16",
            block_size=16,
            load_format=load_format,
            enable_prefix_caching=True,
            shadow_receiver_enabled=True,
        )
        self.engine: AsyncLLMEngine | None = None

    async def init_backend(self) -> float:
        t0 = time.perf_counter()
        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
        elapsed = time.perf_counter() - t0
        logger.info("VllmBackend.init_backend %.3fs", elapsed)
        return elapsed

    async def shutdown(self) -> None:
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None
            logger.info("VllmBackend.shutdown")


class Context:
    def __init__(
        self,
        model: str,
        load_format: str,
        model_path: str | None,
        workload_cpus: int,
        num_requests: int,
        input_len: int,
    ) -> None:
        self.model = model
        self.load_format = load_format
        self.model_path = model_path
        self.workload_cpus = workload_cpus
        self.num_requests = num_requests
        self.input_len = input_len

        self.cold_actor: ActorHandle[VllmBackend] | None = None
        self.shadow_actor: ActorHandle[ShadowBackend] | None = None
        self.mem_bw_actor: ActorHandle[MemoryBandwidthWorkload] | None = None

    async def run_baseline(self) -> None:
        await self.start_cold()

    async def run_concurrent(self) -> None:
        await self.start_shadow()

        forward_ref = self.shadow_actor.run.remote()
        await self.start_cold()

        assert self.shadow_actor is not None
        await self.shadow_actor.shutdown.remote()
        await forward_ref

    async def run_mem_bw(self) -> None:
        await self.start_mem_bw_workload()

        await self.mem_bw_actor.start.remote()
        await self.start_cold()
        await self.mem_bw_actor.stop.remote()

    async def start_cold(self) -> None:
        ray.remote(VllmBackend).options(
            name="cold",
            num_gpus=1,
            lifetime="detached",
        ).remote(self.model, self.load_format, self.model_path)
        self.cold_actor = ray.get_actor("cold")
        await self.cold_actor.init_backend.remote()

    async def start_shadow(self) -> None:
        ray.remote(ShadowBackend).options(
            name="cpu_workload",
            num_cpus=self.workload_cpus,
            lifetime="detached",
        ).remote(
            self.model,
            self.load_format,
            self.model_path,
            self.num_requests,
            self.input_len,
        )
        self.shadow_actor = ray.get_actor("cpu_workload")
        await self.shadow_actor.init_backend.remote()

    async def start_mem_bw_workload(self) -> None:
        ray.remote(MemoryBandwidthWorkload).options(
            name="mem_bw_workload",
            num_cpus=self.workload_cpus,
            lifetime="detached",
        ).remote(self.workload_cpus)
        self.mem_bw_actor = ray.get_actor("mem_bw_workload")
        await self.mem_bw_actor.init.remote()

    async def shutdown(self) -> None:
        cold_actor = self.cold_actor
        shadow_actor = self.shadow_actor
        mem_bw_actor = self.mem_bw_actor
        self.cold_actor = None
        self.shadow_actor = None
        self.mem_bw_actor = None
        if cold_actor is not None:
            await cold_actor.shutdown.remote()
        if shadow_actor is not None:
            await shadow_actor.shutdown.remote()
        if mem_bw_actor is not None:
            await mem_bw_actor.stop.remote()


async def main_async(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    context = Context(
        args.model,
        args.load_format,
        args.model_path,
        args.workload_cpus,
        args.num_requests,
        args.input_len,
    )

    ray.init(
        address="local",
        ignore_reinit_error=True,
        num_cpus=os.cpu_count() or 1,
        num_gpus=2,
    )
    try:
        if args.mode == "baseline":
            await context.run_baseline()
        elif args.mode == "concurrent":
            await context.run_concurrent()
        elif args.mode == "mem_bw":
            await context.run_mem_bw()
    finally:
        await context.shutdown()
        ray.shutdown()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("baseline", "concurrent", "mem_bw"),
        default="concurrent",
        help=(
            "baseline=cold init only; concurrent=overlap shadow KVHTS; "
            "mem_bw=overlap synthetic DRAM copies"
        ),
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--load-format",
        default=DEFAULT_LOAD_FORMAT,
        help="vLLM load_format (default: serverless_llm)",
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="Override model directory (default: $STORAGE_PATH/vllm/<model>)",
    )
    p.add_argument(
        "--workload-cpus",
        type=int,
        default=64,
        help="Ray CPU allocation and mem_bw copy thread count",
    )
    p.add_argument("--num-requests", type=int, default=4)
    p.add_argument(
        "--input-len",
        type=int,
        default=3,
        help="prompt length per KVHTS request (default: 3, prior hardcoded value)",
    )
    return p.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
