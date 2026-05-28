#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from vllm.shadow.models.kv_state import ShadowKvState
from vllm.shadow.models.registry import load_shadow_model_from_config
from vllm.shadow.models.sampling import (
    Sampler,
    ShadowSamplingParams,
    shadow_sampling_metadata,
)
from vllm.shadow.runtime.config import ShadowModelConfig
from vllm.shadow.transfer.kv_transport_common import MemfdTensor, torch_dtype_from_str


@dataclass(slots=True)
class ShadowDecodeBenchmarkResult:
    latencies: list[float]
    avg_latency: float
    percentages: list[float]
    percentiles: list[float]

    def to_json(self) -> dict[str, Any]:
        return {
            "avg_latency": self.avg_latency,
            "latencies": self.latencies,
            "percentiles": dict(zip(self.percentages, self.percentiles.tolist())),
        }


def _alloc_memfd_tensor(shape: tuple[int, ...], dtype: torch.dtype) -> MemfdTensor:
    # Use a real memfd so `MemfdTensor.close()` works correctly.
    import mmap
    import os

    nbytes = int(torch.empty((), dtype=dtype).element_size() * math.prod(shape))
    flags = getattr(os, "MFD_CLOEXEC", 0)
    fd = os.memfd_create("vllm_shadow_bench_kv", flags)
    mm: mmap.mmap | None = None
    try:
        os.ftruncate(fd, nbytes)
        mm = mmap.mmap(fd, nbytes, mmap.MAP_SHARED, mmap.PROT_WRITE)
        t = torch.frombuffer(mm, dtype=dtype, count=math.prod(shape)).reshape(shape)
        t.zero_()
        return MemfdTensor(mm=mm, fd=fd, tensor=t)
    except BaseException:
        if mm is not None:
            mm.close()
        os.close(fd)
        raise


def benchmark_shadow_latency(
    *,
    output_len: int,
    batch_size: int,
    input_len: int,
    model: str = "Qwen/Qwen3-0.6B",
    dtype: str = "bfloat16",
    block_size: int = 16,
    num_iters_warmup: int = 10,
    num_iters: int = 30,
) -> ShadowDecodeBenchmarkResult:
    """Match ``vllm/benchmarks/latency.py`` semantics for a shadow CPU model.

    Each timed sample is wall-clock seconds for **one full batch**: prefill
    ``batch_size`` prompts of length ``input_len`` and generate ``output_len``
    new tokens per request (same idea as ``LLM.generate`` with
    ``max_tokens=output_len``).

    Dummy prompts are drawn once (same as the GPU benchmark) and reused for
    every warmup and benchmark iteration; state is reset between iterations.
    """
    if output_len <= 0:
        raise ValueError("output_len must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if input_len <= 0:
        raise ValueError("input_len must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if num_iters_warmup < 0:
        raise ValueError("num_iters_warmup must be >= 0")
    if num_iters <= 0:
        raise ValueError("num_iters must be positive")

    num_threads = int(torch.get_num_threads())
    print(f"Number of threads: {num_threads}")
    torch.set_num_threads(num_threads)

    shadow_cfg = ShadowModelConfig(model=model, dtype=dtype, block_size=block_size)
    model_obj = load_shadow_model_from_config(shadow_cfg)

    dummy_prompt_token_ids = np.random.randint(
        10_000, size=(batch_size, input_len), dtype=np.int64
    )

    # KV cache sizing: prompt + up to `output_len` decoded tokens.
    total_tokens_per_req = input_len + output_len
    blocks_per_req = math.ceil(total_tokens_per_req / block_size)
    shadow_num_blocks = batch_size * blocks_per_req

    kv_state = ShadowKvState(
        num_layers=int(model_obj.hf.num_hidden_layers),
        num_blocks=shadow_num_blocks,
        num_kv_heads=int(model_obj.hf.num_key_value_heads),
        head_dim=int(model_obj.hf.head_dim),
        block_size=block_size,
        dtype=dtype,
    )

    layer_shape = (
        2,
        shadow_num_blocks,
        int(model_obj.hf.num_key_value_heads),
        block_size,
        int(model_obj.hf.head_dim),
    )
    layer_tensors: list[MemfdTensor] = []
    try:
        for li in range(int(model_obj.hf.num_hidden_layers)):
            mt = _alloc_memfd_tensor(layer_shape, torch_dtype_from_str(dtype))
            layer_tensors.append(mt)
            kv_state.register_layer(li, mt)

        # Align sampling with ``vllm/benchmarks/latency.py`` defaults.
        bench_sampling = ShadowSamplingParams(
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            stop_token_ids=[],
            max_tokens=output_len,
        )
        bench_sampler = Sampler()

        def _reset_batch() -> None:
            request_ids = list(kv_state.requests.keys())
            for rid in request_ids:
                kv_state.free_request(rid)
            for mt in layer_tensors:
                mt.tensor.zero_()
            for i in range(batch_size):
                rid = f"bench-{i}"
                tids = dummy_prompt_token_ids[i].tolist()
                bt = list(range(i * blocks_per_req, (i + 1) * blocks_per_req))
                kv_state.add_request(
                    request_id=rid,
                    prompt_token_ids=tids,
                    output_token_ids=[],
                    num_computed_tokens=0,
                    block_table=bt,
                    sampling_params=bench_sampling,
                )

        def _one_forward() -> None:
            request_ids = list(kv_state.requests.keys())
            qtids: list[list[int]] = []
            num_computed: list[int] = []
            sampling_params: list[ShadowSamplingParams] = []
            for rid in request_ids:
                tids, nc, sp, alloc = kv_state.prepare_request(rid)
                if not alloc:
                    raise RuntimeError("KV blocks exhausted during benchmark")
                qtids.append(tids)
                num_computed.append(nc)
                sampling_params.append(sp)
            b = kv_state.build_attention_batch(request_ids, qtids)
            logits = model_obj(b)
            meta = shadow_sampling_metadata(
                sampling_params,
                vocab=logits.shape[-1],
                device=logits.device,
                generators={},
            )
            token_ids = bench_sampler(logits, meta).tolist()
            for row_idx, rid in enumerate(request_ids):
                kv_state.advance_decoded_token(rid, token_ids[row_idx])

        def run_to_completion() -> float:
            _reset_batch()
            start_time = time.perf_counter()
            for _ in range(output_len):
                _one_forward()
            return time.perf_counter() - start_time

        print("Warming up...")
        for _ in tqdm(range(num_iters_warmup), desc="Warmup iterations"):
            run_to_completion()

        latencies: list[float] = []
        for _ in tqdm(range(num_iters), desc="Bench iterations"):
            latencies.append(run_to_completion())

        latencies_arr = np.array(latencies, dtype=np.float64)
        avg_latency = float(np.mean(latencies_arr))
        percentages = [10, 25, 50, 75, 90, 99]
        percentiles = np.percentile(latencies_arr, percentages)
        return ShadowDecodeBenchmarkResult(
            latencies=latencies,
            avg_latency=avg_latency,
            percentages=percentages,
            percentiles=percentiles,
        )
    finally:
        for t in layer_tensors:
            t.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--block-size", type=int, default=16)
    # Benchmark parameters (names aligned with ``vllm bench latency``)
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=10,
        help="Number of iterations to run for warmup.",
    )
    parser.add_argument(
        "--num-iters", type=int, default=30, help="Number of iterations to run."
    )
    parser.add_argument(
        "--output-json", type=str, default=None, help="Path to output JSON file."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    res = benchmark_shadow_latency(
        output_len=args.output_len,
        batch_size=args.batch_size,
        input_len=args.input_len,
        model=args.model,
        dtype=args.dtype,
        block_size=args.block_size,
        num_iters_warmup=args.num_iters_warmup,
        num_iters=args.num_iters,
    )
    print(f"Avg latency: {res.avg_latency} seconds")
    for percentage, percentile in zip(res.percentages, res.percentiles):
        print(f"{percentage}% percentile latency: {percentile} seconds")
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(res.to_json(), f, indent=4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
