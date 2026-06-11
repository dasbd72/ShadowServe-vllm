# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark prefill vs decode latency on GPU (vLLM) and CPU (Shadow).

ShadowServe offloads active decoding to a CPU shadow while the hot GPU
handles compute-heavy prefills and the cold GPU initializes. This script
measures the prefill/decode gap that motivates that split:

- Prefill: process ``input_len`` prompt tokens for a batch (TTFT on GPU).
- Decode: one autoregressive step (one new token per request in the batch).

Run ``run_suite.py`` for a full matrix, or invoke this script directly:

    python scripts/cpu_decoding_justification/run.py --backend both
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger("scripts.cpu_decoding_justification.run")


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_LOAD_FORMAT = "serverless_llm"


def _resolve_model(model: str, load_format: str, model_path: str | None) -> str:
    if model_path is not None:
        return model_path
    if load_format == "serverless_llm":
        storage_path = os.getenv("STORAGE_PATH", os.path.expanduser("~/models"))
        return os.path.join(storage_path, "vllm", model)
    return model


@dataclass(slots=True)
class PhaseLatencyResult:
    backend: str
    phase: Literal["prefill", "decode"]
    batch_size: int
    input_len: int
    latencies_s: list[float]
    avg_latency_s: float
    ms_per_token: float
    tokens_per_s: float
    percentiles: dict[str, float]

    def to_json(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "phase": self.phase,
            "batch_size": self.batch_size,
            "input_len": self.input_len,
            "avg_latency_s": self.avg_latency_s,
            "ms_per_token": self.ms_per_token,
            "tokens_per_s": self.tokens_per_s,
            "latencies_s": self.latencies_s,
            "percentiles": self.percentiles,
        }


@dataclass(slots=True)
class BenchmarkResult:
    model: str
    load_format: str
    prefill_gpu: PhaseLatencyResult | None
    decode_gpu: PhaseLatencyResult | None
    prefill_cpu: PhaseLatencyResult | None
    decode_cpu: PhaseLatencyResult | None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "load_format": self.load_format,
        }
        for key in ("prefill_gpu", "decode_gpu", "prefill_cpu", "decode_cpu"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value.to_json()
        if self.prefill_gpu is not None and self.prefill_cpu is not None:
            payload["prefill_gpu_over_cpu_speedup"] = (
                self.prefill_cpu.avg_latency_s / self.prefill_gpu.avg_latency_s
                if self.prefill_gpu.avg_latency_s > 0
                else None
            )
        if self.decode_gpu is not None and self.decode_cpu is not None:
            payload["decode_gpu_over_cpu_speedup"] = (
                self.decode_cpu.avg_latency_s / self.decode_gpu.avg_latency_s
                if self.decode_gpu.avg_latency_s > 0
                else None
            )
        return payload


def _summarize_latencies(
    *,
    backend: str,
    phase: Literal["prefill", "decode"],
    batch_size: int,
    input_len: int,
    latencies_s: list[float],
    tokens_per_step: int,
) -> PhaseLatencyResult:
    arr = np.array(latencies_s, dtype=np.float64)
    avg = float(np.mean(arr))
    percentages = [50, 90, 99]
    percentiles = {
        f"p{int(p)}": float(v)
        for p, v in zip(percentages, np.percentile(arr, percentages))
    }
    ms_per_token = (avg / tokens_per_step) * 1000.0 if tokens_per_step > 0 else 0.0
    tokens_per_s = tokens_per_step / avg if avg > 0 else 0.0
    return PhaseLatencyResult(
        backend=backend,
        phase=phase,
        batch_size=batch_size,
        input_len=input_len,
        latencies_s=latencies_s,
        avg_latency_s=avg,
        ms_per_token=ms_per_token,
        tokens_per_s=tokens_per_s,
        percentiles=percentiles,
    )


def _print_phase(result: PhaseLatencyResult) -> None:
    logger.info(
        "%s %s: avg=%.4fs (%.3f ms/token, %.1f tok/s) p50=%.4fs p90=%.4fs",
        result.backend.upper(),
        result.phase,
        result.avg_latency_s,
        result.ms_per_token,
        result.tokens_per_s,
        result.percentiles["p50"],
        result.percentiles["p90"],
    )


def _print_summary(result: BenchmarkResult) -> None:
    for phase_result in (
        result.prefill_gpu,
        result.decode_gpu,
        result.prefill_cpu,
        result.decode_cpu,
    ):
        if phase_result is not None:
            _print_phase(phase_result)

    payload = result.to_json()
    if "prefill_gpu_over_cpu_speedup" in payload:
        logger.info(
            "Prefill GPU speedup over CPU: %.2fx",
            payload["prefill_gpu_over_cpu_speedup"],
        )
    if "decode_gpu_over_cpu_speedup" in payload:
        logger.info(
            "Decode GPU speedup over CPU: %.2fx",
            payload["decode_gpu_over_cpu_speedup"],
        )


def _alloc_memfd_tensor(shape: tuple[int, ...], dtype: torch.dtype):
    import mmap

    nbytes = int(torch.empty((), dtype=dtype).element_size() * math.prod(shape))
    flags = getattr(os, "MFD_CLOEXEC", 0)
    fd = os.memfd_create("vllm_cpu_decode_bench_kv", flags)
    mm: mmap.mmap | None = None
    try:
        os.ftruncate(fd, nbytes)
        mm = mmap.mmap(fd, nbytes, mmap.MAP_SHARED, mmap.PROT_WRITE)
        tensor = torch.frombuffer(mm, dtype=dtype, count=math.prod(shape)).reshape(
            shape
        )
        tensor.zero_()
        from vllm.shadow.transfer.kv_transport_common import MemfdTensor

        return MemfdTensor(mm=mm, fd=fd, tensor=tensor)
    except BaseException:
        if mm is not None:
            mm.close()
        os.close(fd)
        raise


def benchmark_cpu_phases(
    *,
    model: str,
    load_format: str,
    model_path: str | None,
    batch_size: int,
    input_len: int,
    dtype: str,
    block_size: int,
    max_model_len: int | None,
    num_iters_warmup: int,
    num_iters: int,
) -> tuple[PhaseLatencyResult, PhaseLatencyResult]:
    from vllm.shadow.models.kv_state import ShadowKvState
    from vllm.shadow.models.registry import load_shadow_model_from_config
    from vllm.shadow.models.sampling import (
        Sampler,
        ShadowSamplingParams,
        shadow_sampling_metadata,
    )
    from vllm.shadow.runtime.config import ShadowModelConfig
    from vllm.shadow.transfer.kv_transport_common import torch_dtype_from_str

    resolved_model = _resolve_model(model, load_format, model_path)
    num_threads = int(torch.get_num_threads())
    logger.info("CPU shadow threads: %d", num_threads)

    shadow_cfg = ShadowModelConfig(
        model=resolved_model,
        load_format=load_format,
        dtype=dtype,
        max_model_len=max_model_len,
        block_size=block_size,
    )
    model_obj = load_shadow_model_from_config(shadow_cfg)
    sampler = Sampler()

    dummy_prompt_token_ids = np.random.randint(
        10_000, size=(batch_size, input_len), dtype=np.int64
    )

    blocks_per_req = math.ceil(input_len / block_size) + 1
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
    layer_tensors = []
    try:
        for layer_idx in range(int(model_obj.hf.num_hidden_layers)):
            memfd = _alloc_memfd_tensor(layer_shape, torch_dtype_from_str(dtype))
            layer_tensors.append(memfd)
            kv_state.register_layer(layer_idx, memfd)

        sampling = ShadowSamplingParams(
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            stop_token_ids=[],
            max_tokens=2,
        )

        def _reset_requests() -> list[str]:
            for rid in list(kv_state.requests.keys()):
                kv_state.free_request(rid)
            for memfd in layer_tensors:
                memfd.tensor.zero_()
            request_ids: list[str] = []
            for i in range(batch_size):
                rid = f"bench-{i}"
                request_ids.append(rid)
                kv_state.add_request(
                    request_id=rid,
                    prompt_token_ids=dummy_prompt_token_ids[i].tolist(),
                    output_token_ids=[],
                    num_computed_tokens=0,
                    block_table=list(
                        range(i * blocks_per_req, (i + 1) * blocks_per_req)
                    ),
                    sampling_params=sampling,
                )
            return request_ids

        def _one_forward(request_ids: list[str]) -> None:
            query_token_ids: list[list[int]] = []
            sampling_params: list[ShadowSamplingParams] = []
            for rid in request_ids:
                tids, _, sp, allocated = kv_state.prepare_request(rid)
                if not tids or not allocated:
                    raise RuntimeError(f"request {rid!r} not ready for forward")
                query_token_ids.append(tids)
                sampling_params.append(sp)
            batch = kv_state.build_attention_batch(request_ids, query_token_ids)
            logits = model_obj(batch)
            meta = shadow_sampling_metadata(
                sampling_params,
                vocab=logits.shape[-1],
                device=logits.device,
                generators={},
            )
            token_ids = sampler(logits, meta).tolist()
            for row_idx, rid in enumerate(request_ids):
                kv_state.advance_decoded_token(rid, token_ids[row_idx])

        def _run_prefill_once() -> float:
            request_ids = _reset_requests()
            start = time.perf_counter()
            _one_forward(request_ids)
            return time.perf_counter() - start

        def _run_decode_once() -> float:
            request_ids = _reset_requests()
            _one_forward(request_ids)
            start = time.perf_counter()
            _one_forward(request_ids)
            return time.perf_counter() - start

        for _ in tqdm(range(num_iters_warmup), desc="CPU warmup"):
            _run_prefill_once()
            _run_decode_once()

        prefill_latencies = [
            _run_prefill_once() for _ in tqdm(range(num_iters), desc="CPU prefill")
        ]
        decode_latencies = [
            _run_decode_once() for _ in tqdm(range(num_iters), desc="CPU decode")
        ]
    finally:
        for memfd in layer_tensors:
            memfd.close()

    prefill_tokens = batch_size * input_len
    decode_tokens = batch_size
    return (
        _summarize_latencies(
            backend="cpu",
            phase="prefill",
            batch_size=batch_size,
            input_len=input_len,
            latencies_s=prefill_latencies,
            tokens_per_step=prefill_tokens,
        ),
        _summarize_latencies(
            backend="cpu",
            phase="decode",
            batch_size=batch_size,
            input_len=input_len,
            latencies_s=decode_latencies,
            tokens_per_step=decode_tokens,
        ),
    )


def benchmark_gpu_phases(
    *,
    model: str,
    load_format: str,
    model_path: str | None,
    batch_size: int,
    input_len: int,
    dtype: str,
    block_size: int,
    max_model_len: int | None,
    num_iters_warmup: int,
    num_iters: int,
) -> tuple[PhaseLatencyResult, PhaseLatencyResult]:
    import dataclasses

    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs

    resolved_model = _resolve_model(model, load_format, model_path)
    engine_args = EngineArgs(
        model=resolved_model,
        load_format=load_format,
        dtype=dtype,
        block_size=block_size,
        max_model_len=max_model_len,
        enable_prefix_caching=False,
    )
    llm = LLM(**dataclasses.asdict(engine_args))

    max_model_len = llm.llm_engine.model_config.max_model_len
    if max_model_len < input_len + 2:
        raise ValueError(
            f"max_model_len={max_model_len} < input_len + 2 ({input_len + 2})"
        )

    dummy_prompt_token_ids = np.random.randint(
        10_000, size=(batch_size, input_len), dtype=np.int64
    )
    dummy_prompts = [
        {"prompt_token_ids": batch.tolist()} for batch in dummy_prompt_token_ids
    ]

    prefill_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=1,
        detokenize=False,
    )
    decode_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=2,
        detokenize=False,
    )

    def _run_prefill_once() -> float:
        start = time.perf_counter()
        llm.generate(dummy_prompts, sampling_params=prefill_params, use_tqdm=False)
        return time.perf_counter() - start

    def _run_decode_once() -> float:
        start = time.perf_counter()
        llm.generate(dummy_prompts, sampling_params=decode_params, use_tqdm=False)
        end = time.perf_counter()
        return end - start

    for _ in tqdm(range(num_iters_warmup), desc="GPU warmup"):
        _run_prefill_once()
        _run_decode_once()

    prefill_latencies: list[float] = []
    decode_latencies: list[float] = []
    for _ in tqdm(range(num_iters), desc="GPU prefill/decode"):
        prefill = _run_prefill_once()
        total = _run_decode_once()
        prefill_latencies.append(prefill)
        decode_latencies.append(max(total - prefill, 0.0))

    prefill_tokens = batch_size * input_len
    decode_tokens = batch_size
    return (
        _summarize_latencies(
            backend="gpu",
            phase="prefill",
            batch_size=batch_size,
            input_len=input_len,
            latencies_s=prefill_latencies,
            tokens_per_step=prefill_tokens,
        ),
        _summarize_latencies(
            backend="gpu",
            phase="decode",
            batch_size=batch_size,
            input_len=input_len,
            latencies_s=decode_latencies,
            tokens_per_step=decode_tokens,
        ),
    )


def run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    prefill_gpu = decode_gpu = prefill_cpu = decode_cpu = None

    if args.backend in ("gpu", "both"):
        prefill_gpu, decode_gpu = benchmark_gpu_phases(
            model=args.model,
            load_format=args.load_format,
            model_path=args.model_path,
            batch_size=args.batch_size,
            input_len=args.input_len,
            dtype=args.dtype,
            block_size=args.block_size,
            max_model_len=args.max_model_len,
            num_iters_warmup=args.num_iters_warmup,
            num_iters=args.num_iters,
        )

    if args.backend in ("cpu", "both"):
        prefill_cpu, decode_cpu = benchmark_cpu_phases(
            model=args.model,
            load_format=args.load_format,
            model_path=args.model_path,
            batch_size=args.batch_size,
            input_len=args.input_len,
            dtype=args.dtype,
            block_size=args.block_size,
            max_model_len=args.max_model_len,
            num_iters_warmup=args.num_iters_warmup,
            num_iters=args.num_iters,
        )

    result = BenchmarkResult(
        model=args.model,
        load_format=args.load_format,
        prefill_gpu=prefill_gpu,
        decode_gpu=decode_gpu,
        prefill_cpu=prefill_cpu,
        decode_cpu=decode_cpu,
    )
    _print_summary(result)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend",
        choices=("gpu", "cpu", "both"),
        default="both",
        help="Run GPU vLLM, CPU shadow, or both (default: both)",
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--load-format",
        default=DEFAULT_LOAD_FORMAT,
        help="vLLM/shadow load_format (default: serverless_llm)",
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="Override model directory (default: $STORAGE_PATH/vllm/<model>)",
    )
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--input-len",
        type=int,
        default=2048,
        help="Prompt length per request (prefill compute knob)",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help=(
            "Cap KV cache context length (default: model max). "
            "Lower this if GPU KV cache allocation fails."
        ),
    )
    p.add_argument("--num-iters-warmup", type=int, default=5)
    p.add_argument("--num-iters", type=int, default=20)
    p.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write structured results",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args()
    result = run_benchmark(args)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result.to_json(), f, indent=2)
        logger.info("Wrote results to %s", args.output_json)


if __name__ == "__main__":
    main()
