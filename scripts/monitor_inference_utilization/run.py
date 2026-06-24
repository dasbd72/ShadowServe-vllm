# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Poll CPU and GPU utilization while starting AsyncLLM and serving requests.

Single-GPU only. Samples from before engine init through inference completion so
thesis plots can show low host CPU use while the GPU is busy.

Usage:
    python scripts/monitor_inference_utilization/run.py \\
        --model meta-llama/Llama-3.2-1B-Instruct \\
        --num-requests 8 \\
        --output-dir logs/monitor_inference_utilization
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import threading
import time
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import psutil

import vllm.envs as vllm_envs
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logging_utils import NewLineFormatter
from vllm.utils.import_utils import import_pynvml
from vllm.v1.engine.async_llm import AsyncLLM

logger = logging.getLogger("scripts/monitor_inference_utilization/run.py")

Phase = Literal["init", "inference", "shutdown"]


@dataclass
class UtilSample:
    t_ms: float
    phase: Phase
    cpu_system_pct: float
    cpu_process_pct: float
    cpu_vllm_tree_pct: float
    gpu_util_pct: float
    gpu_mem_util_pct: float
    gpu_mem_used_mb: float


@dataclass
class PhaseWindow:
    start_ms: float
    end_ms: float


@dataclass
class RunResult:
    model: str
    num_requests: int
    max_tokens: int
    input_len: int
    max_model_len: int
    gpu_index: int
    poll_interval_ms: float
    init_s: float
    inference_s: float
    total_s: float
    phases: dict[str, PhaseWindow]
    samples: list[UtilSample] = field(default_factory=list)


class UtilizationPoller:
    """Background sampler for system/process CPU and one GPU via NVML."""

    def __init__(
        self,
        *,
        gpu_index: int,
        poll_interval_s: float,
        process: psutil.Process,
    ) -> None:
        self._gpu_index = gpu_index
        self._poll_interval_s = poll_interval_s
        self._process = process
        self._pynvml = import_pynvml()
        self._gpu_handle = None

        self._phase: Phase = "init"
        self._t0 = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[UtilSample] = []

    def start(self) -> None:
        self._pynvml.nvmlInit()
        self._gpu_handle = self._pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
        # Prime non-blocking CPU counters.
        psutil.cpu_percent(interval=None)
        self._process.cpu_percent(interval=None)
        self._prime_tree_cpu()

        self._t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="util-poller",
            daemon=True,
        )
        self._thread.start()

    def set_phase(self, phase: Phase) -> None:
        self._phase = phase

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._pynvml.nvmlShutdown()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def _tree_cpu_percent(self) -> float:
        total = self._process.cpu_percent(interval=None)
        for child in self._process.children(recursive=True):
            with suppress(psutil.NoSuchProcess, psutil.ZombieProcess):
                total += child.cpu_percent(interval=None)
        return total

    def _prime_tree_cpu(self) -> None:
        self._tree_cpu_percent()

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            t_ms = self.elapsed_ms()
            util = self._pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
            mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            self.samples.append(
                UtilSample(
                    t_ms=t_ms,
                    phase=self._phase,
                    cpu_system_pct=psutil.cpu_percent(interval=None),
                    cpu_process_pct=self._process.cpu_percent(interval=None),
                    cpu_vllm_tree_pct=self._tree_cpu_percent(),
                    gpu_util_pct=float(util.gpu),
                    gpu_mem_util_pct=float(util.memory),
                    gpu_mem_used_mb=mem.used / (1024 * 1024),
                )
            )
            self._stop.wait(self._poll_interval_s)


def _configure_vllm_file_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    vllm_logger = logging.getLogger("vllm")
    for handler in list(vllm_logger.handlers):
        if handler.name == "vllm":
            vllm_logger.removeHandler(handler)
            handler.close()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.name = "vllm"
    handler.setLevel(vllm_envs.VLLM_LOGGING_LEVEL)
    handler.setFormatter(
        NewLineFormatter(
            fmt=(
                f"{vllm_envs.VLLM_LOGGING_PREFIX}%(levelname)s %(asctime)s "
                "[%(fileinfo)s:%(lineno)d] %(message)s"
            ),
            datefmt="%m-%d %H:%M:%S",
        )
    )
    vllm_logger.addHandler(handler)


def _make_prompt(input_len: int) -> str:
    # Deterministic filler tokens; tokenizer length is approximate.
    return "word " * max(1, input_len)


async def _run_request(
    engine: AsyncLLM,
    *,
    prompt: str,
    request_id: str,
    max_tokens: int,
) -> None:
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    async for output in engine.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id=request_id,
    ):
        if output.finished:
            break


async def run_benchmark(args: argparse.Namespace) -> RunResult:
    model = args.model
    prompt = _make_prompt(args.input_len)
    poll_interval_s = args.poll_interval_ms / 1000.0

    poller = UtilizationPoller(
        gpu_index=args.gpu,
        poll_interval_s=poll_interval_s,
        process=psutil.Process(),
    )
    phase_windows: dict[str, PhaseWindow] = {}

    poller.start()
    t_start = time.perf_counter()
    phase_windows["init"] = PhaseWindow(start_ms=0.0, end_ms=0.0)

    logger.info("Initializing AsyncLLM on GPU %d (model=%s)", args.gpu, model)
    init_t0 = time.perf_counter()
    engine_args = AsyncEngineArgs(
        model=model,
        dtype=args.dtype,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        disable_log_stats=True,
        max_model_len=args.max_model_len,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    max_model_len = engine.model_config.max_model_len
    min_context = args.input_len + args.max_tokens
    if max_model_len < min_context:
        raise ValueError(
            f"max_model_len={max_model_len} < input_len + max_tokens ({min_context})"
        )
    init_s = time.perf_counter() - init_t0
    phase_windows["init"].end_ms = poller.elapsed_ms()

    poller.set_phase("inference")
    phase_windows["inference"] = PhaseWindow(
        start_ms=phase_windows["init"].end_ms,
        end_ms=phase_windows["init"].end_ms,
    )
    logger.info(
        "Running %d concurrent requests "
        "(max_tokens=%d, input_len~%d, max_model_len=%d)",
        args.num_requests,
        args.max_tokens,
        args.input_len,
        max_model_len,
    )

    infer_t0 = time.perf_counter()
    await asyncio.gather(
        *[
            _run_request(
                engine,
                prompt=f"{i}: {prompt}",
                request_id=f"util-{i}",
                max_tokens=args.max_tokens,
            )
            for i in range(args.num_requests)
        ]
    )
    inference_s = time.perf_counter() - infer_t0
    phase_windows["inference"].end_ms = poller.elapsed_ms()

    poller.set_phase("shutdown")
    phase_windows["shutdown"] = PhaseWindow(
        start_ms=phase_windows["inference"].end_ms,
        end_ms=phase_windows["inference"].end_ms,
    )
    engine.shutdown()
    phase_windows["shutdown"].end_ms = poller.elapsed_ms()
    poller.stop()

    total_s = time.perf_counter() - t_start
    logger.info(
        "Done init=%.2fs inference=%.2fs total=%.2fs samples=%d",
        init_s,
        inference_s,
        total_s,
        len(poller.samples),
    )

    return RunResult(
        model=model,
        num_requests=args.num_requests,
        max_tokens=args.max_tokens,
        input_len=args.input_len,
        max_model_len=max_model_len,
        gpu_index=args.gpu,
        poll_interval_ms=args.poll_interval_ms,
        init_s=init_s,
        inference_s=inference_s,
        total_s=total_s,
        phases=phase_windows,
        samples=poller.samples,
    )


def _write_outputs(result: RunResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "monitor_inference_utilization.json"
    csv_path = output_dir / "monitor_inference_utilization.csv"

    payload = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": result.model,
            "num_requests": result.num_requests,
            "max_tokens": result.max_tokens,
            "input_len": result.input_len,
            "max_model_len": result.max_model_len,
            "gpu_index": result.gpu_index,
            "poll_interval_ms": result.poll_interval_ms,
            "init_s": result.init_s,
            "inference_s": result.inference_s,
            "total_s": result.total_s,
            "phases": {name: asdict(window) for name, window in result.phases.items()},
        },
        "samples": [asdict(s) for s in result.samples],
    }
    json_path.write_text(json.dumps(payload, indent=2))

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "t_ms",
                "phase",
                "cpu_system_pct",
                "cpu_process_pct",
                "cpu_vllm_tree_pct",
                "gpu_util_pct",
                "gpu_mem_util_pct",
                "gpu_mem_used_mb",
            ],
        )
        writer.writeheader()
        for sample in result.samples:
            writer.writerow(asdict(sample))

    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--gpu", type=int, default=0, help="NVML GPU index to monitor")
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="Fraction of GPU memory for KV cache (single GPU)",
    )
    p.add_argument("--num-requests", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument(
        "--input-len",
        type=int,
        default=128,
        help="Approximate prompt length in words",
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
    p.add_argument(
        "--poll-interval-ms",
        type=float,
        default=10.0,
        help="CPU/GPU sample period",
    )
    p.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs (faster startup, less realistic serving)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/monitor_inference_utilization"),
    )
    return p.parse_args()


async def main_async() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    vllm_log_path = args.output_dir / "vllm.log"
    _configure_vllm_file_logging(vllm_log_path)

    result = await run_benchmark(args)
    json_path, csv_path = _write_outputs(result, args.output_dir)
    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", csv_path)
    logger.info("Wrote %s", vllm_log_path)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
