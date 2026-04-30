# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import time

logger = logging.getLogger("vllm.shadow.runtime.logger")


class StatLogger:
    def __init__(self):
        self._tot_generation_tokens: int = 0
        self._last_log_time: float = time.monotonic()
        self._num_requests: int = 0
        self._kv_cache_usage: float = 0.0
        self._num_generation_tokens: int = 0
        self._preprocess_elapsed: float = 0.0
        self._forward_elapsed: float = 0.0
        self._postprocess_elapsed: float = 0.0

    def record(
        self,
        num_requests: int,
        kv_cache_usage: float,
        num_generation_tokens: int,
        preprocess_elapsed: float,
        forward_elapsed: float,
        postprocess_elapsed: float,
    ):
        self._tot_generation_tokens += num_generation_tokens
        self._num_requests = num_requests
        self._kv_cache_usage = kv_cache_usage
        self._num_generation_tokens += num_generation_tokens
        self._preprocess_elapsed += preprocess_elapsed
        self._forward_elapsed += forward_elapsed
        self._postprocess_elapsed += postprocess_elapsed

    def _get_generation_throughput(self, now: float) -> float:
        delta_time = now - self._last_log_time
        if delta_time <= 0.0:
            return 0.0
        return float(self._num_generation_tokens / delta_time)

    def _reset(self, now: float):
        self._last_log_time = now
        self._num_generation_tokens = 0
        self._preprocess_elapsed = 0
        self._forward_elapsed = 0
        self._postprocess_elapsed = 0

    def log(self):
        now = time.monotonic()
        num_requests = self._num_requests
        generation_throughput = self._get_generation_throughput(now)
        kv_cache_usage = self._kv_cache_usage * 100
        preprocess_elapsed = self._preprocess_elapsed
        forward_elapsed = self._forward_elapsed
        postprocess_elapsed = self._postprocess_elapsed
        is_idle = not any(
            (
                num_requests,
                generation_throughput,
                kv_cache_usage,
                preprocess_elapsed,
                forward_elapsed,
                postprocess_elapsed,
            )
        )
        self._reset(now)

        if is_idle:
            return

        logger.info(
            "Total generation tokens: %d, "
            "Avg generation throughput: %.1f tokens/s, Running: %d reqs, "
            "CPU KV cache usage: %.1f%%, "
            "Preprocess: %.3f s, Forward: %.3f s, Postprocess: %.3f s",
            self._tot_generation_tokens,
            generation_throughput,
            num_requests,
            kv_cache_usage,
            preprocess_elapsed,
            forward_elapsed,
            postprocess_elapsed,
        )
