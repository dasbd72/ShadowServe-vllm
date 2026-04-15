# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal sampling for shadow wire ``sampling_params``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

__all__ = (
    "ShadowSamplingParams",
    "sampling_params_from_wire_dict",
    "sample_from_logits",
    "stop_finish_reason",
)


_DEFAULT_MAX_TOKENS_WHEN_MISSING = 2048


@dataclass(slots=True)
class ShadowSamplingParams:
    temperature: float
    top_k: int
    stop_token_ids: list[int]
    max_tokens: int


def sampling_params_from_wire_dict(d: dict[str, Any]) -> ShadowSamplingParams:
    temp = float(d.get("temperature", 1.0))
    top_k_raw = d.get("top_k", -1)
    top_k = int(top_k_raw) if top_k_raw is not None else -1
    stops = d.get("stop_token_ids")
    stop_list: list[int] = []
    if isinstance(stops, list):
        stop_list = [int(x) for x in stops]
    max_tokens = int(d.get("max_tokens", _DEFAULT_MAX_TOKENS_WHEN_MISSING))
    return ShadowSamplingParams(
        temperature=max(0.0, temp),
        top_k=top_k,
        stop_token_ids=stop_list,
        max_tokens=max_tokens,
    )


def sample_from_logits(
    logits: torch.Tensor,
    params: ShadowSamplingParams,
    *,
    generator: torch.Generator | None = None,
) -> int:
    """Greedy with optional temperature / top-k (``top_k <= 0`` disables top-k)."""
    if logits.dim() != 1:
        raise ValueError("logits must be 1-dimensional")
    logits = logits.float()
    if params.temperature <= 0.0:
        return int(torch.argmax(logits).item())
    scaled = logits / max(params.temperature, 1e-8)
    if params.top_k > 0:
        v, _ = torch.topk(scaled, min(params.top_k, scaled.numel()))
        thresh = v[-1]
        scaled = torch.where(scaled < thresh, float("-inf"), scaled)
    probs = torch.softmax(scaled, dim=-1)
    if generator is None:
        return int(torch.multinomial(probs, num_samples=1).item())
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def stop_finish_reason(
    token_id: int,
    params: ShadowSamplingParams,
    eos_token_id: int | None,
    *,
    num_output_tokens_after_append: int | None = None,
) -> str | None:
    """If generation should end after ``token_id``, return TKCTH ``finish_reason``.

    ``\"length\"`` = output token budget (``max_tokens``); ``\"stop\"`` = EOS or
    ``stop_token_ids``; ``None`` = continue.  ``max_tokens`` is evaluated before
    EOS / custom stop tokens.
    """
    if (
        num_output_tokens_after_append is not None
        and params.max_tokens > 0
        and num_output_tokens_after_append >= params.max_tokens
    ):
        return "length"
    if eos_token_id is not None and token_id == eos_token_id:
        return "stop"
    if token_id in params.stop_token_ids:
        return "stop"
    return None
