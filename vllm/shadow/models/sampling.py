# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shadow CPU sampling: ``Sampler`` + ``TopKTopPSampler``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

__all__ = (
    "ShadowSamplingParams",
    "sampling_params_from_wire_dict",
    "ShadowSamplingMetadata",
    "shadow_sampling_metadata",
    "TopKTopPSampler",
    "Sampler",
    "stop_finish_reason",
)


_DEFAULT_MAX_TOKENS_WHEN_MISSING = 2048

# Match ``vllm.v1.sample.sampler._SAMPLING_EPS`` (greedy vs stochastic boundary).
_SAMPLING_EPS = 1e-5


@dataclass(slots=True)
class ShadowSamplingParams:
    temperature: float
    top_k: int
    top_p: float
    stop_token_ids: list[int]
    max_tokens: int


def sampling_params_from_wire_dict(d: dict[str, Any]) -> ShadowSamplingParams:
    temp = float(d.get("temperature", 1.0))
    top_k = int(d.get("top_k", 0))
    top_p = float(d.get("top_p", 1.0))
    if top_p <= 0.0 or top_p > 1.0:
        top_p = 1.0
    stops = d.get("stop_token_ids", [])
    stop_list = [int(x) for x in stops]
    max_tokens = int(d.get("max_tokens", _DEFAULT_MAX_TOKENS_WHEN_MISSING))
    return ShadowSamplingParams(
        temperature=max(0.0, temp),
        top_k=top_k,
        top_p=top_p,
        stop_token_ids=stop_list,
        max_tokens=max_tokens,
    )


@dataclass(slots=True)
class ShadowSamplingMetadata:
    """Batch fields consumed by ``Sampler.sample``."""

    temperature: torch.Tensor
    top_k: torch.Tensor | None
    top_p: torch.Tensor | None
    generators: dict[int, torch.Generator]
    all_greedy: bool
    all_random: bool


def _k_p_tensors_from_params(
    params: Sequence[ShadowSamplingParams],
    vocab: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    use_top_k = any(0 < p.top_k < vocab for p in params)
    use_top_p = any(p.top_p < 1.0 for p in params)
    k_tensor = None
    p_tensor = None
    if use_top_p:
        p_tensor = torch.tensor(
            [float(p.top_p) for p in params],
            device=device,
            dtype=torch.float32,
        )
    if use_top_k:
        k_list: list[int] = []
        for p in params:
            tk = p.top_k
            if 0 < tk < vocab:
                k_list.append(min(tk, vocab))
            else:
                k_list.append(vocab)
        k_tensor = torch.tensor(k_list, device=device, dtype=torch.long)
    return k_tensor, p_tensor


def shadow_sampling_metadata(
    params: Sequence[ShadowSamplingParams],
    *,
    vocab: int,
    device: torch.device,
    generators: dict[int, torch.Generator],
) -> ShadowSamplingMetadata:
    temperature = torch.tensor(
        [float(p.temperature) for p in params],
        device=device,
        dtype=torch.float32,
    )
    all_greedy = bool((temperature < _SAMPLING_EPS).all().item())
    all_random = bool((temperature >= _SAMPLING_EPS).all().item())
    top_k, top_p = _k_p_tensors_from_params(params, vocab, device)
    return ShadowSamplingMetadata(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        generators=generators,
        all_greedy=all_greedy,
        all_random=all_random,
    )


@torch.compile(dynamic=True)
def _compiled_random_sample(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty_like(probs)
    q.exponential_()
    return probs.div(q).argmax(dim=-1).view(-1)


class TopKTopPSampler(nn.Module):
    """CPU top-k / top-p + Gumbel–max."""

    @staticmethod
    def _apply_top_k_only(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        no_top_k_mask = k == logits.shape[1]
        k = k.masked_fill(no_top_k_mask, 1)
        max_top_k = k.max()
        k_index = k.sub_(1).unsqueeze(1)
        top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
        top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
        return logits.masked_fill_(logits < top_k_mask, -float("inf"))

    @staticmethod
    def _apply_top_k_top_p_pytorch(
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
        *,
        allow_cpu_sync: bool,
    ) -> torch.Tensor:
        if p is None:
            if k is None:
                return logits
            if allow_cpu_sync:
                return TopKTopPSampler._apply_top_k_only(logits, k)

        logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

        if k is not None:
            top_k_mask = logits_sort.size(1) - k.to(torch.long)
            top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
            top_k_mask = logits_sort < top_k_mask
            logits_sort.masked_fill_(top_k_mask, -float("inf"))

        if p is not None:
            probs_sort = logits_sort.softmax(dim=-1)
            probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
            top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
            top_p_mask[:, -1] = False
            logits_sort.masked_fill_(top_p_mask, -float("inf"))

        return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)

    def forward(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> torch.Tensor:
        logits = TopKTopPSampler._apply_top_k_top_p_pytorch(
            logits, k, p, allow_cpu_sync=True
        )
        if len(generators) != logits.shape[0]:
            return _compiled_random_sample(logits)

        probs = logits.softmax(dim=-1, dtype=torch.float32)
        q = torch.empty_like(probs)
        q.exponential_()
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)

        return probs.div_(q).argmax(dim=-1).view(-1)


class Sampler(nn.Module):
    """Samples next tokens from logits."""

    def __init__(self) -> None:
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler()

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: ShadowSamplingMetadata,
    ) -> torch.Tensor:
        logits = logits.to(torch.float32)
        return self.sample(logits, sampling_metadata)

    @staticmethod
    def apply_temperature(
        logits: torch.Tensor,
        temp: torch.Tensor,
        all_random: bool,
    ) -> torch.Tensor:
        if not all_random:
            temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        return logits.div_(temp.unsqueeze(dim=1))

    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: ShadowSamplingMetadata,
    ) -> torch.Tensor:
        assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                return greedy_sampled

        logits = self.apply_temperature(
            logits,
            sampling_metadata.temperature,
            sampling_metadata.all_random,
        )

        random_sampled = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        if greedy_sampled is None:
            return random_sampled

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,
        )
        return sampled


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
