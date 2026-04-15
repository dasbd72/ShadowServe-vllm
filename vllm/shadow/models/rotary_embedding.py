# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPT-NeoX style RoPE ``cos_sin`` cache for shadow CPU.

Only ``default`` and ``llama3`` are supported.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from vllm.shadow.models.hf_config import ShadowHfConfig


def _inv_freq_standard(
    rotary_dim: int, base: float, device: torch.device
) -> torch.Tensor:
    return 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )


def _cos_sin_table(
    inv_freq: torch.Tensor,
    num_rows: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    t = torch.arange(num_rows, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.cat((cos, sin), dim=-1)


def _inv_freq_llama3(
    rotary_dim: int,
    base: float,
    device: torch.device,
    *,
    scaling_factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    orig_max_position: int,
) -> torch.Tensor:
    inv_freqs = _inv_freq_standard(rotary_dim, base, device)
    low_freq_wavelen = orig_max_position / low_freq_factor
    high_freq_wavelen = orig_max_position / high_freq_factor
    wave_len = 2 * math.pi / inv_freqs
    if low_freq_factor != high_freq_factor:
        smooth = (orig_max_position / wave_len - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
    else:
        smooth = 0
    return torch.where(
        wave_len < high_freq_wavelen,
        inv_freqs,
        torch.where(
            wave_len > low_freq_wavelen,
            inv_freqs / scaling_factor,
            (1 - smooth) * inv_freqs / scaling_factor + smooth * inv_freqs,
        ),
    )


def build_shadow_cos_sin_cache(
    hf: ShadowHfConfig,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Materialize ``cos_sin_cache`` ``[num_rows, rotary_dim]``."""
    rp: dict[str, Any] = hf.rope_parameters
    rope_type = str(rp["rope_type"])
    rotary_dim = int(hf.rotary_dim)
    max_pe = int(hf.max_position_embeddings)
    base = float(rp["rope_theta"])

    if rotary_dim <= 0 or rotary_dim > hf.head_dim or rotary_dim % 2 != 0:
        raise ValueError(
            f"invalid rotary_dim={rotary_dim} for RoPE (head_dim={hf.head_dim})"
        )

    if rope_type == "default":
        inv = _inv_freq_standard(rotary_dim, base, device)
        cache = _cos_sin_table(inv, max_pe, device=device)
        return cache.to(dtype=dtype)

    if rope_type == "llama3":
        for key in (
            "factor",
            "low_freq_factor",
            "high_freq_factor",
            "original_max_position_embeddings",
        ):
            if key not in rp:
                raise ValueError(f"llama3 RoPE requires rope_parameters[{key!r}]")
        inv = _inv_freq_llama3(
            rotary_dim,
            base,
            device,
            scaling_factor=float(rp["factor"]),
            low_freq_factor=float(rp["low_freq_factor"]),
            high_freq_factor=float(rp["high_freq_factor"]),
            orig_max_position=int(rp["original_max_position_embeddings"]),
        )
        cache = _cos_sin_table(inv, max_pe, device=device)
        return cache.to(dtype=dtype)

    raise ValueError(
        f"shadow RoPE: unsupported rope_type {rope_type!r} "
        f"(only 'default' and 'llama3' are supported)"
    )
