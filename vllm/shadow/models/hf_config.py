# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read ``config.json`` from a local HF checkpoint (no ``transformers`` import)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


def _model_type(raw: dict[str, Any]) -> str:
    return str(raw.get("model_type", "")).lower()


def _default_rope_theta_for_model(raw: dict[str, Any]) -> float:
    mt = _model_type(raw)
    if mt == "qwen3":
        return 1_000_000.0
    return 10_000.0


def _extract_initial_rope_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten HF ``rope_parameters`` / legacy ``rope_scaling`` to a single dict."""
    rp = raw.get("rope_parameters")
    if isinstance(rp, dict) and rp:
        out = dict(rp)
        if list(out.keys()) == ["default"] and isinstance(out["default"], dict):
            return dict(out["default"])
        return out
    rs = raw.get("rope_scaling")
    if isinstance(rs, dict) and rs:
        return dict(rs)
    return {}


def normalize_shadow_rope_parameters(
    raw: dict[str, Any], *, max_position_embeddings: int
) -> dict[str, Any]:
    """Normalize HF RoPE fields.

    Shadow CPU supports only ``rope_type`` ``default`` and ``llama3``.
    """
    rp = _extract_initial_rope_dict(raw)
    for key in (
        "rope_theta",
        "partial_rotary_factor",
        "original_max_position_embeddings",
    ):
        if key in raw and key not in rp:
            rp[key] = raw[key]

    if "rope_type" not in rp and "type" in rp:
        rp["rope_type"] = rp["type"]

    # Nested per-layer ``rope_parameters`` (e.g. Gemma3):
    # only dict values, no rope_type.
    if (
        rp
        and "rope_type" not in rp
        and "type" not in rp
        and all(isinstance(v, dict) for v in rp.values())
    ):
        raise NotImplementedError(
            "shadow RoPE: nested rope_parameters by attention layer type are not "
            "supported (text-only shadow stack)"
        )

    if "rope_theta" not in rp:
        rp["rope_theta"] = _default_rope_theta_for_model(raw)

    rp.setdefault("rope_type", "default")
    rp.setdefault("partial_rotary_factor", 1.0)

    rope_type = str(rp["rope_type"])
    if rope_type not in ("default", "llama3"):
        raise NotImplementedError(
            "shadow RoPE: only rope_type 'default' and 'llama3' are supported; "
            f"got {rope_type!r}"
        )

    if rope_type == "default" and "mrope_section" in rp:
        raise NotImplementedError(
            "shadow RoPE: mrope_section (mRoPE) is not supported on the CPU path"
        )

    if rope_type == "default" and rp.get("use_fope"):
        raise NotImplementedError(
            "shadow RoPE: Fourier RoPE (use_fope) is not supported on the CPU path"
        )

    if rope_type == "llama3":
        rp.setdefault(
            "original_max_position_embeddings",
            raw.get("original_max_position_embeddings", max_position_embeddings),
        )

    return rp


@dataclass(slots=True)
class ShadowHfConfig:
    """Fields needed for Llama / Qwen2 / Qwen3–style CPU decode."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    rope_theta: float
    partial_rotary_factor: float
    rotary_dim: int
    sliding_window: int | None
    head_dim: int
    eos_token_id: int | None
    architectures: list[str]
    # Normalized RoPE dict (``rope_type``, ``rope_theta``, scaling fields)
    rope_parameters: dict[str, Any]


def load_shadow_hf_config(model_dir: str) -> ShadowHfConfig:
    path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing config.json in {model_dir}")
    with open(path) as f:
        raw: dict = json.load(f)

    model_type = str(raw.get("model_type", "llama"))
    hidden_size = int(raw["hidden_size"])
    num_hidden_layers = int(raw["num_hidden_layers"])
    num_attention_heads = int(raw["num_attention_heads"])
    num_key_value_heads = int(raw.get("num_key_value_heads", num_attention_heads))
    intermediate_size = int(raw["intermediate_size"])
    rms_norm_eps = float(raw.get("rms_norm_eps", 1e-6))
    vocab_size = int(raw["vocab_size"])
    max_position_embeddings = int(raw.get("max_position_embeddings", 8192))
    rope_parameters = normalize_shadow_rope_parameters(
        raw, max_position_embeddings=max_position_embeddings
    )
    rope_theta = float(rope_parameters["rope_theta"])
    partial_rotary_factor = float(rope_parameters["partial_rotary_factor"])
    if not (0.0 < partial_rotary_factor <= 1.0):
        raise ValueError(
            f"partial_rotary_factor must be in (0, 1], got {partial_rotary_factor}"
        )
    sw = raw.get("sliding_window")
    sliding_window = None if sw is None else int(sw)

    if "head_dim" in raw:
        head_dim = int(raw["head_dim"])
        if head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
    else:
        head_dim = hidden_size // num_attention_heads
        if head_dim * num_attention_heads != hidden_size:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

    rotary_dim = int(head_dim * partial_rotary_factor)
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError(
            f"invalid rotary_dim={rotary_dim} (head_dim={head_dim}, "
            f"partial_rotary_factor={partial_rotary_factor})"
        )

    eos = raw.get("eos_token_id")
    eos_token_id: int | None
    if eos is None:
        eos_token_id = None
    elif isinstance(eos, list):
        eos_token_id = int(eos[0]) if eos else None
    else:
        eos_token_id = int(eos)

    arch_raw = raw.get("architectures")
    architectures = [str(x) for x in arch_raw] if isinstance(arch_raw, list) else []

    return ShadowHfConfig(
        model_type=model_type,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        rms_norm_eps=rms_norm_eps,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        rope_theta=rope_theta,
        partial_rotary_factor=partial_rotary_factor,
        rotary_dim=rotary_dim,
        sliding_window=sliding_window,
        head_dim=head_dim,
        eos_token_id=eos_token_id,
        architectures=architectures,
        rope_parameters=dict(rope_parameters),
    )
