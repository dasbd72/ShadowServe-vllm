# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Factory for loading shadow CPU models."""

from __future__ import annotations

import logging
import time

import torch

from vllm.shadow.models.hf_config import load_shadow_hf_config
from vllm.shadow.models.llama import LlamaLikeShadowModel, build_llama_like_model
from vllm.shadow.models.weight_loader import ShadowWeightsLoader
from vllm.shadow.runtime.config import ShadowConfig

logger = logging.getLogger("vllm.shadow.models.registry")

__all__ = ("load_shadow_model_from_config",)


def load_shadow_model_from_config(
    shadow_config: ShadowConfig,
) -> LlamaLikeShadowModel:
    """Load a model from a configuration."""
    dtype = shadow_config.dtype
    if not isinstance(dtype, torch.dtype):
        raise TypeError("ShadowConfig.dtype must be resolved to torch.dtype")

    t0 = time.perf_counter()
    loader = ShadowWeightsLoader(shadow_config)
    hf = load_shadow_hf_config(loader.model_dir)

    weights: dict[str, torch.Tensor] = {}
    for name, tensor in loader.iter_tensors():
        weights[name] = tensor.to(dtype=dtype, device=torch.device("cpu")).contiguous()

    if hf.model_type in (
        "qwen2",
        "qwen3",
        "llama",
    ):
        model = build_llama_like_model(
            hf, weights, dtype, block_size=shadow_config.block_size
        )
    else:
        raise ValueError(
            f"unsupported shadow model_type={hf.model_type!r} "
            f"(architectures={hf.architectures}); "
            "supported: llama (incl. Llama 3.x), mistral, stablelm, qwen2, qwen3"
        )

    del weights
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    logger.info(
        "Shadow CPU model loaded model_type=%s dtype=%s layers=%d time_ms=%.1f",
        hf.model_type,
        dtype,
        hf.num_hidden_layers,
        elapsed_ms,
    )
    return model
