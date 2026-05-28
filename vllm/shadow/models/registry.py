# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Factory for loading shadow CPU models."""

from __future__ import annotations

import logging
import time

import torch

from vllm.shadow.models.llama import LlamaForCausalLM, build_llama_for_causal_lm
from vllm.shadow.models.loader import ShadowModelLoader
from vllm.shadow.runtime.config import ShadowModelConfig

logger = logging.getLogger("vllm.shadow.models.registry")

__all__ = ("load_shadow_model_from_config",)


def load_shadow_model_from_config(
    model_config: ShadowModelConfig,
) -> LlamaForCausalLM:
    """Load a model from a configuration."""
    dtype = model_config.dtype
    if not isinstance(dtype, torch.dtype):
        raise TypeError("ShadowModelConfig.dtype must be resolved to torch.dtype")

    t0 = time.perf_counter()
    loader = ShadowModelLoader(model_config)
    hf = loader.hf_config

    weights = {name: tensor for name, tensor in loader.iter_tensors()}
    t1 = time.perf_counter()
    logger.info("Weights loaded time_ms=%.1f", (t1 - t0) * 1000.0)

    if hf.model_type in (
        "qwen2",
        "qwen3",
        "llama",
    ):
        model = build_llama_for_causal_lm(
            hf, weights, dtype, block_size=model_config.block_size
        )
    else:
        raise ValueError(
            f"unsupported shadow model_type={hf.model_type!r} "
            f"(architectures={hf.architectures}); "
            "supported: llama (incl. Llama 3.x), mistral, stablelm, qwen2, qwen3"
        )

    elapsed_ms = (time.perf_counter() - t1) * 1000.0

    logger.info(
        "Shadow CPU model loaded model_type=%s dtype=%s layers=%d time_ms=%.1f",
        hf.model_type,
        dtype,
        hf.num_hidden_layers,
        elapsed_ms,
    )
    return model
