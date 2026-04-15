# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU linear layers for the shadow decode path.

Weights use the same HF / ``nn.Linear`` layout ``[out_features, in_features]``.
When oneDNN can build an FP/BF matmul primitive for the weight dtype, we cache
that plan (opaque, not part of ``state_dict``). Many x86 CPUs support BF16
oneDNN matmul but not FP16; in that case we fall back to ``torch.nn.functional.linear``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.shadow.models import cpu_ops

# Matches prior ``create_onednn_mm(..., 32)`` call sites in shadow Llama.
_DEFAULT_ONEDNN_PRIMITIVE_CACHE_SIZE = 32

logger = logging.getLogger("vllm.shadow.models.linear")


class Linear(nn.Module):
    """``y = x @ W^T + b`` using oneDNN when a matmul primitive exists, else PyTorch.

    ``weight`` is ``[out_features, in_features]``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        primitive_cache_size: int = _DEFAULT_ONEDNN_PRIMITIVE_CACHE_SIZE,
    ) -> None:
        super().__init__()
        self._primitive_cache_size = int(primitive_cache_size)
        self.register_buffer("weight", weight)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

        self.in_features = in_features
        self.out_features = out_features

        self._linear_fn = self._build_linear_fn()

    def _build_linear_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        if cpu_ops.supports_onednn():
            try:
                onednn = cpu_ops.create_onednn_mm(
                    self.weight.t(),  # type: ignore
                    self._primitive_cache_size,
                )
                linear_fn = lambda x: cpu_ops.onednn_mm(onednn, x, self.bias)
                self.weight = nn.Parameter(torch.empty(0), requires_grad=False)
                return linear_fn
            except RuntimeError:
                pass
        return lambda x: F.linear(x, self.weight, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self._linear_fn is not None
        return self._linear_fn(x)


__all__ = ("Linear",)
