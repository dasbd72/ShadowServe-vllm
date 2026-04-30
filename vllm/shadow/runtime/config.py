# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shadow process config aligned with :class:`vllm.engine.arg_utils.EngineArgs`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch

from vllm.shadow.transfer.kv_transport_common import torch_dtype_from_str

ModelDType: TypeAlias = Literal["auto", "float16", "bfloat16"]


@dataclass(slots=True)
class ShadowConfig:
    """Subset of serve-time settings mirrored into the shadow CPU process."""

    model: str = "Qwen/Qwen3-0.6B"
    dtype: ModelDType | torch.dtype = "auto"
    max_model_len: int | None = None
    block_size: int = 16

    def __post_init__(self) -> None:
        if self.dtype == "auto":
            self.dtype = torch.bfloat16
        elif isinstance(self.dtype, str):
            self.dtype = torch_dtype_from_str(self.dtype)
