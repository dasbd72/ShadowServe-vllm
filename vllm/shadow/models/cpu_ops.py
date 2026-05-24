# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bindings to vLLM CPU custom ops (``csrc/cpu``) via ``torch.ops``.

If built with GPU, import the auxiliary CPU extension from ``vllm._cpu_C``, this
registers operations under ``torch.ops._cpu_ops``. Otherwise, import the main CPU
extension from ``vllm._C``, this registers operations under ``torch.ops._C``.
"""

from __future__ import annotations

import importlib
import platform
from typing import Any

import torch

_OPS_NS: Any | None = None
_OPS_READY: bool = False
_OPS_SUPPORTS_ONEDNN: bool = False

# Python attribute on ``torch.ops`` for the CPU kernel library slice.
_TORCH_OPS_LIB_CPU = "_C"
_TORCH_OPS_LIB_GPU_AUX = "_cpu_ops"


def _torch_built_with_gpu() -> bool:
    """True when this PyTorch build targets CUDA or ROCm (vLLM GPU wheels)."""
    return torch.version.cuda is not None or torch.version.hip is not None


def _ops_lib_names_for_runtime() -> tuple[str, ...]:
    """Order of ``torch.ops.<name>`` attributes to probe for CPU csrc ops."""
    if _torch_built_with_gpu():
        return (_TORCH_OPS_LIB_GPU_AUX, _TORCH_OPS_LIB_CPU)
    return (_TORCH_OPS_LIB_CPU,)


def _find_ops_ns() -> Any | None:
    for name in _ops_lib_names_for_runtime():
        ns = getattr(torch.ops, name, None)
        if ns is not None and hasattr(ns, "cpu_attention_with_kv_cache"):
            return ns
    return None


def _ensure_cpu_ops() -> Any:
    global _OPS_NS, _OPS_READY, _OPS_SUPPORTS_ONEDNN
    if _OPS_READY and _OPS_NS is not None:
        return _OPS_NS
    ns = _find_ops_ns()
    if ns is not None:
        _OPS_NS = ns
        _OPS_READY = True
        _OPS_SUPPORTS_ONEDNN = bool(hasattr(ns, "create_onednn_mm_handler"))
        return _OPS_NS
    if _torch_built_with_gpu():
        candidates = ("vllm._cpu_C", "vllm._cpu_C_AVX2")
    else:
        candidates = ("vllm._C", "vllm._C_AVX2")
    for mod in candidates:
        try:
            importlib.import_module(mod)
        except ImportError:
            continue
        ns = _find_ops_ns()
        if ns is not None:
            _OPS_NS = ns
            _OPS_READY = True
            _OPS_SUPPORTS_ONEDNN = bool(hasattr(ns, "create_onednn_mm_handler"))
            return _OPS_NS
    raise RuntimeError(
        "vLLM CPU extension is unavailable: cpu_attention_with_kv_cache missing "
        "(GPU builds need the auxiliary CPU extension, e.g. "
        "VLLM_BUILD_CPU_OPS_WITH_GPU=1 / vllm._cpu_C → torch.ops._cpu_ops)"
    )


def shadow_cpu_attn_isa(dtype: torch.dtype, block_size: int, head_dim: int) -> str:
    if head_dim % 32 != 0 and head_dim % 16 == 0:
        return "vec16"
    amx = bool(getattr(torch._C._cpu, "_is_amx_tile_supported", lambda: False)())
    machine = platform.machine().lower()
    is_arm = machine in ("aarch64", "arm64")
    is_s390x = machine.startswith("s390")
    if amx and dtype == torch.bfloat16 and block_size % 32 == 0:
        return "amx"
    if block_size % 32 == 0:
        if is_arm:
            return "neon"
        if is_s390x:
            return "vxe"
        return "vec"
    return "vec16"


def rms_norm(
    out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    ops = _ensure_cpu_ops()
    ops.rms_norm(out, x, weight, float(eps))


def fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> None:
    ops = _ensure_cpu_ops()
    ops.fused_add_rms_norm(input, residual, weight, float(eps))


def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    ops = _ensure_cpu_ops()
    ops.silu_and_mul(out, x)


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    ops = _ensure_cpu_ops()
    ops.rotary_embedding(
        positions, query, key, int(head_size), cos_sin_cache, bool(is_neox)
    )


def cpu_attn_reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    isa: str,
) -> None:
    ops = _ensure_cpu_ops()
    ops.cpu_attn_reshape_and_cache(
        key, value, key_cache, value_cache, slot_mapping, isa
    )


def cpu_attn_get_scheduler_metadata(
    *,
    num_reqs: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_lens: torch.Tensor,
    dtype: torch.dtype,
    query_start_loc: torch.Tensor,
    causal: bool,
    sliding_window_size: int,
    isa: str,
    enable_kv_split: bool,
) -> torch.Tensor:
    ops = _ensure_cpu_ops()
    return ops.get_scheduler_metadata(
        int(num_reqs),
        int(num_heads),
        int(num_kv_heads),
        int(head_dim),
        seq_lens,
        dtype,
        query_start_loc,
        bool(causal),
        int(sliding_window_size),
        isa,
        bool(enable_kv_split),
    )


def cpu_attention_with_kv_cache(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    causal: bool,
    alibi_slopes: torch.Tensor | None,
    sliding_window_left: int,
    sliding_window_right: int,
    block_table: torch.Tensor,
    softcap: float,
    scheduler_metadata: torch.Tensor,
    s_aux: torch.Tensor | None,
) -> None:
    ops = _ensure_cpu_ops()
    ops.cpu_attention_with_kv_cache(
        query,
        key_cache,
        value_cache,
        output,
        query_start_loc,
        seq_lens,
        float(scale),
        bool(causal),
        alibi_slopes,
        int(sliding_window_left),
        int(sliding_window_right),
        block_table,
        float(softcap),
        scheduler_metadata,
        s_aux,
    )


class CPUDNNLGEMMHandler:
    """Opaque oneDNN matmul plan; mirrors ``vllm._custom_ops.CPUDNNLGEMMHandler``."""

    __slots__ = ("handler_tensor", "n", "k")

    def __init__(self) -> None:
        self.handler_tensor: torch.Tensor | None = None
        self.n = -1
        self.k = -1

    def __del__(self) -> None:
        if self.handler_tensor is None:
            return
        ops = _ensure_cpu_ops()
        ops.release_dnnl_matmul_handler(int(self.handler_tensor.item()))


def supports_onednn() -> bool:
    """True when ``create_onednn_mm_handler`` is registered on the CPU ops namespace."""
    _ensure_cpu_ops()
    return _OPS_SUPPORTS_ONEDNN


def is_onednn_acl_supported() -> bool:
    ops = _ensure_cpu_ops()
    if not hasattr(ops, "is_onednn_acl_supported"):
        return False
    return bool(ops.is_onednn_acl_supported())


def create_onednn_mm(
    weight: torch.Tensor,
    primitive_cache_size: int = 32,
) -> CPUDNNLGEMMHandler:
    """Build a matmul plan from ``weight`` shaped ``[K, N]`` (``Linear.weight.t()``)."""
    ops = _ensure_cpu_ops()
    handler = CPUDNNLGEMMHandler()
    handler.k, handler.n = weight.size()
    handler.handler_tensor = torch.tensor(
        ops.create_onednn_mm_handler(weight, int(primitive_cache_size)),
        dtype=torch.int64,
    )
    return handler


def onednn_mm(
    dnnl_handler: CPUDNNLGEMMHandler,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    ops = _ensure_cpu_ops()
    output = torch.empty((*x.shape[0:-1], dnnl_handler.n), dtype=x.dtype)
    ops.onednn_mm(
        output,
        x.reshape(-1, dnnl_handler.k),
        bias,
        dnnl_handler.handler_tensor,
    )
    return output
