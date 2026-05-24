# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared building blocks for shadow KV transfers."""

from __future__ import annotations

import mmap
import os
from contextlib import suppress
from math import prod
from typing import Final

import torch

__all__ = (
    "MemfdTensor",
    "dtype_str_from_torch",
    "torch_dtype_from_str",
)

# Handoff/holder ``dtype`` is a canonical, torch-free string (no "torch." prefix).
_DTYPE_STR_TO_TORCH: Final[dict[str, torch.dtype]] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def dtype_str_from_torch(dt: torch.dtype) -> str:
    for k, v in _DTYPE_STR_TO_TORCH.items():
        if v == dt:
            return k
    raise ValueError(f"unsupported torch dtype for shadow KV handoff: {dt}")


def torch_dtype_from_str(name: str) -> torch.dtype:
    if not isinstance(name, str):
        raise TypeError("dtype must be a string")
    key = name.strip()
    if key.startswith("torch."):
        key = key[len("torch.") :]
    dt = _DTYPE_STR_TO_TORCH.get(key)
    if dt is None:
        raise ValueError(f"unsupported dtype string for shadow KV handoff: {name!r}")
    return dt


class MemfdTensor:
    """Received memfd: ``mmap`` + owning fd + tensor view over the mapping.

    Call :meth:`close` to ``munmap`` and :func:`os.close` the descriptor.
    """

    __slots__ = ("_fd", "_mm", "_tensor", "_closed")

    def __init__(self, mm: mmap.mmap, fd: int, tensor: torch.Tensor) -> None:
        self._mm = mm
        self._fd = fd
        self._tensor = tensor
        self._closed = False

    @property
    def fd(self) -> int:
        return self._fd

    @property
    def mm(self) -> mmap.mmap:
        return self._mm

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    def __del__(self) -> None:
        with suppress(SystemExit, KeyboardInterrupt, OSError, AttributeError):
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with suppress(OSError):
            self._mm.close()
            os.close(self._fd)

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> MemfdTensor:
        if tensor.device.type != "cpu":
            raise ValueError("tensor must be a CPU tensor")
        if not tensor.is_contiguous():
            raise ValueError("tensor must be contiguous")
        if tensor.dim() != 5:
            raise ValueError(f"tensor must be 5D, got shape {tuple(tensor.shape)}")
        total_bytes = tensor.numel() * tensor.element_size()
        mfd_flags = getattr(os, "MFD_CLOEXEC", 0)
        fd = os.memfd_create("vllm_shadow_kv_layer", mfd_flags)
        os.ftruncate(fd, total_bytes)
        mm = mmap.mmap(fd, total_bytes, mmap.MAP_SHARED, mmap.PROT_WRITE)
        out = torch.frombuffer(
            mm,
            dtype=tensor.dtype,
            count=tensor.numel(),
        ).reshape(tensor.shape)
        out.copy_(tensor, non_blocking=True)
        return MemfdTensor(mm=mm, fd=fd, tensor=out)

    @staticmethod
    def from_memfd(fd: int, dtype: torch.dtype, shape: tuple[int, ...]) -> MemfdTensor:
        size = prod(shape) * dtype.itemsize
        if os.fstat(fd).st_size != size:
            raise ValueError(f"memfd size {os.fstat(fd).st_size} != required {size}")
        mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_WRITE)
        tensor = torch.frombuffer(mm, dtype=dtype, count=prod(shape)).reshape(shape)
        return MemfdTensor(
            mm=mm,
            fd=fd,
            tensor=tensor,
        )
