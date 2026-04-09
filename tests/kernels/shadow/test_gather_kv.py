# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for csrc/shadow_kernels.cu — gather_kv_blocks op.

The kernel copies KV blocks from GPU (NHD layout) to pinned CPU memory
(HND layout).

GPU KV cache shape:  (2, num_gpu_blocks, block_size, num_kv_heads, head_size) (NHD)
Pinned host shape:   (2, num_cpu_blocks, num_kv_heads, block_size, head_size) (HND)
- V: Row-major over head_size
- K: Column-major packed tile
"""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reference_gather(
    kv_cache: torch.Tensor,
    src_block_table: list[int],
    dst_block_table: list[int],
    num_kv_heads: int,
    head_size: int,
    block_size: int,
) -> torch.Tensor:
    """Pure-Python reference matching ``gather_kv_blocks_kernel``.

    K (side 0): column-major packed tile — flat index ``token + d * block_size``
    within each (block, head) region of size ``block_size * head_size``.

    V (side 1): HND — ``permute(1, 0, 2)`` of the NHD src block.
    """
    num_blocks = len(src_block_table)
    max_dst_blk = max(dst_block_table) + 1
    out = torch.zeros(
        2,
        max_dst_blk,
        num_kv_heads,
        block_size,
        head_size,
        dtype=kv_cache.dtype,
    )
    for b in range(num_blocks):
        src_blk = src_block_table[b]
        dst_blk = dst_block_table[b]

        # Key: NHD → column-major packed tile (same memory order as cpu_attn_vec.hpp).
        src_k = kv_cache[0, src_blk]  # [block_size, num_kv_heads, head_size]
        flat_k = out[0, dst_blk].reshape(-1)  # [num_kv_heads * block_size * head_size]
        for tok in range(block_size):
            for h in range(num_kv_heads):
                for d in range(head_size):
                    flat_k[h * (block_size * head_size) + tok + d * block_size] = src_k[
                        tok, h, d
                    ]

        # Value: NHD → HND (contiguous row-major over head_size).
        out[1, dst_blk] = kv_cache[1, src_blk].permute(1, 0, 2)

    return out


def _make_kv_cache(
    num_gpu_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: str,
    seed: int = 0,
) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(
        2,
        num_gpu_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=dtype,
        device=device,
    )


def _make_pinned(
    num_cpu_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.zeros(
        2,
        num_cpu_blocks,
        num_kv_heads,
        block_size,
        head_size,
        dtype=dtype,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Parametrisation
# ---------------------------------------------------------------------------

DTYPES = [torch.float16, torch.bfloat16, torch.float32]
HEAD_SIZES = [64, 128]  # powers of 2 — exercised vectorised path
BLOCK_SIZES = [8, 16]
NUM_KV_HEADS = [1, 8]
SEEDS = [0, 42]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS)
@pytest.mark.parametrize("seed", SEEDS)
def test_gather_kv_blocks_correctness(
    dtype: torch.dtype,
    head_size: int,
    block_size: int,
    num_kv_heads: int,
    seed: int,
) -> None:
    """Verify that gather_kv_blocks matches the reference (K packed, V HND)."""
    num_gpu_blocks = 32
    num_logical_blocks = 4
    num_cpu_blocks = 16
    device = "cuda:0"

    kv_cache = _make_kv_cache(
        num_gpu_blocks, block_size, num_kv_heads, head_size, dtype, device, seed
    )

    torch.manual_seed(seed + 1)
    src_block_list = torch.randperm(num_gpu_blocks)[:num_logical_blocks].tolist()
    dst_block_list = list(range(num_logical_blocks))

    pinned_kv = _make_pinned(num_cpu_blocks, block_size, num_kv_heads, head_size, dtype)

    ops.gather_kv_blocks(
        kv_cache, src_block_list, pinned_kv, dst_block_list, num_logical_blocks
    )
    torch.cuda.synchronize()

    ref = _reference_gather(
        kv_cache.cpu(),
        src_block_list,
        dst_block_list,
        num_kv_heads,
        head_size,
        block_size,
    )

    for b, dst_blk in enumerate(dst_block_list):
        actual = pinned_kv[:, dst_blk]
        expected = ref[:, dst_blk]
        assert torch.allclose(actual, expected, atol=0.0, rtol=0.0), (
            f"Mismatch at block {b} (src={src_block_list[b]}, dst={dst_blk}), "
            f"dtype={dtype}, head_size={head_size}, block_size={block_size}, "
            f"num_kv_heads={num_kv_heads}"
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_gather_kv_blocks_multiple_requests(dtype: torch.dtype) -> None:
    """Multiple back-to-back calls sharing one KV cache and one pinned buffer."""
    device = "cuda:0"
    block_size, num_kv_heads, head_size = 16, 4, 64
    kv_cache = _make_kv_cache(64, block_size, num_kv_heads, head_size, dtype, device)
    pinned_kv = _make_pinned(32, block_size, num_kv_heads, head_size, dtype)

    req_specs = [
        (list(range(0, 4)), list(range(0, 4))),
        (list(range(4, 8)), list(range(4, 8))),
    ]
    for src_list, dst_list in req_specs:
        ops.gather_kv_blocks(kv_cache, src_list, pinned_kv, dst_list, len(src_list))

    torch.cuda.synchronize()

    for src_list, dst_list in req_specs:
        ref = _reference_gather(
            kv_cache.cpu(), src_list, dst_list, num_kv_heads, head_size, block_size
        )
        for b, dst_blk in enumerate(dst_list):
            assert torch.allclose(
                pinned_kv[:, dst_blk], ref[:, dst_blk], atol=0.0, rtol=0.0
            ), f"Multi-request mismatch: block {b}, dst={dst_blk}"


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_gather_kv_blocks_zero_blocks() -> None:
    """num_blocks=0 must not crash."""
    device = "cuda:0"
    kv_cache = _make_kv_cache(8, 16, 4, 64, torch.float16, device)
    pinned_kv = _make_pinned(8, 16, 4, 64, torch.float16)
    ops.gather_kv_blocks(kv_cache, [], pinned_kv, [], 0)
    torch.cuda.synchronize()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_gather_kv_blocks_non_aligned_head_size() -> None:
    """head_size=48 triggers scalar path for both K and V."""
    device = "cuda:0"
    head_size, block_size, num_kv_heads = 48, 8, 2
    dtype = torch.float16
    kv_cache = _make_kv_cache(16, block_size, num_kv_heads, head_size, dtype, device)
    pinned_kv = _make_pinned(8, block_size, num_kv_heads, head_size, dtype)

    src_list = [0, 1, 2]
    dst_list = [0, 1, 2]
    ops.gather_kv_blocks(kv_cache, src_list, pinned_kv, dst_list, len(src_list))
    torch.cuda.synchronize()

    ref = _reference_gather(
        kv_cache.cpu(), src_list, dst_list, num_kv_heads, head_size, block_size
    )
    for b, dst_blk in enumerate(dst_list):
        assert torch.allclose(
            pinned_kv[:, dst_blk], ref[:, dst_blk], atol=0.0, rtol=0.0
        ), f"Scalar-path mismatch at block {b}"


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_gather_kv_blocks_python_list_block_tables() -> None:
    """Block tables as Python lists; wrapper builds int32 tensors for the CUDA op."""
    device = "cuda:0"
    kv_cache = _make_kv_cache(16, 16, 4, 64, torch.float16, device)
    pinned_kv = _make_pinned(8, 16, 4, 64, torch.float16)

    ops.gather_kv_blocks(kv_cache, [0, 1, 2], pinned_kv, [0, 1, 2], 3)
    torch.cuda.synchronize()

    ref = _reference_gather(kv_cache.cpu(), [0, 1, 2], [0, 1, 2], 4, 64, 16)
    for b in range(3):
        assert torch.allclose(pinned_kv[:, b], ref[:, b], atol=0.0, rtol=0.0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_gather_kv_blocks_does_not_clobber_other_slots() -> None:
    """Unwritten CPU slots must remain zero after a partial transfer."""
    device = "cuda:0"
    num_cpu_blocks = 8
    kv_cache = _make_kv_cache(16, 16, 2, 64, torch.float16, device)
    pinned_kv = _make_pinned(num_cpu_blocks, 16, 2, 64, torch.float16)

    ops.gather_kv_blocks(kv_cache, [0, 1], pinned_kv, [0, 1], 2)
    torch.cuda.synchronize()

    for slot in range(2, num_cpu_blocks):
        assert pinned_kv[:, slot].eq(0).all(), f"Slot {slot} was unexpectedly modified"


# ---------------------------------------------------------------------------
# Tests — gather_kv_blocks_batched
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_batched_gather_correctness(dtype: torch.dtype) -> None:
    """Batched gather of multiple requests matches per-request reference."""
    device = "cuda:0"
    block_size, num_kv_heads, head_size = 16, 4, 128
    num_gpu_blocks, num_cpu_blocks = 64, 32

    kv_cache = _make_kv_cache(
        num_gpu_blocks, block_size, num_kv_heads, head_size, dtype, device
    )
    pinned_kv = _make_pinned(num_cpu_blocks, block_size, num_kv_heads, head_size, dtype)

    req_specs = [
        ([0, 5, 10], [0, 1, 2]),
        ([20, 21, 22, 23], [3, 4, 5, 6]),
        ([40], [7]),
    ]

    src_tables = [s for s, _ in req_specs]
    dst_tables = [d for _, d in req_specs]
    num_blocks_per_req = torch.tensor([len(s) for s, _ in req_specs], dtype=torch.int32)

    ops.gather_kv_blocks_batched(
        kv_cache, src_tables, pinned_kv, dst_tables, num_blocks_per_req
    )
    torch.cuda.synchronize()

    kv_cpu = kv_cache.cpu()
    for src_list, dst_list in req_specs:
        ref = _reference_gather(
            kv_cpu, src_list, dst_list, num_kv_heads, head_size, block_size
        )
        for b, dst_blk in enumerate(dst_list):
            assert torch.allclose(
                pinned_kv[:, dst_blk], ref[:, dst_blk], atol=0.0, rtol=0.0
            ), (
                f"Batched mismatch: req src={src_list}, block idx {b}, "
                f"dst={dst_blk}, dtype={dtype}"
            )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_batched_gather_single_request() -> None:
    """Batched with one request must match the non-batched call."""
    device = "cuda:0"
    block_size, num_kv_heads, head_size = 16, 2, 64
    dtype = torch.float16

    kv_cache = _make_kv_cache(32, block_size, num_kv_heads, head_size, dtype, device)
    pinned_single = _make_pinned(8, block_size, num_kv_heads, head_size, dtype)
    pinned_batch = _make_pinned(8, block_size, num_kv_heads, head_size, dtype)

    src_list, dst_list = [1, 3, 7], [0, 1, 2]

    ops.gather_kv_blocks(kv_cache, src_list, pinned_single, dst_list, len(src_list))

    ops.gather_kv_blocks_batched(
        kv_cache,
        [src_list],
        pinned_batch,
        [dst_list],
        torch.tensor([len(src_list)], dtype=torch.int32),
    )
    torch.cuda.synchronize()

    for dst_blk in dst_list:
        assert torch.equal(pinned_single[:, dst_blk], pinned_batch[:, dst_blk]), (
            f"Single vs batched mismatch at dst={dst_blk}"
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_batched_gather_empty() -> None:
    """Zero requests must be a no-op without crashing."""
    device = "cuda:0"
    kv_cache = _make_kv_cache(8, 16, 2, 64, torch.float16, device)
    pinned_kv = _make_pinned(8, 16, 2, 64, torch.float16)

    ops.gather_kv_blocks_batched(
        kv_cache, [], pinned_kv, [], torch.tensor([], dtype=torch.int32)
    )
    torch.cuda.synchronize()
    assert pinned_kv.eq(0).all(), "Pinned buffer should be untouched"
