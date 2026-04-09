# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for csrc/shadow_kernels.cu — gather_kv_blocks and scatter_kv_blocks.

Both ops share the unified ``transfer_kv_blocks_kernel``:
  gather (IS_GATHER=true)  : GPU NHD → pinned CPU (K packed tile, V HND)
  scatter (IS_GATHER=false): pinned CPU → GPU NHD

Layout conventions
------------------
GPU KV cache (NHD):  (2, num_gpu_blocks, block_size, num_kv_heads, head_size)
Pinned host (HND):   (2, num_cpu_blocks, num_kv_heads, block_size, head_size)
  V (side 1): row-major over head_size
  K (side 0): column-major packed tile — flat index ``token + d * block_size``
               within each (block, head) region of size block_size * head_size
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
    """Pure-Python reference for ``gather_kv_blocks``.

    K (side 0): NHD → column-major packed tile (flat index token + d*block_size
                within each (block, head) region).
    V (side 1): NHD → HND via permute(1, 0, 2).
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

        # Key: NHD → column-major packed tile.
        src_k = kv_cache[0, src_blk]  # [block_size, num_kv_heads, head_size]
        flat_k = out[0, dst_blk].reshape(-1)
        for tok in range(block_size):
            for h in range(num_kv_heads):
                for d in range(head_size):
                    flat_k[h * (block_size * head_size) + tok + d * block_size] = src_k[
                        tok, h, d
                    ]

        # Value: NHD → HND.
        out[1, dst_blk] = kv_cache[1, src_blk].permute(1, 0, 2)

    return out


def _reference_scatter(
    pinned_kv: torch.Tensor,
    src_block_table: list[int],
    dst_block_table: list[int],
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    num_gpu_blocks: int,
) -> torch.Tensor:
    """Pure-Python reference for ``scatter_kv_blocks``.

    Inverse of ``_reference_gather``:
      K (side 0): column-major packed tile → NHD.
      V (side 1): HND → NHD via permute(1, 0, 2).
    """
    num_blocks = len(src_block_table)
    out = torch.zeros(
        2,
        num_gpu_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=pinned_kv.dtype,
    )
    for b in range(num_blocks):
        src_blk = src_block_table[b]
        dst_blk = dst_block_table[b]

        # Value: HND → NHD.
        out[1, dst_blk] = pinned_kv[1, src_blk].permute(1, 0, 2)

        # Key: column-major packed tile → NHD.
        flat_k = pinned_kv[0, src_blk].reshape(-1)
        for tok in range(block_size):
            for h in range(num_kv_heads):
                for d in range(head_size):
                    out[0, dst_blk, tok, h, d] = flat_k[
                        h * (block_size * head_size) + tok + d * block_size
                    ]

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
HEAD_SIZES = [48, 64, 128]  # 48 exercises the scalar fallback path
BLOCK_SIZES = [8, 16]
NUM_KV_HEADS = [1, 8]
SEEDS = [0, 42]


# ---------------------------------------------------------------------------
# Tests — gather_kv_blocks
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
    """Verify gather_kv_blocks matches the reference (K packed, V HND)."""
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
        assert torch.allclose(
            pinned_kv[:, dst_blk], ref[:, dst_blk], atol=0.0, rtol=0.0
        ), (
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


# ---------------------------------------------------------------------------
# Tests — scatter_kv_blocks
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS)
@pytest.mark.parametrize("seed", SEEDS)
def test_scatter_kv_blocks_correctness(
    dtype: torch.dtype,
    head_size: int,
    block_size: int,
    num_kv_heads: int,
    seed: int,
) -> None:
    """scatter_kv_blocks matches the pure-Python reference unpack.

    Pinned buffers are seeded via ``_reference_gather`` (pure Python, independent
    of the gather kernel) so the scatter result can be verified directly without
    relying on the gather kernel being correct.
    """
    num_gpu_blocks = 32
    num_logical_blocks = 4
    num_cpu_blocks = 16
    device = "cuda:0"

    kv_src = _make_kv_cache(
        num_gpu_blocks, block_size, num_kv_heads, head_size, dtype, device, seed
    )

    torch.manual_seed(seed + 1)
    gpu_src_blocks = torch.randperm(num_gpu_blocks)[:num_logical_blocks].tolist()
    cpu_slots = list(range(num_logical_blocks))
    torch.manual_seed(seed + 2)
    gpu_dst_blocks = torch.randperm(num_gpu_blocks)[:num_logical_blocks].tolist()

    # Build pinned buffer via pure-Python reference (independent of gather kernel).
    ref_pinned = _reference_gather(
        kv_src.cpu(), gpu_src_blocks, cpu_slots, num_kv_heads, head_size, block_size
    )
    pinned_kv = _make_pinned(num_cpu_blocks, block_size, num_kv_heads, head_size, dtype)
    pinned_kv[:, cpu_slots] = ref_pinned[:, cpu_slots]

    kv_dst = torch.zeros_like(kv_src)
    ops.scatter_kv_blocks(
        kv_dst, cpu_slots, pinned_kv, gpu_dst_blocks, num_logical_blocks
    )
    torch.cuda.synchronize()

    ref_gpu = _reference_scatter(
        ref_pinned,
        cpu_slots,
        gpu_dst_blocks,
        num_kv_heads,
        head_size,
        block_size,
        num_gpu_blocks,
    )
    for i, dst_blk in enumerate(gpu_dst_blocks):
        assert torch.allclose(
            kv_dst[:, dst_blk].cpu(), ref_gpu[:, dst_blk], atol=0.0, rtol=0.0
        ), (
            f"Mismatch at block {i} (cpu_slot={cpu_slots[i]}, gpu_dst={dst_blk}), "
            f"dtype={dtype}, head_size={head_size}, block_size={block_size}, "
            f"num_kv_heads={num_kv_heads}"
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("head_size", [48, 64, 128])
def test_scatter_kv_blocks_roundtrip(dtype: torch.dtype, head_size: int) -> None:
    """gather (GPU→pinned) then scatter (pinned→GPU) reproduces the source."""
    device = "cuda:0"
    block_size, num_kv_heads = 16, 4
    num_gpu_blocks, num_cpu_blocks = 64, 32
    num_logical_blocks = 6

    kv_src = _make_kv_cache(
        num_gpu_blocks, block_size, num_kv_heads, head_size, dtype, device, seed=123
    )
    pinned = _make_pinned(num_cpu_blocks, block_size, num_kv_heads, head_size, dtype)

    src_blocks = torch.randperm(num_gpu_blocks)[:num_logical_blocks].tolist()
    cpu_slots = list(range(num_logical_blocks))

    ops.gather_kv_blocks(kv_src, src_blocks, pinned, cpu_slots, num_logical_blocks)
    torch.cuda.synchronize()

    kv_dst = torch.zeros_like(kv_src)
    dst_blocks = torch.randperm(num_gpu_blocks)[:num_logical_blocks].tolist()
    ops.scatter_kv_blocks(kv_dst, cpu_slots, pinned, dst_blocks, num_logical_blocks)
    torch.cuda.synchronize()

    for i in range(num_logical_blocks):
        got = kv_dst[:, dst_blocks[i]].cpu()
        expected = kv_src[:, src_blocks[i]].cpu()
        assert torch.allclose(got, expected, atol=0.0, rtol=0.0), (
            f"Roundtrip mismatch at i={i}, src_gpu={src_blocks[i]}, "
            f"dst_gpu={dst_blocks[i]}, dtype={dtype}, head_size={head_size}"
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_scatter_kv_blocks_zero_blocks() -> None:
    """num_blocks=0 must not crash and must leave kv_cache unchanged."""
    device = "cuda:0"
    kv_cache = torch.zeros(2, 8, 16, 4, 64, dtype=torch.float16, device=device)
    pinned_kv = _make_pinned(8, 16, 4, 64, torch.float16)
    ops.scatter_kv_blocks(kv_cache, [], pinned_kv, [], 0)
    torch.cuda.synchronize()
    assert kv_cache.eq(0).all(), "kv_cache must be untouched"


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_scatter_kv_blocks_does_not_clobber_other_slots() -> None:
    """Unwritten GPU blocks must remain zero after a partial scatter."""
    device = "cuda:0"
    num_gpu_blocks = 16
    block_size, num_kv_heads, head_size = 16, 2, 64
    dtype = torch.float16

    kv_src = _make_kv_cache(
        num_gpu_blocks, block_size, num_kv_heads, head_size, dtype, device
    )
    pinned = _make_pinned(8, block_size, num_kv_heads, head_size, dtype)

    # Gather two blocks so pinned is non-trivial.
    ops.gather_kv_blocks(kv_src, [0, 1], pinned, [0, 1], 2)
    torch.cuda.synchronize()

    kv_dst = torch.zeros_like(kv_src)
    written_blocks = [3, 7]
    ops.scatter_kv_blocks(kv_dst, [0, 1], pinned, written_blocks, 2)
    torch.cuda.synchronize()

    for blk in range(num_gpu_blocks):
        if blk not in written_blocks:
            assert kv_dst[:, blk].eq(0).all(), (
                f"GPU block {blk} was unexpectedly modified"
            )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_scatter_kv_blocks_python_list_block_tables() -> None:
    """Block tables passed as Python lists must be accepted without error."""
    device = "cuda:0"
    block_size, num_kv_heads, head_size = 16, 4, 64
    dtype = torch.float16

    kv_src = _make_kv_cache(16, block_size, num_kv_heads, head_size, dtype, device)
    pinned = _make_pinned(8, block_size, num_kv_heads, head_size, dtype)

    ops.gather_kv_blocks(kv_src, [0, 1, 2], pinned, [0, 1, 2], 3)
    torch.cuda.synchronize()

    kv_dst = torch.zeros_like(kv_src)
    ops.scatter_kv_blocks(kv_dst, [0, 1, 2], pinned, [0, 1, 2], 3)
    torch.cuda.synchronize()

    for blk in range(3):
        assert torch.allclose(
            kv_dst[:, blk].cpu(), kv_src[:, blk].cpu(), atol=0.0, rtol=0.0
        ), f"Python-list scatter mismatch at block {blk}"


# ---------------------------------------------------------------------------
# Tests — scatter_kv_blocks_batched
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_batched_scatter_correctness(dtype: torch.dtype) -> None:
    """Batched scatter of multiple requests matches per-request reference."""
    device = "cuda:0"
    block_size, num_kv_heads, head_size = 16, 4, 128
    num_gpu_blocks, num_cpu_blocks = 64, 32

    kv_src = _make_kv_cache(
        num_gpu_blocks, block_size, num_kv_heads, head_size, dtype, device
    )
    pinned = _make_pinned(num_cpu_blocks, block_size, num_kv_heads, head_size, dtype)

    req_specs = [
        ([0, 5, 10], [0, 1, 2]),  # (gpu_src_blocks, cpu_slots)
        ([20, 21, 22, 23], [3, 4, 5, 6]),
        ([40], [7]),
    ]
    gpu_dst_specs = [
        [50, 51, 52],
        [53, 54, 55, 56],
        [57],
    ]

    # Populate pinned via individual gather calls.
    for (gpu_srcs, cpu_slots), _ in zip(req_specs, gpu_dst_specs):
        ops.gather_kv_blocks(kv_src, gpu_srcs, pinned, cpu_slots, len(gpu_srcs))
    torch.cuda.synchronize()

    # Batched scatter.
    cpu_src_tables = [cpu_slots for _, cpu_slots in req_specs]
    gpu_dst_tables = gpu_dst_specs
    num_blocks_per_req = torch.tensor([len(s) for s, _ in req_specs], dtype=torch.int32)
    kv_dst = torch.zeros_like(kv_src)
    ops.scatter_kv_blocks_batched(
        kv_dst, cpu_src_tables, pinned, gpu_dst_tables, num_blocks_per_req
    )
    torch.cuda.synchronize()

    # Each scattered destination block must match the original GPU source block.
    for (gpu_srcs, _), gpu_dsts in zip(req_specs, gpu_dst_specs):
        for i, (src_blk, dst_blk) in enumerate(zip(gpu_srcs, gpu_dsts)):
            assert torch.allclose(
                kv_dst[:, dst_blk].cpu(), kv_src[:, src_blk].cpu(), atol=0.0, rtol=0.0
            ), (
                f"Batched scatter mismatch: req gpu_src={gpu_srcs}, block {i}, "
                f"gpu_dst={dst_blk}, dtype={dtype}"
            )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_batched_scatter_single_request() -> None:
    """Batched scatter with one request must match the non-batched call."""
    device = "cuda:0"
    block_size, num_kv_heads, head_size = 16, 2, 64
    dtype = torch.float16

    kv_src = _make_kv_cache(32, block_size, num_kv_heads, head_size, dtype, device)
    pinned = _make_pinned(8, block_size, num_kv_heads, head_size, dtype)

    cpu_slots, gpu_dst_list = [0, 1, 2], [10, 11, 12]

    ops.gather_kv_blocks(kv_src, [1, 3, 7], pinned, cpu_slots, len(cpu_slots))
    torch.cuda.synchronize()

    kv_single = torch.zeros_like(kv_src)
    ops.scatter_kv_blocks(kv_single, cpu_slots, pinned, gpu_dst_list, len(cpu_slots))

    kv_batch = torch.zeros_like(kv_src)
    ops.scatter_kv_blocks_batched(
        kv_batch,
        [cpu_slots],
        pinned,
        [gpu_dst_list],
        torch.tensor([len(cpu_slots)], dtype=torch.int32),
    )
    torch.cuda.synchronize()

    for dst_blk in gpu_dst_list:
        assert torch.equal(kv_single[:, dst_blk], kv_batch[:, dst_blk]), (
            f"Single vs batched scatter mismatch at gpu dst={dst_blk}"
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
def test_batched_scatter_empty() -> None:
    """Zero requests must be a no-op without crashing."""
    device = "cuda:0"
    kv_cache = torch.zeros(2, 8, 16, 2, 64, dtype=torch.float16, device=device)
    pinned_kv = _make_pinned(8, 16, 2, 64, torch.float16)

    ops.scatter_kv_blocks_batched(
        kv_cache, [], pinned_kv, [], torch.tensor([], dtype=torch.int32)
    )
    torch.cuda.synchronize()
    assert kv_cache.eq(0).all(), "kv_cache should be untouched"
