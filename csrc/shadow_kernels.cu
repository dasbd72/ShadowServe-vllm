/*
 * Block-level GPU-to-pinned-host KV gather.
 *
 * GPU KV cache (NHD) per-block layout:
 *     kv_cache[kv_side, block_id, token, head, dim]
 *     shape: (2, num_blocks, block_size, num_kv_heads, head_size)
 *
 * Pinned host tensor shape (same for both sides):
 *     pinned_kv[kv_side, block_id, head, token, dim]
 *     shape: (2, max_num_blocks, num_kv_heads, block_size, head_size)
 *
 * Work items are divided over (pair, v) where pair = (token, head) and v is
 * a VEC_ELEMS-wide chunk index within head_dim.  One uint4 load covers
 * VEC_ELEMS contiguous dim elements from the source (NHD dim is innermost).
 *
 * Value (kv_side == 1): dst dim axis is also contiguous (HND row-major over
 * head_dim) → single uint4 store.  Non-aligned head_size falls back to scalar.
 *
 * Key (kv_side == 0): dst uses the column-major packed tile layout required by
 * cpu_attn_reshape_and_cache / cpu_attention_with_kv_cache
 * (csrc/cpu/cpu_attn_vec.hpp). Within each (dst_block, head) tile the element
 * at (token, dim) sits at flat index  token + dim * block_size  (stride
 * block_size along dim, stride 1 along token).  The uint4 load of VEC_ELEMS
 * contiguous dim values is still valid; each element is then stored with a
 * scalar write to its scattered dst address (stride = block_size per element).
 * Non-aligned head_size uses all-scalar.
 *
 * Grid: (num_logical_blocks, 2) — one thread block per (block, kv_side).
 * Writes to pinned host memory via cudaHostGetDevicePointer.
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "dispatch_utils.h"
#include "shadow.h"

namespace vllm {
namespace shadow {

template <typename scalar_t>
__global__ void gather_kv_blocks_kernel(
    const scalar_t* __restrict__ kv_cache, scalar_t* __restrict__ pinned_kv,
    const int32_t* __restrict__ src_block_table,
    const int32_t* __restrict__ dst_block_table, const int num_kv_heads,
    const int head_size, const int block_size, const int64_t src_kv_stride,
    const int64_t src_block_stride, const int64_t dst_kv_stride,
    const int64_t dst_block_stride) {
  const int block_idx = blockIdx.x;
  const int kv_side = blockIdx.y;

  const int32_t src_blk = src_block_table[block_idx];
  const int32_t dst_blk = dst_block_table[block_idx];

  const scalar_t* __restrict__ src =
      kv_cache + kv_side * src_kv_stride +
      static_cast<int64_t>(src_blk) * src_block_stride;
  scalar_t* dst = pinned_kv + kv_side * dst_kv_stride +
                  static_cast<int64_t>(dst_blk) * dst_block_stride;

  constexpr int VEC_ELEMS = 16 / sizeof(scalar_t);
  const bool aligned = (head_size % VEC_ELEMS == 0);

  if (aligned) {
    // Vectorised path — one uint4 load per work item (VEC_ELEMS consecutive
    // dim elements from the NHD src; dim is innermost → always contiguous).
    const int head_vecs = head_size / VEC_ELEMS;
    const int total_vecs = block_size * num_kv_heads * head_vecs;

    for (int i = threadIdx.x; i < total_vecs; i += blockDim.x) {
      const int pair = i / head_vecs;
      const int v = i % head_vecs;
      const int token = pair / num_kv_heads;
      const int head = pair % num_kv_heads;

      const int64_t src_off =
          static_cast<int64_t>(token) * num_kv_heads * head_size +
          static_cast<int64_t>(head) * head_size +
          static_cast<int64_t>(v) * VEC_ELEMS;

      // Single 16-byte load from contiguous src.
      scalar_t elems[VEC_ELEMS];
      *reinterpret_cast<uint4*>(elems) =
          *reinterpret_cast<const uint4*>(src + src_off);

      if (kv_side == 1) {
        // Value: dst dim is also contiguous (HND row-major) → single uint4
        // store.
        const int64_t dst_off =
            static_cast<int64_t>(head) * block_size * head_size +
            static_cast<int64_t>(token) * head_size +
            static_cast<int64_t>(v) * VEC_ELEMS;
        *reinterpret_cast<uint4*>(dst + dst_off) =
            *reinterpret_cast<const uint4*>(elems);
      } else {
        // Key: dst uses column-major packed tile (dim-stride = block_size).
        // Scatter VEC_ELEMS elements to strided dst positions.
        const int64_t dst_base =
            static_cast<int64_t>(head) * block_size * head_size +
            static_cast<int64_t>(token);
#pragma unroll
        for (int e = 0; e < VEC_ELEMS; ++e) {
          dst[dst_base + static_cast<int64_t>(v * VEC_ELEMS + e) * block_size] =
              elems[e];
        }
      }
    }
  } else {
    // Scalar fallback for non-standard head sizes.
    const int total_elems = block_size * num_kv_heads * head_size;
    for (int i = threadIdx.x; i < total_elems; i += blockDim.x) {
      const int pair = i / head_size;
      const int d = i % head_size;
      const int token = pair / num_kv_heads;
      const int head = pair % num_kv_heads;

      const int64_t src_off =
          static_cast<int64_t>(token) * num_kv_heads * head_size +
          static_cast<int64_t>(head) * head_size + d;

      int64_t dst_off;
      if (kv_side == 1) {
        // Value: HND row-major.
        dst_off = static_cast<int64_t>(head) * block_size * head_size +
                  static_cast<int64_t>(token) * head_size + d;
      } else {
        // Key: column-major packed tile (token + d * block_size per head).
        dst_off = static_cast<int64_t>(head) * block_size * head_size +
                  static_cast<int64_t>(token) +
                  static_cast<int64_t>(d) * block_size;
      }
      dst[dst_off] = src[src_off];
    }
  }
}

void gather_kv_blocks(torch::Tensor& kv_cache, torch::Tensor& src_block_table,
                      torch::Tensor& pinned_kv, torch::Tensor& dst_block_table,
                      int64_t num_blocks) {
  if (num_blocks == 0) return;

  TORCH_CHECK(kv_cache.is_cuda(), "kv_cache must be on CUDA");
  TORCH_CHECK(kv_cache.dim() == 5,
              "kv_cache must be [2, N, block_size, num_kv_heads, head_size]");
  TORCH_CHECK(pinned_kv.is_cpu() && pinned_kv.is_pinned(),
              "pinned_kv must be pinned CPU memory");
  TORCH_CHECK(pinned_kv.dim() == 5,
              "pinned_kv must be [2, M, num_kv_heads, block_size, head_size]");
  TORCH_CHECK(kv_cache.dtype() == pinned_kv.dtype(),
              "kv_cache and pinned_kv dtype mismatch");
  TORCH_CHECK(src_block_table.dtype() == torch::kInt32,
              "src_block_table must be int32");
  TORCH_CHECK(dst_block_table.dtype() == torch::kInt32,
              "dst_block_table must be int32");
  TORCH_CHECK(kv_cache.size(0) == 2 && pinned_kv.size(0) == 2,
              "first dimension must be 2 (key + value)");
  TORCH_CHECK(kv_cache.is_contiguous(), "kv_cache must be contiguous");
  TORCH_CHECK(pinned_kv.is_contiguous(), "pinned_kv must be contiguous");

  const int block_size = kv_cache.size(2);
  const int num_kv_heads = kv_cache.size(3);
  const int head_size = kv_cache.size(4);

  TORCH_CHECK(pinned_kv.size(2) == num_kv_heads,
              "num_kv_heads mismatch between kv_cache and pinned_kv");
  TORCH_CHECK(pinned_kv.size(3) == block_size,
              "block_size mismatch between kv_cache and pinned_kv");
  TORCH_CHECK(pinned_kv.size(4) == head_size,
              "head_size mismatch between kv_cache and pinned_kv");

  void* pinned_dev_ptr = nullptr;
  C10_CUDA_CHECK(
      cudaHostGetDevicePointer(&pinned_dev_ptr, pinned_kv.data_ptr(), 0));

  auto src_bt = src_block_table.is_cuda()
                    ? src_block_table
                    : src_block_table.to(kv_cache.device());
  auto dst_bt = dst_block_table.is_cuda()
                    ? dst_block_table
                    : dst_block_table.to(kv_cache.device());

  const int64_t src_kv_stride = kv_cache.stride(0);
  const int64_t src_block_stride = kv_cache.stride(1);
  const int64_t dst_kv_stride = pinned_kv.stride(0);
  const int64_t dst_block_stride = pinned_kv.stride(1);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(kv_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(num_blocks, 2);
  dim3 block(256);

  VLLM_DISPATCH_FLOATING_TYPES(kv_cache.scalar_type(), "gather_kv_blocks", [&] {
    gather_kv_blocks_kernel<scalar_t><<<grid, block, 0, stream>>>(
        kv_cache.data_ptr<scalar_t>(), static_cast<scalar_t*>(pinned_dev_ptr),
        src_bt.data_ptr<int32_t>(), dst_bt.data_ptr<int32_t>(), num_kv_heads,
        head_size, block_size, src_kv_stride, src_block_stride, dst_kv_stride,
        dst_block_stride);
  });
}

}  // namespace shadow
}  // namespace vllm
