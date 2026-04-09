/*
 * Unified GPU↔pinned-host KV block transfer kernel.
 *
 * A single template kernel handles both directions:
 *   IS_GATHER = true  → GPU NHD → pinned CPU (gather)
 *   IS_GATHER = false → pinned CPU → GPU NHD (scatter)
 *
 * Layout conventions
 * ------------------
 * GPU KV cache (NHD):
 *     kv_cache[kv_side, block_id, token, head, dim]
 *     shape: (2, num_blocks, block_size, num_kv_heads, head_size)
 *
 * Pinned host tensor (HND + packed-K):
 *     pinned_kv[kv_side, block_id, head, token, dim]
 *     shape: (2, max_num_blocks, num_kv_heads, block_size, head_size)
 *   - V (kv_side == 1): HND row-major over head_size — vectorised store/load.
 *   - K (kv_side == 0): column-major packed tile required by
 *     cpu_attn_reshape_and_cache. Within each (block, head) tile the element
 *     at (token, dim) sits at flat index  token + dim * block_size.
 *
 * Vectorised path (head_size divisible by VEC_ELEMS = 16/sizeof(scalar_t))
 * -------------------------------------------------------------------------
 * Work items iterate over (token, head, vec-chunk v).
 * The NHD side is always contiguous in dim → uint4 load or store.
 * V packed side is also contiguous in dim → uint4 load or store.
 * K packed side is strided (stride = block_size per element) → individual
 * loads or stores of VEC_ELEMS elements.
 *
 * Grid: (num_logical_blocks, 2) — one thread block per (block, kv_side).
 * Pinned host memory is accessed via cudaHostGetDevicePointer.
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "dispatch_utils.h"
#include "shadow.h"

namespace vllm {
namespace shadow {

/*
 * IS_GATHER == true  : NHD (gpu_ptr) → packed (pinned_ptr)
 * IS_GATHER == false : packed (pinned_ptr) → NHD (gpu_ptr)
 *
 * gpu_block_table   : block indices into kv_cache (NHD, GPU side)
 * pinned_block_table: block indices into pinned_kv (packed, CPU side)
 */
template <typename scalar_t, bool IS_GATHER>
__global__ void transfer_kv_blocks_kernel(
    scalar_t* __restrict__ kv_cache, scalar_t* __restrict__ pinned_kv,
    const int32_t* __restrict__ gpu_block_table,
    const int32_t* __restrict__ pinned_block_table, const int num_kv_heads,
    const int head_size, const int block_size, const int64_t gpu_kv_stride,
    const int64_t gpu_block_stride, const int64_t pinned_kv_stride,
    const int64_t pinned_block_stride) {
  const int block_idx = blockIdx.x;
  const int kv_side = blockIdx.y;

  const int32_t gpu_blk = gpu_block_table[block_idx];
  const int32_t pinned_blk = pinned_block_table[block_idx];

  scalar_t* gpu_ptr = kv_cache + kv_side * gpu_kv_stride +
                      static_cast<int64_t>(gpu_blk) * gpu_block_stride;
  scalar_t* pinned_ptr = pinned_kv + kv_side * pinned_kv_stride +
                         static_cast<int64_t>(pinned_blk) * pinned_block_stride;

  constexpr int VEC_ELEMS = 16 / sizeof(scalar_t);
  const bool aligned = (head_size % VEC_ELEMS == 0);

  if (aligned) {
    const int head_vecs = head_size / VEC_ELEMS;
    const int total_vecs = block_size * num_kv_heads * head_vecs;

    for (int i = threadIdx.x; i < total_vecs; i += blockDim.x) {
      const int pair = i / head_vecs;
      const int v = i % head_vecs;
      const int token = pair / num_kv_heads;
      const int head = pair % num_kv_heads;

      // NHD layout: dim is innermost → always contiguous → uint4 load/store.
      const int64_t nhd_off =
          static_cast<int64_t>(token) * num_kv_heads * head_size +
          static_cast<int64_t>(head) * head_size +
          static_cast<int64_t>(v) * VEC_ELEMS;

      scalar_t elems[VEC_ELEMS];

      if constexpr (IS_GATHER) {
        // Load VEC_ELEMS from NHD (contiguous).
        *reinterpret_cast<uint4*>(elems) =
            *reinterpret_cast<const uint4*>(gpu_ptr + nhd_off);

        if (kv_side == 1) {
          // Value dst: HND row-major → single uint4 store.
          const int64_t packed_off =
              static_cast<int64_t>(head) * block_size * head_size +
              static_cast<int64_t>(token) * head_size +
              static_cast<int64_t>(v) * VEC_ELEMS;
          *reinterpret_cast<uint4*>(pinned_ptr + packed_off) =
              *reinterpret_cast<const uint4*>(elems);
        } else {
          // Key dst: column-major packed tile — scatter VEC_ELEMS elements.
          const int64_t packed_base =
              static_cast<int64_t>(head) * block_size * head_size +
              static_cast<int64_t>(token);
#pragma unroll
          for (int e = 0; e < VEC_ELEMS; ++e) {
            pinned_ptr[packed_base + static_cast<int64_t>(v * VEC_ELEMS + e) *
                                         block_size] = elems[e];
          }
        }
      } else {
        // Scatter direction: load from packed, store to NHD.
        if (kv_side == 1) {
          // Value src: HND row-major → single uint4 load.
          const int64_t packed_off =
              static_cast<int64_t>(head) * block_size * head_size +
              static_cast<int64_t>(token) * head_size +
              static_cast<int64_t>(v) * VEC_ELEMS;
          *reinterpret_cast<uint4*>(elems) =
              *reinterpret_cast<const uint4*>(pinned_ptr + packed_off);
        } else {
          // Key src: column-major packed tile — gather VEC_ELEMS elements.
          const int64_t packed_base =
              static_cast<int64_t>(head) * block_size * head_size +
              static_cast<int64_t>(token);
#pragma unroll
          for (int e = 0; e < VEC_ELEMS; ++e) {
            elems[e] = pinned_ptr[packed_base +
                                  static_cast<int64_t>(v * VEC_ELEMS + e) *
                                      block_size];
          }
        }
        // Store VEC_ELEMS to NHD (contiguous).
        *reinterpret_cast<uint4*>(gpu_ptr + nhd_off) =
            *reinterpret_cast<const uint4*>(elems);
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

      const int64_t nhd_off =
          static_cast<int64_t>(token) * num_kv_heads * head_size +
          static_cast<int64_t>(head) * head_size + d;

      // V packed: HND row-major; K packed: column-major tile (token +
      // d*block_size).
      const int64_t packed_off =
          (kv_side == 1) ? static_cast<int64_t>(head) * block_size * head_size +
                               static_cast<int64_t>(token) * head_size + d
                         : static_cast<int64_t>(head) * block_size * head_size +
                               static_cast<int64_t>(token) +
                               static_cast<int64_t>(d) * block_size;

      if constexpr (IS_GATHER) {
        pinned_ptr[packed_off] = gpu_ptr[nhd_off];
      } else {
        gpu_ptr[nhd_off] = pinned_ptr[packed_off];
      }
    }
  }
}

/*
 * Shared validation and kernel launch for gather/scatter.
 *
 * kv_cache_bt   : block indices into kv_cache  (GPU NHD side)
 * pinned_kv_bt  : block indices into pinned_kv (CPU packed side)
 */
template <bool IS_GATHER>
static void launch_transfer_kv_blocks(torch::Tensor& kv_cache,
                                      torch::Tensor& kv_cache_bt,
                                      torch::Tensor& pinned_kv,
                                      torch::Tensor& pinned_kv_bt,
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
  TORCH_CHECK(kv_cache_bt.dtype() == torch::kInt32,
              "kv_cache block table must be int32");
  TORCH_CHECK(pinned_kv_bt.dtype() == torch::kInt32,
              "pinned_kv block table must be int32");
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

  auto gpu_bt =
      kv_cache_bt.is_cuda() ? kv_cache_bt : kv_cache_bt.to(kv_cache.device());
  auto pin_bt = pinned_kv_bt.is_cuda() ? pinned_kv_bt
                                       : pinned_kv_bt.to(kv_cache.device());

  const int64_t gpu_kv_stride = kv_cache.stride(0);
  const int64_t gpu_block_stride = kv_cache.stride(1);
  const int64_t pinned_kv_stride = pinned_kv.stride(0);
  const int64_t pinned_block_stride = pinned_kv.stride(1);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(kv_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(num_blocks, 2);
  dim3 block(256);

  VLLM_DISPATCH_FLOATING_TYPES(
      kv_cache.scalar_type(),
      IS_GATHER ? "gather_kv_blocks" : "scatter_kv_blocks", [&] {
        transfer_kv_blocks_kernel<scalar_t, IS_GATHER>
            <<<grid, block, 0, stream>>>(
                kv_cache.data_ptr<scalar_t>(),
                static_cast<scalar_t*>(pinned_dev_ptr),
                gpu_bt.data_ptr<int32_t>(), pin_bt.data_ptr<int32_t>(),
                num_kv_heads, head_size, block_size, gpu_kv_stride,
                gpu_block_stride, pinned_kv_stride, pinned_block_stride);
      });
}

// ---------------------------------------------------------------------------
// Public C++ entry points (torch op schema unchanged)
// ---------------------------------------------------------------------------

/*
 * gather_kv_blocks: GPU NHD → pinned CPU
 *   src_block_table : kv_cache  (GPU) block indices
 *   dst_block_table : pinned_kv (CPU) block indices
 */
void gather_kv_blocks(torch::Tensor& kv_cache, torch::Tensor& src_block_table,
                      torch::Tensor& pinned_kv, torch::Tensor& dst_block_table,
                      int64_t num_blocks) {
  launch_transfer_kv_blocks<true>(kv_cache, src_block_table, pinned_kv,
                                  dst_block_table, num_blocks);
}

/*
 * scatter_kv_blocks: pinned CPU → GPU NHD
 *   src_block_table : pinned_kv (CPU) block indices
 *   dst_block_table : kv_cache  (GPU) block indices
 */
void scatter_kv_blocks(torch::Tensor& kv_cache, torch::Tensor& src_block_table,
                       torch::Tensor& pinned_kv, torch::Tensor& dst_block_table,
                       int64_t num_blocks) {
  launch_transfer_kv_blocks<false>(kv_cache, dst_block_table, pinned_kv,
                                   src_block_table, num_blocks);
}

}  // namespace shadow
}  // namespace vllm
