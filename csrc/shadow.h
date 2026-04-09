#pragma once

#include <torch/all.h>

namespace vllm {
namespace shadow {

void gather_kv_blocks(torch::Tensor& kv_cache, torch::Tensor& src_block_table,
                      torch::Tensor& pinned_kv, torch::Tensor& dst_block_table,
                      int64_t num_blocks);

}  // namespace shadow
}  // namespace vllm
