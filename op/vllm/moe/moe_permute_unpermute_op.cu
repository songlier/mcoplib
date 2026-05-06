#include <c10/core/ScalarType.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include "core/registration.h"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

void moe_permute(const torch::Tensor& input,
                 const torch::Tensor& topk_ids,  // [n_token, topk]
                 const torch::Tensor& token_expert_indices,
                 const std::optional<torch::Tensor>& expert_map,
                 int64_t n_expert, int64_t n_local_expert, int64_t topk,
                 torch::Tensor& permuted_input,
                 torch::Tensor& expert_first_token_offset,
                 torch::Tensor& inv_permuted_idx,  // [n_token, topk]
                 torch::Tensor& permuted_idx) {      // [permute_size]
  DEBUG_TRACE_PARAMS(input, topk_ids, token_expert_indices, expert_map, n_expert, n_local_expert, topk, permuted_input, expert_first_token_offset, inv_permuted_idx, permuted_idx);
  DEBUG_DUMP_PARAMS(input, topk_ids, token_expert_indices, expert_map, n_expert, n_local_expert, topk, permuted_input, expert_first_token_offset, inv_permuted_idx, permuted_idx);
  TORCH_CHECK(false, "moe_unpermute is not supported on MACA");
}

void moe_unpermute(
    const torch::Tensor& permuted_hidden_states,  // [n_token * topk, hidden]
    const torch::Tensor& topk_weights,            // [n_token, topk]
    const torch::Tensor& inv_permuted_idx,        // [n_token, topk]
    const std::optional<torch::Tensor>&
        expert_first_token_offset,  // [n_local_expert+1]
    int64_t topk,
    torch::Tensor& hidden_states  // [n_token, hidden]
) {
  DEBUG_TRACE_PARAMS(permuted_hidden_states, topk_weights, inv_permuted_idx, expert_first_token_offset, topk, hidden_states);
  DEBUG_DUMP_PARAMS(permuted_hidden_states, topk_weights, inv_permuted_idx, expert_first_token_offset, topk, hidden_states);
  TORCH_CHECK(false, "moe_unpermute is not supported on MACA");
}

void shuffle_rows(const torch::Tensor& input_tensor,
                  const torch::Tensor& dst2src_map,
                  torch::Tensor& output_tensor) {
  DEBUG_TRACE_PARAMS(input_tensor, dst2src_map, output_tensor);
  DEBUG_DUMP_PARAMS(input_tensor, dst2src_map, output_tensor);
  TORCH_CHECK(false, "shuffle_rows is not supported on MACA");
}

bool moe_permute_unpermute_supported() { return false; }

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("moe_permute", &moe_permute);
  m.impl("moe_unpermute", &moe_unpermute);
}
