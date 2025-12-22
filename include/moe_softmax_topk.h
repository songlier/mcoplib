#include <ATen/ATen.h>

void moe_softmax_topk(
    at::Tensor topk_weights,                // [num_tokens, topk]
    at::Tensor topk_indices,                // [num_tokens, topk]
    at::Tensor gating_output,
    const bool pre_softmax);