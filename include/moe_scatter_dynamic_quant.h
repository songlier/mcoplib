#include <ATen/ATen.h>

void moe_scatter_dynamic_quant(at::Tensor hidden_status, at::Tensor selected_experts, at::Tensor moe_weights, at::Tensor smooth_scale,
                                at::Tensor scatter_tokens, at::Tensor scatter_per_token_scale, at::Tensor scatter_tokens_offset, at::Tensor experts_token_count, at::Tensor experts_token_start,
                                const int experts_per_rank, const int shared_experts_per_rank, const int shared_tokens_per_sp);