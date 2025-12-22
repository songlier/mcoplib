#include <ATen/ATen.h>

void moe_swiglu_dynamic_quantize(at::Tensor scatter_tokens, at::Tensor smooth_scale, at::Tensor experts_tokens_start, at::Tensor experts_tokens_count,
    at::Tensor& y, at::Tensor& per_tokens_scale, int total_experts_num);


