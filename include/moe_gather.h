#include <ATen/ATen.h>

void moe_gather(
    at::Tensor scatter_tokens,
    at::Tensor scatter_tokens_offset, 
    at::Tensor convergent_tokens);