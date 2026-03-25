#include <ATen/ATen.h>
void per_token_cast_to_fp8(
    torch::Tensor& out,
    torch::Tensor& scale,   
    torch::Tensor const& input);