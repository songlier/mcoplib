#include <ATen/ATen.h>
void rms_norm_dynamic_per_token_quant_custom(
    at::Tensor& out,           // [..., hidden_size]
    at::Tensor const& input,   // [..., hidden_size]
    at::Tensor const& weight,  // [hidden_size]
    at::Tensor& scales,        // [num_tokens]
    double const var_epsilon,     // Variance epsilon used in norm calculation
    c10::optional<at::Tensor> scale_ub, c10::optional<at::Tensor> residual);