#include <ATen/ATen.h>

void rms_norm_dynamic_per_token_quant(
    torch::Tensor& out,
    torch::Tensor const& input,   
    torch::Tensor const& weight,
    torch::Tensor const& smooth_scale,  
    torch::Tensor& scales,        
    double const var_epsilon,
    torch::Tensor& after_res,
    torch::Tensor& after_norm,
    std::optional<at::Tensor> residual);

void head_rms_norm(torch::Tensor& out, torch::Tensor const& hidden_states, torch::Tensor const &weight, double const var_epsilon, int head_offset, int head_norm);

void rms_norm(
    torch::Tensor& out,           
    torch::Tensor const& input,   
    torch::Tensor const& weight,  
    double const var_epsilon,
    std::optional<at::Tensor> after_res,
    std::optional<at::Tensor> residual,
    bool rms_div
);