#include <ATen/ATen.h>

at::Tensor fused_bias_swiglu_fwd(at::Tensor input);

at::Tensor fused_bias_swiglu_bwd(at::Tensor input, at::Tensor grad_output);