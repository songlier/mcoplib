#include <ATen/ATen.h>

at::Tensor fused_gelu_fwd(at::Tensor input, at::Tensor bias);
at::Tensor fused_gelu_bwd(at::Tensor input, at::Tensor input1, at::Tensor bias);