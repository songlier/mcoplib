#include <ATen/ATen.h>

at::Tensor fused_rope_fwd(at::Tensor qkv, at::Tensor cos, at::Tensor sin, c10::optional<at::Tensor> indexes, bool force_bf16_attn = false);

at::Tensor fused_rope_bwd(at::Tensor qkv, at::Tensor cos, at::Tensor sin, c10::optional<at::Tensor> indexes, bool force_bf16_attn = false);

