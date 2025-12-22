#include <ATen/ATen.h>

at::Tensor fused_repeat_kv_fwd(at::Tensor input, int q_num_head, int kv_num_head, int head_dim);
at::Tensor fused_repeat_kv_bwd(at::Tensor input, int q_num_head, int kv_num_head, int partition);