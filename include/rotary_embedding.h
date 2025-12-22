#include <ATen/ATen.h>

void rotary_embedding(at::Tensor packed_qkv, // [num_tokens, total_head_num, head_dim]
                        at::Tensor q_len, at::Tensor accum_q_lens, at::Tensor cache_lens, at::Tensor cos,
                        at::Tensor sin,  at::Tensor output, const int q_head_num, const int kv_head_num, const int rope_offset = 0);