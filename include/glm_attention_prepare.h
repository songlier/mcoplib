
#include <ATen/ATen.h>

void FusedAttentionPrepare(torch::Tensor qkv,
                            torch::Tensor weight,
                            torch::Tensor positions,
                            torch::Tensor out_q,
                            torch::Tensor out_kv,
                            const int num_heads,
                            const int num_kv_heads,
                            const int head_dim,
                            const float base,
                            const int max_position_embeddings,
                            const float rms_norm_eps,
                            const float partial_rotary_factor);