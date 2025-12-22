#include <ATen/ATen.h>
void store_kv_cache_cuda_interface(
    torch::Tensor packed_qkv,
    torch::Tensor q_lens,
    torch::Tensor accum_q_lens,
    torch::Tensor cache_lens,
    torch::Tensor cache_slot_ids,
    torch::Tensor &k_cache,
    torch::Tensor &v_cache,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    int batch_size,
    int q_head_num,
    int kv_head_num
);