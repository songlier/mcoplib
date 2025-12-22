#include <ATen/ATen.h>

torch::Tensor rotary_pos_emb_forward(
    torch::Tensor input, 
    torch::Tensor sin, 
    torch::Tensor cos, 
    torch::Tensor cumsum_len, 
    int batch_size,
    int cut_head_dim = 0
);

torch::Tensor rotary_pos_emb_backward(
    torch::Tensor input, 
    torch::Tensor sin, 
    torch::Tensor cos, 
    torch::Tensor cumsum_len, 
    int batch_size,
    int cut_head_dim = 0
);