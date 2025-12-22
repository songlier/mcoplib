#include <ATen/ATen.h>

void send_to_attention_node_pre_process(at::Tensor moe_hidden_status, at::Tensor deepep_topk_weights, at::Tensor ori_index, 
                                            at::Tensor new_index, at::Tensor output, at::Tensor valid_idx_size, const int max_index_size);