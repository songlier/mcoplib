#include <ATen/ATen.h>

void recv_from_attention_node_post_process(at::Tensor hidden_status, at::Tensor topk_ids, at::Tensor topk_weights, 
                                        at::Tensor ori_index, at::Tensor new_index, at::Tensor deep_hidden_status,
                                        at::Tensor deepep_topk_weights,  at::Tensor expert_cnt, at::Tensor valid_idx_size, const int begin_expert_id, 
                                        const int num_local_experts, const int max_index_size, const int work_count);