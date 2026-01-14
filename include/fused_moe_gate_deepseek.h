/*
    Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
*/

// 从256个专家中选择top8个专家，输出专家权重和索引。
int fused_moe_gate_deepseek(
    torch::Tensor& gating_outputs, //[bs, num_experts], dtype=bf16
    torch::Tensor& correction_bias, //[num_experts], dtype=bf16
    torch::Tensor& out_routing_weights, //[bs, num_selected_experts], dtype=float
    torch::Tensor& out_selected_experts, //[bs, num_selected_experts], dtype=int32
    int topk,
    bool renormalize,
    int num_expert_group,
    int topk_group,
    std::optional<int> num_fused_shared_experts,
    std::optional<float> routed_scaling_factor,
    std::optional<int> moegate_type
);
