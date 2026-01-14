"""
Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""

import os
import sys
import torch
import unittest
from typing import Callable, NamedTuple, Optional
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
from mcoplib.op import fused_moe_gate_deepseek

def biased_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    routed_scaling_factor: float = 1.0,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    # gating_output_float = gating_output.float()
    # correction_bias_f = correction_bias.float()

    scores = gating_output.sigmoid()
    num_token = scores.shape[0]
    # print("Number of tokens: {}".format(num_token))
    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    print("scores_for_choice: {}".format(scores_for_choice.shape), scores_for_choice.dtype)
    # print("scores_for_choice: ", scores_for_choice)
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )  # [n, n_group]
    # print("group_scores: ", group_scores)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    # print("group_idx: ", group_idx)
    group_mask = torch.zeros_like(group_scores)
    # print("group_mask: ", group_mask)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    # print("group_mask: ", group_mask)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    # print("score_mask: ", score_mask)
    # print("score_mask[0]: ", score_mask[0])  # [n, e]
    tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
    # print("tmp_scores: ", tmp_scores)  # [n, e]
    # print("tmp_scores[0]: ", tmp_scores[0])  # [n, e]
    _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    # print("topk_ids: ", topk_ids)
    topk_weights = scores.gather(1, topk_ids)
    # print("scores: ", scores[0])
    # print("topk_weights: ", topk_weights)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights * routed_scaling_factor # must multiply the scaling factor

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

def show_error(golden, v, tag="DIFF ERROR"):
    errors = torch.abs(golden - v)

    errors_max = torch.max(errors)
    errors_ave = torch.sum(errors) / errors.numel()

    max_idx_flat = torch.argmax(errors)
    max_idx = torch.unravel_index(max_idx_flat, errors.shape)

    golden_val = golden[max_idx]
    v_val = v[max_idx]

    print(f"{tag}: error_max={errors_max}, error_ave={errors_ave}, max_error_idx={max_idx}")

def moe_gate_func(q_len, num_experts, topk, num_expert_group, top_k_group, renormalize, test_dtype, scale_factor):
    gating_output = torch.rand(q_len, num_experts, dtype=test_dtype).cuda()
    correction_bias = torch.rand(num_experts, dtype=test_dtype).cuda()
    out_routing_weights = torch.zeros(q_len, topk, dtype=torch.float).cuda()
    out_selected_experts = torch.zeros(q_len, topk, dtype=torch.int32).cuda()

    golden_routing_weights, golden_selected_experts = biased_grouped_topk(gating_output, gating_output, correction_bias, topk, renormalize, num_expert_group, top_k_group, scale_factor)
    # routing_weights, selected_experts = 
    fused_moe_gate_deepseek(gating_output, correction_bias, out_routing_weights, out_selected_experts, topk, renormalize, num_expert_group, top_k_group, None, scale_factor, 0)

    sorted_golden_routing = torch.sort(golden_routing_weights, dim=1).values  # 按行排序
    sorted_routing = torch.sort(out_routing_weights, dim=1).values
    show_error(sorted_golden_routing, sorted_routing, "DIFF ERROR OF SORTED ROUTING_WEIGHTS")
    assert torch.allclose(sorted_golden_routing, sorted_routing, rtol=1e-03, atol=1e-03, equal_nan=False)
    sorted_golden_experts = torch.sort(golden_selected_experts, dim=1).values  # 按行排序
    sorted_selected_experts = torch.sort(out_selected_experts, dim=1).values
    show_error(sorted_golden_experts, sorted_selected_experts, "DIFF ERROR OF SORTED SELECTED_EXPERTS")
    assert torch.equal(sorted_golden_experts, sorted_selected_experts)

def moe_gate_func_diff_dtype(q_len, num_experts, topk, num_expert_group, top_k_group, renormalize, test_dtype, bias_dtype, scale_factor):
    gating_output = torch.rand(q_len, num_experts, dtype=test_dtype).cuda()
    correction_bias = torch.rand(num_experts, dtype=bias_dtype).cuda()
    out_routing_weights = torch.zeros(q_len, topk, dtype=torch.float).cuda()
    out_selected_experts = torch.zeros(q_len, topk, dtype=torch.int32).cuda()

    golden_routing_weights, golden_selected_experts = biased_grouped_topk(gating_output, gating_output, correction_bias, topk, renormalize, num_expert_group, top_k_group, scale_factor)
    fused_moe_gate_deepseek(gating_output, correction_bias, out_routing_weights, out_selected_experts, topk, renormalize, num_expert_group, top_k_group, None, scale_factor, 0)

    sorted_golden_routing = torch.sort(golden_routing_weights, dim=1).values  # 按行排序
    sorted_routing = torch.sort(out_routing_weights, dim=1).values
    show_error(sorted_golden_routing, sorted_routing, "DIFF ERROR OF SORTED ROUTING_WEIGHTS")
    assert torch.allclose(sorted_golden_routing, sorted_routing, rtol=1e-03, atol=1e-03, equal_nan=False)
    sorted_golden_experts = torch.sort(golden_selected_experts, dim=1).values  # 按行排序
    sorted_selected_experts = torch.sort(out_selected_experts, dim=1).values
    show_error(sorted_golden_experts, sorted_selected_experts, "DIFF ERROR OF SORTED SELECTED_EXPERTS")
    assert torch.equal(sorted_golden_experts, sorted_selected_experts)

class TestMoeGate(unittest.TestCase):
    def test_moe_gate_256_experts_bfloat16(self):
        moe_gate_func(q_len=16, num_experts=256, topk=8, num_expert_group=8, top_k_group=4, renormalize=True, test_dtype=torch.bfloat16, scale_factor=1.0)

    def test_moe_gate_256_experts_float16(self):
        moe_gate_func(q_len=16, num_experts=256, topk=8, num_expert_group=8, top_k_group=4, renormalize=True, test_dtype=torch.float16, scale_factor=1.5)

    def test_moe_gate_256_experts_float(self):
        moe_gate_func(q_len=16, num_experts=256, topk=8, num_expert_group=8, top_k_group=4, renormalize=True, test_dtype=torch.float32, scale_factor=3.0)

    def test_moe_gate_320_experts_bfloat16(self):
        moe_gate_func(q_len=16, num_experts=320, topk=8, num_expert_group=1, top_k_group=1, renormalize=True, test_dtype=torch.bfloat16, scale_factor=1.0)

    def test_moe_gate_320_experts_float16(self):
        moe_gate_func(q_len=16, num_experts=320, topk=8, num_expert_group=1, top_k_group=1, renormalize=True, test_dtype=torch.float16, scale_factor=5.5)

    def test_moe_gate_320_experts_float(self):
        moe_gate_func(q_len=16, num_experts=320, topk=8, num_expert_group=1, top_k_group=1, renormalize=True, test_dtype=torch.float32, scale_factor=10.5)

    def test_moe_gate_384_experts_bfloat16(self):
        moe_gate_func(q_len=16, num_experts=384, topk=8, num_expert_group=1, top_k_group=1, renormalize=True, test_dtype=torch.bfloat16, scale_factor=1.0)

    def test_moe_gate_384_experts_float16(self):
        moe_gate_func(q_len=16, num_experts=384, topk=8, num_expert_group=1, top_k_group=1, renormalize=True, test_dtype=torch.float16, scale_factor=5.5)

    def test_moe_gate_384_experts_float(self):
        moe_gate_func(q_len=16, num_experts=384, topk=8, num_expert_group=1, top_k_group=1, renormalize=True, test_dtype=torch.float32, scale_factor=10.5)

    def test_moe_gate_448_experts_bfloat16(self):
        moe_gate_func(q_len=16, num_experts=448, topk=8, num_expert_group=1, top_k_group=1, renormalize=True, test_dtype=torch.bfloat16, scale_factor=1.0)

    def test_moe_gate_448_experts_float16(self):
        moe_gate_func(q_len=16, num_experts=448, topk=8, num_expert_group=1, top_k_group=1, renormalize=True, test_dtype=torch.float16, scale_factor=5.5)

    def test_moe_gate_448_experts_float(self):
        moe_gate_func(q_len=16, num_experts=448, topk=8, num_expert_group=1, top_k_group=1, renormalize=True, test_dtype=torch.float32, scale_factor=10.5)

    def test_moe_gate_256_experts_bfloat16_float32(self):
        moe_gate_func_diff_dtype(q_len=16, num_experts=256, topk=8, num_expert_group=8, top_k_group=4, renormalize=True, test_dtype=torch.bfloat16, bias_dtype=torch.float32, scale_factor=1.0)

    def test_moe_gate_256_experts_float16_float32(self):
        moe_gate_func_diff_dtype(q_len=16, num_experts=256, topk=8, num_expert_group=8, top_k_group=4, renormalize=True, test_dtype=torch.float16, bias_dtype=torch.float32, scale_factor=1.5)

    def test_moe_gate_256_experts_float32_float32(self):
        moe_gate_func_diff_dtype(q_len=16, num_experts=256, topk=8, num_expert_group=8, top_k_group=4, renormalize=True, test_dtype=torch.float32, bias_dtype=torch.float32, scale_factor=3.0)

if __name__ == '__main__':
    unittest.main()
