import os
import sys
import torch
import pytest
import numpy as np
import mcoplib
from mcoplib import op as ops
from mcoplib.marlin_utils import marlin_quantize 
import torch.nn.functional as F


@pytest.mark.parametrize("size_m", [64])
@pytest.mark.parametrize("size_k", [2048])  # we only support 2048 now
@pytest.mark.parametrize("size_n", [7168])
@pytest.mark.parametrize("is_k_full", [True])
@pytest.mark.parametrize("act_order", [False])
@torch.inference_mode()
def test_gptq_marlin_gemm(
    size_m,
    size_k,
    size_n,
    is_k_full,
    act_order,
    ):

    device="cuda"
    cpu_device="cpu"
    num_bits:int = 4
    group_size:int = 64

    scalar_t=torch.bfloat16

    weight = (torch.rand(size_k, size_n) - 0.5) / 10
    x = (torch.rand(size_m, size_k) - 0.5) / 10


    weight = weight.to(device)
    w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
        weight, num_bits, group_size, act_order
    )
    print(f"m={size_m}, k={size_k}, n={size_n}, g_idx={g_idx}, sort_indices = {sort_indices}")

    ept = torch.empty(1)

    x_fp16 = x.to(scalar_t).to(device)
    w_ref = w_ref.to(scalar_t).to(device)
    bsz_tensor = torch.tensor([size_m], dtype=torch.int).to(device)
    print(f"bsz_tensor = {bsz_tensor.cpu().numpy()}")
    marlin_s = marlin_s.to(scalar_t).to(torch.half)
    x_fp16_half = x_fp16.to(torch.half)

    out = ops.gptq_marlin_gemm(
        x_fp16_half,
        marlin_q_w,
        marlin_s,
        g_idx,
        sort_indices,
        ept,
        num_bits,
        bsz_tensor,
        x_fp16_half.shape[0],
        size_n,
        x_fp16_half.shape[-1],
        -1,
        is_k_full,
        scalar_t,
        True)

    # out_vllm = torch.ops._C.gptq_marlin_gemm(
    #     x_fp16_half,
    #     marlin_q_w,
    #     marlin_s,
    #     g_idx,
    #     sort_indices,
    #     ept,
    #     num_bits,
    #     bsz_tensor,
    #     x_fp16_half.shape[0],
    #     size_n,
    #     x_fp16_half.shape[-1],
    #     -1,
    #     is_k_full,
    #     15)
    # print("out.shape", out.shape, "dtype", out.dtype, "device", out.device)
    # print("out_vllm.shape", out_vllm.shape, "dtype", out_vllm.dtype, "device", out_vllm.device) 

    # cos_sim = F.cosine_similarity(out, out_vllm)  # 默认dim=1
    # mean = cos_sim.mean()
    # print("Cosine Similarity:", mean.item())
    golden = (x_fp16 @ w_ref)
    errors = torch.abs(golden.reshape(out.shape).to(out.device) - out)
    error_max = torch.max(errors)
    error_ave = torch.sum(errors)/errors.numel()
    print("Doing quanted gemm torch vs llvm errors = error_max = {}, error_ave = {}".format(error_max, error_ave))

    assert error_ave < 0.05


if __name__ == "__main__":
    pytest.main([__file__])