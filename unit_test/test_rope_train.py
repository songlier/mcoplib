# import torch

# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)

# def apply_rotary_pos_emb_forward(input, sin, cos, cumsum_len, batch_size, head_dim, cut_head_dim=0):
#     skip_head_dim = head_dim - cut_head_dim
#     input_rope = input[:, :, skip_head_dim:]
#     input_nope = input[:, :, :skip_head_dim]
#     input_float = input_rope.float()
#     sin_float = sin.float()
#     cos_float = cos.float()
#     out_float = torch.zeros_like(input_float)
#     query_start = 0
#     for i in range(batch_size):
#         seq_len = cumsum_len[i + 1] - cumsum_len[i]
#         input_tmp = input_float[query_start : query_start + seq_len, ...].permute(1, 0, 2)
#         out_float[query_start : query_start + seq_len, ...] = (
#             input_tmp * cos_float[:seq_len, :] + rotate_half(input_tmp) * sin_float[:seq_len, :]
#         ).permute(1, 0, 2)
#         query_start += seq_len
#     out = out_float.to(input.dtype)
#     out = torch.cat((input_nope, out), dim=-1)
#     return out

import torch
import numpy as np
import random
from mcoplib.op import rotary_pos_emb_forward, rotary_pos_emb_backward
import pytest

def rotate_backward(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x2, -x1), dim=-1)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def gen_embeds(head_dim, base=10000, max_seq_len=2048):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2,device='cuda').float() * (1 / head_dim)))
    t = torch.arange(max_seq_len,device='cuda')
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb


def apply_rotary_pos_emb_backward(
    input, sin, cos, cumsum_len, q_len, batch_size, head_num
):
    output = torch.empty_like(input)
    if input.dtype == torch.half:
        for i in range(batch_size):
            for j in range(q_len[i]):
                for k in range(head_num):
                    output[cumsum_len[i] + j, k, :] = input[cumsum_len[i] + j, k, :] * (
                        cos[j, :] + 1e-10
                    ) + rotate_backward(
                        (input[cumsum_len[i] + j, k, :] * sin[j, :])
                    )
    else:
        input_float = input.float()
        sin_float = sin.float()
        cos_float = cos.float()
        output_float = torch.empty_like(input_float)
        for i in range(batch_size):
            for j in range(q_len[i]):
                for k in range(head_num):
                    output_float[cumsum_len[i] + j, k, :] = input_float[
                        cumsum_len[i] + j, k, :
                    ] * (cos_float[j, :]) + rotate_backward(
                        (input_float[cumsum_len[i] + j, k, :] * sin_float[j, :])
                    )
        output = output_float.bfloat16()
    return output


def apply_rotary_pos_emb_forward(
    input, sin, cos, cumsum_len, q_len, batch_size, head_num
):
    output = torch.empty_like(input)
    if input.dtype == torch.half:
        for i in range(batch_size):
            for j in range(q_len[i]):
                for k in range(head_num):
                    output[cumsum_len[i] + j, k, :] = (
                        input[cumsum_len[i] + j, k, :] * cos[j, :]
                        + rotate_half(input[cumsum_len[i] + j, k, :]) * sin[j, :]
                    )
    else:
        input_float = input.float()
        sin_float = sin.float()
        cos_float = cos.float()
        output_float = torch.empty_like(input_float)
        for i in range(batch_size):
            for j in range(q_len[i]):
                for k in range(head_num):
                    output_float[cumsum_len[i] + j, k, :] = (
                        input_float[cumsum_len[i] + j, k, :] * cos_float[j, :]
                        + rotate_half(input_float[cumsum_len[i] + j, k, :])
                        * sin_float[j, :]
                    )
        output = output_float
    return output


@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.half, torch.bfloat16])
@pytest.mark.parametrize(
    "batch_size, head_num, seq_len, head_dim, rope_cut_head_dim",
    [
        (1, 11, 1, 128, 128),
    ],
)
def test_rope_backward(batch_size, head_num, seq_len, head_dim, rope_cut_head_dim, dtype):
    start_seqlen = max(1, seq_len / 2 - 1)
    q_len = torch.arange(
        start_seqlen, start_seqlen + batch_size, 1, dtype=torch.int64,device='cuda'
    )
    embeds = gen_embeds(rope_cut_head_dim)
    cumsum_len = torch.zeros(batch_size + 1, dtype=torch.int64,device='cuda')
    cumsum_len[1:] = torch.cumsum(q_len, dim=-1)
    input = torch.normal(0, 1, (cumsum_len[-1], head_num, head_dim),device='cuda').to(dtype)
    npu_output_fused = rotary_pos_emb_backward(
        input,
        embeds.sin().to(dtype),
        embeds.cos().to(dtype),
        cumsum_len,
        batch_size,
        rope_cut_head_dim
    )

    npu_output_unfused = apply_rotary_pos_emb_backward(
        input,
        embeds.sin().to(dtype),
        embeds.cos().to(dtype),
        cumsum_len,
        q_len,
        batch_size,
        head_num,
    )
    assert torch.allclose(
        npu_output_unfused.to(dtype), npu_output_fused.to(dtype), atol=1e-2, rtol=1e-2
    )

# 
@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.float, torch.half, torch.bfloat16])
@pytest.mark.parametrize(
    "batch_size, head_num, seq_len, head_dim, rope_cut_head_dim",
    [
        (1, 11, 1, 128, 128),
    ],
)
def test_rope_forward(batch_size, head_num, seq_len, head_dim, rope_cut_head_dim, dtype):
    start_seqlen = max(1, seq_len / 2 - 1)
    q_len = torch.arange(
        start_seqlen, start_seqlen + batch_size, 1, dtype=torch.int64,device='cuda'
    )
    embeds = gen_embeds(rope_cut_head_dim)
    cumsum_len = torch.zeros(batch_size + 1, dtype=torch.int64,device='cuda')
    cumsum_len[1:] = torch.cumsum(q_len, dim=-1)
    input = torch.normal(0, 1, (cumsum_len[-1], head_num, head_dim), device='cuda').to(dtype)
    npu_output_fused = rotary_pos_emb_forward(
        input,
        embeds.sin().to(dtype),
        embeds.cos().to(dtype),
        cumsum_len,
        batch_size,
        rope_cut_head_dim
    )
    npu_output_unfused = apply_rotary_pos_emb_forward(
        input,
        embeds.sin().to(dtype),
        embeds.cos().to(dtype),
        cumsum_len,
        q_len,
        batch_size,
        head_num,
    )

    assert torch.allclose(
        npu_output_unfused.to(dtype), npu_output_fused.to(dtype), atol=1e-2, rtol=1e-2
    )
