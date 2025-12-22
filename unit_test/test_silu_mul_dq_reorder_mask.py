import argparse
import torch
import triton
import triton.language as tl
from mcoplib.profiler import profiler
# from vllm._custom_ops import scaled_int8_quant, fused_silu_mul_dq_reorder_quant, fused_silu_mul_reorder
from mcoplib.op import fused_silu_mul_dq_reorder_quant
from typing import Optional, Tuple, Union

def as_float32_tensor(x: Union[float, torch.tensor]) -> torch.tensor:
    return torch.as_tensor(x, dtype=torch.float32, device='cuda')

def ref_dynamic_per_token_quant(x: torch.tensor,
                                quant_dtype: torch.dtype,
                                scale_ub: Optional[torch.tensor] = None) \
        -> Tuple[torch.tensor, torch.tensor]:

    assert quant_dtype in [torch.int8]
    
    qtype_traits = torch.iinfo(quant_dtype) if quant_dtype == torch.int8 \
            else torch.finfo(quant_dtype)
    qtype_traits_max =  qtype_traits.max
    qtype_traits_min =  qtype_traits.min
    qtype_max = as_float32_tensor(qtype_traits_max)
    s_1 = as_float32_tensor(1.0)
    s_512 = as_float32_tensor(512.0)

    # For fp8, in order to match the cuda kernel output, we have to do exactly
    # the same operations as in the corresponding fp8 kernel to prevent
    # rounding errors.

    # Compute scales
    x_token_max, _ = x.abs().max(dim=-1)
    x_token_max = as_float32_tensor(x_token_max)
    if scale_ub is not None:
        x_token_max = x_token_max.clamp(max=scale_ub)
    scales = (x_token_max / qtype_max)[:, None]

    # Quant
    if quant_dtype == torch.int8:
        iscales = as_float32_tensor(s_1 / scales)
        torch_out = as_float32_tensor(x) * iscales
        torch_out = torch_out.round()
        torch_out = torch_out.clamp(qtype_traits_min,
                                    qtype_traits_max).to(quant_dtype)
    else:
        assert quant_dtype == FP8_DTYPE
        min_scaling_factor = s_1 / (qtype_max * s_512)
        scales = scales.clamp(min=min_scaling_factor)
        torch_out = as_float32_tensor(x) / scales
        torch_out = torch_out.clamp(qtype_traits_min,
                                    qtype_traits_max).to(quant_dtype)

    return torch_out, scales

def safe_cosine_similarity(x1, x2,  eps=1e-8):
    # 添加极小值避免除零
    dot_product = torch.sum(x1 * x2)
    norm_x1 = torch.norm(x1, p=2).clamp(min=eps)
    norm_x2 = torch.norm(x2, p=2).clamp(min=eps)
    return dot_product / (norm_x1 * norm_x2)

def show_error(golden, v, tag="DIFF ERROR"):
    # import pdb
    # pdb.set_trace()
    errors = torch.abs(golden - v)
    err_max = torch.max(errors)
    err_ave = torch.sum(errors) / errors.numel()
    max_idx_flat = torch.argmax(errors)
    max_idx = torch.unravel_index(max_idx_flat, errors.shape)
    # golden_val= golden[max_idx]
    # v_val = v[max_idx]
    print(f"{tag}: error_max={err_max} error_ave={err_ave}, max_error_idx={max_idx}")
    
@triton.jit
def silu_and_mul_triton_kernel(
    gateup_output,
    down_input,
    hidden_size,
    reorder_topk_ids,
    scales,
    start_expert_id,
    end_expert_id,
    BLOCK_SIZE: tl.constexpr,
):
    InDtype = gateup_output.dtype.element_ty
    OutDtype = down_input.dtype.element_ty

    half_hidden_size = hidden_size // 2

    pid = tl.program_id(0)
    expert_id = tl.load(reorder_topk_ids + pid)
    if expert_id >= start_expert_id and expert_id <= end_expert_id:
        gateup_output_ptr = gateup_output + pid * hidden_size
        gate_output_ptr = gateup_output_ptr
        up_output_ptr = gateup_output_ptr + half_hidden_size
        down_input_ptr = down_input + pid * half_hidden_size

        if scales is not None:
            scale = tl.load(scales + expert_id - start_expert_id)
            scale = (1 / scale).to(InDtype)
        else:
            scale = 1

        for start_offset in tl.range(0, half_hidden_size, BLOCK_SIZE):
            offset = start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offset < half_hidden_size

            gate_output = tl.load(gate_output_ptr + offset, mask=mask).to(tl.float32)
            up_output = tl.load(up_output_ptr + offset, mask=mask)

            # silu & mul & quantize
            gate_output = gate_output * tl.sigmoid(gate_output)
            gate_output = gate_output.to(InDtype)

            silu_mul_output = gate_output * up_output * scale
            silu_mul_output = silu_mul_output.to(OutDtype)
            tl.store(down_input_ptr + offset, silu_mul_output, mask=mask)

def main(mode):
    gateup_output_clone = gateup_output.clone()

    silu_and_mul_triton_kernel[(gateup_output.shape[0],)](
        gateup_output,
        down_input,
        gateup_output.shape[1],
        reorder_topk_ids,
        w2_input_scale,
        0,
        num_experts_per_partition - 1,
        BLOCK_SIZE=512,
    )
    # down_input = fused_silu_mul_reorder(gateup_output, reorder_topk_ids, torch.empty(0), 0, num_experts_per_partition - 1)
    # import pdb 
    # pdb.set_trace()
    # a, scales , _  = scaled_int8_quant(down_input)
    a, scales = ref_dynamic_per_token_quant(down_input, torch.int8)
    
    down_inputc = torch.empty((gateup_output.shape[0], gateup_output.shape[1] // 2), device='cuda', dtype=torch.int8)
    down_input_scalec = torch.empty((gateup_output.shape[0],1), device='cuda', dtype=torch.float32)
    fused_silu_mul_dq_reorder_quant(down_inputc,down_input_scalec,
        gateup_output_clone,
        reorder_topk_ids,
        torch.empty(0),
        0,
        num_experts_per_partition - 1
    )
    show_error(a, down_inputc, "quant tensor")
    show_error(scales, down_input_scalec, "scale")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="profile", choices=["profile", "acc"])
    args = parser.parse_args()

    num_experts_per_partition = 8
    
    w2_input_scale = torch.ones(num_experts_per_partition, dtype=torch.float32, device="cuda")

    num_tokens = [512, 1024, 2048, 3200]
    for num_token in num_tokens:
        gateup_output = torch.randn((num_token, 4096), device="cuda", dtype=torch.bfloat16)
        down_input =torch.empty(
            gateup_output.shape[0],
            gateup_output.shape[1] // 2,
            device=gateup_output.device,
            dtype=torch.bfloat16
        )
        reorder_topk_ids = torch.randint(0, 8, (num_token,), device="cuda", dtype=torch.int64)
        main(args.mode)



                

    


    
