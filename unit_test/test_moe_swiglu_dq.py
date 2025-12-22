import torch
from mcoplib.op import moe_swiglu_dynamic_quantize

import torch

def smooth_per_token_dynamic_quant(
    hidden_states, 
    smooth_scale, 
    dst_torch_dtype=torch.int8
):
    smoothed_input = torch.mul(hidden_states, smooth_scale).type(torch.float32)
    per_token_scale = torch.div(torch.max(smoothed_input.abs(), -1, keepdim=False)[0], 127.0)
    quant_tokens = torch.div(smoothed_input, per_token_scale.unsqueeze(-1)).round().type(dst_torch_dtype)
    return quant_tokens, per_token_scale

def moe_swiglu_dynamic_quant_run(scatter_tokens, smooth_scale, experts_token_count, experts_token_start, quant_tokens, per_token_scale, total_experts_num): 
    # get pre-allocated input tensors
    # scatter_tokens = tensor_mapping["scatter_tokens"]
    # smooth_scale = tensor_mapping["smooth_scale"]
    # experts_token_count = tensor_mapping["experts_token_count"]
    # experts_token_start = tensor_mapping["experts_token_start"]

    # # get per-allocated output tensors
    # quant_tokens = tensor_mapping["quant_tokens"]
    # per_token_scale = tensor_mapping["per_token_scale"]


    # swiglu, x1 used as gating, x2 used as up
    x1, x2 = torch.chunk(scatter_tokens, 2, dim=-1)
    swiglu_tokens = torch.mul(torch.nn.functional.silu(x1), x2)

    # per expert dynamic quant
    for i in range(total_experts_num):
        cur_token_start = experts_token_start[i]
        cur_token_end = cur_token_start + experts_token_count[i]

        cur_quant_tokens, cur_per_token_scale = smooth_per_token_dynamic_quant(
            swiglu_tokens[cur_token_start:cur_token_end], 
            smooth_scale[i]
        )
        quant_tokens[cur_token_start:cur_token_end] = cur_quant_tokens
        per_token_scale[cur_token_start:cur_token_end] = cur_per_token_scale

    return quant_tokens, per_token_scale

def benchmark(func, args, warmup=2, rep=10):
    for _ in range(warmup):
        func(*args)
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]
    for i in range(rep):
        start_event[i].record()
        func(*args)
        end_event[i].record()
    torch.cuda.synchronize()
    dur = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float,
    )
    return dur

def fwd_flash(scatter_tokens, smooth_scale, experts_tokens_start, experts_tokens_count, y, per_tokens_scale, total_expers_num):
    moe_swiglu_dynamic_quantize(scatter_tokens, smooth_scale, experts_tokens_start, experts_tokens_count, y, per_tokens_scale, total_expers_num)
    

def test_moe_swiglu_dynamic_quantize(n,dtype,warmup=10, rep=100):
    m = 20
    total_experts_num = 8
    torch.manual_seed(42)
    scatter_tokens = torch.randn(m, n, dtype=dtype, device='cuda', requires_grad=True)
    smooth_scale = torch.randn(total_experts_num,n//2,dtype=torch.float32,device='cuda', requires_grad=True)
    experts_token_count=torch.tensor([3, 3, 3, 3, 2, 2, 2, 2], device='cuda:0', dtype=torch.int32)
    experts_token_start=torch.tensor([0, 3, 6, 9, 12, 14, 16, 18], device='cuda:0', dtype=torch.int32)
    y_ref = torch.zeros(m,n//2,dtype=torch.int8, device='cuda')
    y = torch.zeros(m,n//2,dtype=torch.int8, device='cuda')
    per_tokens_scale_ref = torch.zeros(m,dtype=torch.float32,device='cuda')
    per_tokens_scale = torch.zeros(m,dtype=torch.float32,device='cuda')
    y_ref, per_tokens_scale_ref = moe_swiglu_dynamic_quant_run(scatter_tokens, smooth_scale, experts_token_count,experts_token_start, y_ref, per_tokens_scale_ref,total_experts_num)
    # import pdb
    # pdb.set_trace()
    fwd_flash(scatter_tokens, smooth_scale, experts_token_start, experts_token_count, y, per_tokens_scale, total_experts_num)
    torch.testing.assert_close(y_ref, y, atol=1.0, rtol=1e-3)
    torch.testing.assert_close(per_tokens_scale_ref, per_tokens_scale, atol=1e-3, rtol=1e-3)
    dur_infi = benchmark(fwd_flash, (scatter_tokens, smooth_scale, experts_token_start, experts_token_count, y, per_tokens_scale, total_experts_num), warmup, rep)
    print(f"m={m}, n={n}, dtype={dtype}, infi={dur_infi.mean().item()}")
    print("done")
    
if __name__ == "__main__":
    test_moe_swiglu_dynamic_quantize(2048,torch.float16)
    test_moe_swiglu_dynamic_quantize(4096,torch.float16)
    test_moe_swiglu_dynamic_quantize(512*8*4+4096,torch.float16)
    test_moe_swiglu_dynamic_quantize(2048,torch.bfloat16)
    test_moe_swiglu_dynamic_quantize(4096,torch.bfloat16)
    test_moe_swiglu_dynamic_quantize(512*8*4+4096,torch.bfloat16)