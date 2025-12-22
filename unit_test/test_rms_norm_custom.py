import torch
from mcoplib.custom_ops import rms_norm_dynamic_per_token_quant


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

def quantize_bf16_to_int8(input_tensor, weight, eps):
    input_tensor = torch.rms_norm(input_tensor, input_tensor.shape, weight=weight.unsqueeze(0), eps=eps)
    input_fp32 = input_tensor.float()
    max_abs = torch.max(torch.abs(input_fp32))
    scale = max_abs / 127.0  # 标量
    quantized = torch.clamp(
        torch.round(input_fp32 / scale), -128, 127
    ).to(torch.int8)
    
    return quantized, scale

def test_rms_norm_custom(warmup=10, rep=100):
    eps = 1e-6
    dtype = torch.bfloat16
    batch_size = 1
    hidden_size = 7168
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")
    
    dur_forward = benchmark(rms_norm_dynamic_per_token_quant, (x, weight, eps, torch.int8), warmup, rep)
    print(f"batch_size={batch_size}, hidden_size={hidden_size}, dtype={dtype}, infi_forward={dur_forward.mean().item()}")

    output, scales = rms_norm_dynamic_per_token_quant(x, weight, eps, torch.int8)
    outout_torch, scales_torch = quantize_bf16_to_int8(x, weight, eps)
    torch.allclose(outout_torch, output)
    torch.allclose(scales, scales_torch)

if __name__ == "__main__":

    # In this case, Megatron foward cost is 700 us , inficom forward cost is 200 us 
    # we assume the similar speedup in backward 
    test_rms_norm_custom(10, 100) 
