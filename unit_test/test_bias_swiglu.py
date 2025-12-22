import torch
from mcoplib.op import fused_bias_swiglu_fwd, fused_bias_swiglu_bwd

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

def fwd_ref(input, bias):
    dtype = input.dtype
    input = input.to(torch.float32)
    size = input.size(-1)
    output = torch.nn.functional.silu(input[..., :size // 2]) * input[..., size // 2:]
    return output.to(dtype)

def fwd_flash(input, bias):
    output = fused_bias_swiglu_fwd(input)
    return output

def bwd_ref(grad_output, output, input, bias):
    dtype = input.dtype
    input = input.to(torch.float32)
    grad_output = grad_output.to(torch.float32)
    x, y = input.chunk(2, dim=-1)
    sigmoid_x = torch.sigmoid(x)
    gix = grad_output * y * sigmoid_x * (1 + x * (1 - sigmoid_x))
    giy = grad_output * x * sigmoid_x
    grad_input = torch.cat([gix, giy], dim=-1)
    return grad_input.to(dtype)

def bwd_flash(grad_output, output, input, bias):
    grad_input = fused_bias_swiglu_bwd(input, grad_output)
    return grad_input

def test_bias_silu_dot(m, n, dtype,warmup=10, rep=100):
    input = torch.randn(m, 1, n * 2, dtype=dtype, device='cuda', requires_grad=True)
    # [TODO] support bias with non-zero value
    bias = torch.zeros(n * 2, dtype=dtype, device='cuda', requires_grad=True)
    grad_output = torch.randn(m, 1, n, dtype=dtype, device='cuda', requires_grad=True)
    
    output_ref = fwd_ref(input, bias)
    output_flash = fwd_flash(input, bias)
    all_close = torch.allclose(output_ref, output_flash, atol=1e-5, rtol=1e-5)
    assert all_close
    grad_input_ref = bwd_ref(grad_output, output_ref, input, bias)
    grad_input_flash = bwd_flash(grad_output, output_ref, input, bias)
    all_close = torch.allclose(grad_input_ref, grad_input_flash, atol=1e-2, rtol=1e-2)
    assert all_close

    dur_ref = benchmark(fwd_ref, (input, bias), warmup, rep)
    dur_infi = benchmark(fwd_flash, (input, bias), warmup, rep)
    print(f"fwd: m={m}, n={n}, dtype={dtype}, ref={dur_ref.mean().item()}, infi={dur_infi.mean().item()}")
    
    dur_ref = benchmark(bwd_ref, (grad_output, output_ref, input, bias), warmup, rep)
    dur_infi = benchmark(bwd_flash, (grad_output, output_ref, input, bias), warmup, rep)
    print(f"bwd: m={m}, n={n}, dtype={dtype}, ref={dur_ref.mean().item()}, infi={dur_infi.mean().item()}")
    
if __name__ == "__main__":
#    test_bias_silu_dot(4096, 28672 // 4, torch.float16)
#    test_bias_silu_dot(4096, 28672 // 4, torch.bfloat16)
#    test_bias_silu_dot(4096, 28672 // 2, torch.float16)
    test_bias_silu_dot(4096, 28672 // 2, torch.bfloat16)
