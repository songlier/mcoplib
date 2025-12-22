import torch
from mcoplib.op import fused_bias_dropout

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

def fwd_ref(input, bias, residual, dropout_prob):
    assert dropout_prob == 0.0
    out = torch.nn.functional.dropout(input, p=dropout_prob, training=True)
    return out + residual

def fwd_flash(input, bias, residual, dropout_prob):
    output = fused_bias_dropout(input, residual, dropout_prob)
    return output

def bwd_ref(grad_output, input, bias, residual, dropout_prob):
    assert dropout_prob == 0.0
    grad_input = grad_output
    grad_bias = grad_output
    grad_residual = grad_output
    return grad_input, grad_bias, grad_residual

def test_bias_add(m, n, dtype,warmup=10, rep=100):
    input = torch.randn(m, n, dtype=dtype, device='cuda', requires_grad=True)
    bias = torch.randn(m, n, dtype=dtype, device='cuda', requires_grad=True)
    residual = torch.randn(m, n, dtype=dtype, device='cuda', requires_grad=True)
    dropout_prob = 0.0
    
    output_ref = fwd_ref(input, bias, residual, dropout_prob)
    output_flash = fwd_flash(input, bias, residual, dropout_prob)
    all_close = torch.allclose(output_ref, output_flash, atol=1e-4, rtol=1e-4)
    assert all_close

    dur_ref = benchmark(fwd_ref, (input, bias, residual, dropout_prob), warmup, rep)
    dur_infi = benchmark(fwd_flash, (input, bias, residual, dropout_prob), warmup, rep)
    print(f"m={m}, n={n}, dtype={dtype}, ref={dur_ref.mean().item()}, infi={dur_infi.mean().item()}")
    
if __name__ == "__main__":
    test_bias_add(4096, 4096, torch.float16)
    test_bias_add(4096, 4096, torch.bfloat16)