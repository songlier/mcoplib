import torch
from mcoplib.op import fused_rope_fwd, fused_rope_bwd


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


def flash_forward(input, cos, sin):
    fused_rope_fwd(input, cos, sin, None, False)
    return

def flash_backward(input, cos, sin):
    fused_rope_bwd(input, cos, sin, None, False)
    return


def test_flash_rope(seq_len, head_num, head_dim, dtype, warmup=10, rep=100):
    input = torch.randn(seq_len, 3, head_num, head_dim, dtype=dtype, device='cuda')
    cos = torch.randn(seq_len, head_dim // 1, dtype=dtype, device='cuda')
    sin = torch.randn(seq_len, head_dim // 1, dtype=dtype, device='cuda')
    
    dur_forward = benchmark(flash_forward, (input, cos, sin), warmup, rep)
    print(f"seq_len={seq_len}, head_num={head_num}, head_dim={head_dim}, dtype={dtype}, infi_forward={dur_forward.mean().item()}")

    dur_backward = benchmark(flash_backward, (input, cos, sin), warmup, rep)
    print(f"seq_len={seq_len}, head_num={head_num}, head_dim={head_dim}, dtype={dtype}, infi_forward={dur_backward.mean().item()}")


if __name__ == "__main__":

    # In this case, Megatron foward cost is 700 us , inficom forward cost is 200 us 
    # we assume the similar speedup in backward 
    test_flash_rope(4096, 64, 128, torch.bfloat16) 
