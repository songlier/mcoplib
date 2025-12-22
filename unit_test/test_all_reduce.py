import torch
from mcoplib.op import all_reduce_max
from mcoplib.op import all_reduce_sum

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

def fwd_flash_max(input, output):
    all_reduce_max(input, output)

def fwd_flash_sum(input, output):
    all_reduce_sum(input, output)
    

def test_all_reduce(m, n, dtype, warmup=10, rep=100):
    device = 'cuda'
    gating_out = torch.randn((m, n), dtype=dtype, device=device)
    output = torch.zeros(m, device=device, dtype=dtype)
    dur_infi = benchmark(fwd_flash_max, (gating_out, output), warmup, rep)
    print(f"fwd_flash_max: m={m}, n={n}, dtype={dtype}, infi={dur_infi.mean().item()}")

    dur_infi = benchmark(fwd_flash_sum, (gating_out, output), warmup, rep)
    print(f"fwd_flash_sum: m={m}, n={n}, dtype={dtype}, infi={dur_infi.mean().item()}")
    
if __name__ == "__main__":
    test_all_reduce(4096, 4096, torch.bfloat16)