import torch
from mcoplib.op import moe_softmax_topk


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


def fwd_test(topk_weights, topk_indices, gating_out, pre_softmax=False):
    moe_softmax_topk(topk_weights, topk_indices, gating_out, pre_softmax)
    


def test_softmax_topk(topk, num_experts, num_tokens, dtype, warmup=10, rep=100):
    device = 'cuda'
    topk_weights = torch.zeros((num_tokens, topk), dtype=torch.float, device=device)
    topk_indices = torch.zeros((num_tokens, topk), dtype=torch.int32, device=device)
    gating_out = torch.randn((num_tokens, num_experts), dtype=torch.float, device=device)
    dur_infi = benchmark(fwd_test, (topk_weights, topk_indices, gating_out, False), warmup, rep)
    print(f"k={topk}, m={num_tokens}, n={num_experts}, dtype={dtype}, dur_infi={dur_infi.mean().item()}")
    
if __name__ == "__main__":
    test_softmax_topk(8, 256, 32, torch.float)