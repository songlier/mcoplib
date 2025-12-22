import torch
from mcoplib.fused_router_drop import FusedRouterDrop


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

def flash_fwd(logits, topk):
    return FusedRouterDrop.apply(logits, topk)

def flash_bwd(probs, grad_output):
    return probs.backward(grad_output, retain_graph=True)

def test_router_drop(seq_len, expert, topk, warmup=10, rep=100):
    logits = torch.randn(seq_len, expert, device='cuda', requires_grad=True)
    grad_output = torch.randn(seq_len, topk, device='cuda')
    probs, top_indices, tokens_per_expert = flash_fwd(logits, topk)
    dur_forward = benchmark(flash_fwd, (logits, topk), warmup, rep)
    print(f"seq_len={seq_len}, expert={expert}, topk={topk}, dur_forward={dur_forward.mean().item()}")

    dur_backward = benchmark(flash_bwd, (probs, grad_output), warmup, rep)
    print(f"seq_len={seq_len}, expert={expert}, topk={topk},  dur_backward={dur_backward.mean().item()}")

if __name__ == "__main__":
    # In this case, Megatron foward cost is 700 us , inficom forward cost is 200 us 
    # we assume the similar speedup in backward 
    test_router_drop(4096, 8, 2) 
