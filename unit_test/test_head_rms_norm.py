import torch
from mcoplib.op import head_rms_norm

def head_rms_norm_ref(hidden_states, weight, eps=1e-5):
    # 提取需要进行归一化的部分
    selected_heads = hidden_states

    # 计算均方根归一化
    variance = selected_heads.to(torch.float32).pow(2).mean(-1, keepdim=True)
    normalized_heads = selected_heads / torch.rsqrt(variance + eps)

    # 应用权重
    weighted_heads = weight * normalized_heads

    # 将归一化后的部分放回原张量
    
    return weighted_heads



def head_rms_norm_run(token_data, weight, y, eps, norm_head_num):
        # per head rms_norm
        for head_idx in range(norm_head_num):
            head_data = token_data[:, head_idx, :]
            head_weight = weight[head_idx, :]
            # y[:, head_idx, :] = torch.nn.functional.rms_norm(
            #     head_data, 
            #     normalized_shape=head_weight.shape,
            #     weight=head_weight,
            #     eps=eps
            # )
            y[:, head_idx, :] = head_rms_norm_ref(head_data, head_weight, eps)
        return y

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

def fwd_flash(output,hidden_states, weight, eps, head_offset,head_norm):
    head_rms_norm(output,hidden_states, weight, eps, head_offset, head_norm)

def test_head_rms_norm(num_tokens, head_norm, head_dim, head_size, dtype,warmup=10, rep=100):
    torch.manual_seed(42)
    hidden_states = torch.randn(num_tokens, head_dim, head_size, dtype=dtype, device='cuda', requires_grad=True)
    # bfloat16 or float32
    # weight = torch.randn(head_dim,head_size,dtype=torch.float32,device='cuda', requires_grad=True)
    weight = torch.randn(head_dim,head_size,dtype=dtype,device='cuda', requires_grad=True)
    output_ref = hidden_states.clone()
    output = hidden_states.clone()
    head_offset = 32
    head_rms_norm_run(hidden_states, weight, output_ref,  1e-5, head_norm)
    fwd_flash(output, hidden_states, weight, 1e-5, 0, head_norm)
    torch.testing.assert_close(output_ref, output, atol=1e-5, rtol=1e-5)
    print("done")

if __name__ == "__main__":
    # test_head_rms_norm(128,64,128,4096,torch.bfloat16)
    # test_head_rms_norm(128,64,128,1024*8,torch.bfloat16)
    # test_head_rms_norm(128,64,128,1024*8*2,torch.bfloat16)
    # test_head_rms_norm(128,64,128,1024*8*3,torch.bfloat16)

    test_head_rms_norm(128,64,128,4096,torch.float32)
    test_head_rms_norm(128,64,128,1024*4,torch.float32)
    test_head_rms_norm(128,64,128,1024*4*2,torch.float32)
    test_head_rms_norm(128,64,128,1024*4*3,torch.float32)
    