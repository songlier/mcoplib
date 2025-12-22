import torch
from mcoplib.op import fused_repeat_kv_fwd, fused_repeat_kv_bwd


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

def ref_forward(qkv, q_num_head, kv_num_head, head_dim, partition):

    (query_layer,key_layer,value_layer) = torch.split(qkv, [q_num_head // partition * head_dim, kv_num_head // partition * head_dim, kv_num_head // partition * head_dim], dim=3)

    query_layer = query_layer.reshape(query_layer.size(0), query_layer.size(1), -1, head_dim)

    repeat = q_num_head // kv_num_head

    key_layer_repeat = key_layer.repeat_interleave(repeat, dim = 2)

    value_layer_repeat = value_layer.repeat_interleave(repeat,dim = 2)
        
    result = torch.stack([query_layer, key_layer_repeat, value_layer_repeat], dim=2)

    return result

def flash_forward(qkv, q_num_head, kv_num_head, head_dim):
    result = fused_repeat_kv_fwd(qkv, q_num_head, kv_num_head, head_dim)
    return result

# clark_note: in training, mainly elementwise cal will use fp32 from chatgbt
def ref_backward(qkv, q_num_head, kv_num_head, partition):
    q = qkv[:, :, 0, :, :]
    k_repeated = qkv[:, :, 1, :, :]
    v_repeated = qkv[:, :, 2, :, :]

    seq = q.shape[0]
    bs = q.shape[1]
    head_dim = q.shape[3]

    repeat = q_num_head // kv_num_head
    
    result = torch.zeros(seq, bs, partition, (q_num_head // partition + 2 * kv_num_head // partition) * head_dim, dtype=torch.float32, device='cuda')

    result[:, :, :, :q_num_head * head_dim // partition] = q.reshape(seq, bs, partition, q_num_head * head_dim // partition)

    for i in range(repeat):
        result[:, :, :, q_num_head * head_dim // partition: (q_num_head + kv_num_head) * head_dim // partition] += k_repeated[:, :, i::repeat].reshape(seq, bs, partition, kv_num_head // partition * head_dim)
        result[:, :, :, (q_num_head + kv_num_head) * head_dim // partition: (q_num_head + 2 * kv_num_head) * head_dim // partition] += v_repeated[:, :, i::repeat].reshape(seq, bs, partition, kv_num_head // partition * head_dim)

    result = result.to(q.dtype)

    return result


def flash_backward(qkv, q_num_head, kv_num_head, partition):
    result = fused_repeat_kv_bwd(qkv, q_num_head, kv_num_head, partition)
    return result


def test_flash(seq_len, batch_size, partition, q_num_head, kv_num_head, head_dim, dtype, warmup=10, rep=100):
    
    qkv_input = torch.randn(seq_len, batch_size, partition, (q_num_head // partition + 2 * kv_num_head // partition) * head_dim, dtype=dtype, device='cuda')
    print('before: ', qkv_input.shape)
    output_forward_ref = ref_forward(qkv_input, q_num_head, kv_num_head, head_dim, partition)
    print('after: ', output_forward_ref.shape)
    output_forward_flash = flash_forward(qkv_input, q_num_head, kv_num_head, head_dim)
    all_close = torch.allclose(output_forward_flash, output_forward_ref, atol=1e-3, rtol=1e-3)
    assert all_close
    
    output_backward_ref = ref_backward(output_forward_ref, q_num_head, kv_num_head, partition)
    output_backward_flash = flash_backward(output_forward_ref, q_num_head, kv_num_head, partition)
    all_close = torch.allclose(output_backward_ref, output_backward_flash, atol=1e-3, rtol=1e-3)
    assert all_close
   
    
    dur_forward = benchmark(flash_forward, (qkv_input, q_num_head, kv_num_head, head_dim), warmup, rep)
    print(f"seq_len={seq_len}, batch_size={batch_size}, partition={partition}, q_num_head={q_num_head}, kv_num_head={kv_num_head}, head_dim={head_dim}, dtype={dtype}, infi_forward={dur_forward.mean().item()}")

    dur_backward = benchmark(flash_backward, (output_forward_ref, q_num_head, kv_num_head, partition), warmup, rep)
    print(f"seq_len={seq_len}, batch_size={batch_size}, partition={partition}, q_num_head={q_num_head}, kv_num_head={kv_num_head}, head_dim={head_dim}, dtype={dtype}, infi_forward={dur_backward.mean().item()}")


if __name__ == "__main__":
    
    print('llama2-70B-TP4: ')
    test_flash(4096, 1, 2, 16, 2, 128, torch.bfloat16)
    print('llama2-70B-TP8: ')
    test_flash(4096, 1, 1, 8, 1, 128, torch.bfloat16)
    test_flash(4096, 2, 1, 8, 1, 128, torch.bfloat16)
    print('llama3-8B-TP1: ')
    test_flash(8192, 2, 8, 32, 8, 128, torch.bfloat16)
    print('llama2-130B-TP4: ')
    test_flash(8192, 1, 2, 24, 2, 128, torch.bfloat16)
    print('llama2-130B-TP8: ')
    test_flash(8192, 1, 1, 12, 1, 128, torch.bfloat16)
    print('qwen2-32B-TP4: ')
    test_flash(4096, 1, 2, 10, 2, 128, torch.bfloat16)
    print('qwen2-32B-TP2: ')
    test_flash(4096, 1, 4, 20, 4, 128, torch.bfloat16)
    print('other: ')
    test_flash(4096, 1, 1, 7, 1, 128, torch.bfloat16)
    test_flash(4096, 2, 1, 7, 1, 128, torch.bfloat16)
    test_flash(4096, 1, 2, 8, 2, 128, torch.bfloat16)
    test_flash(4096, 1, 1, 4, 1, 128, torch.bfloat16)
    test_flash(4096, 2, 1, 4, 1, 128, torch.bfloat16)

