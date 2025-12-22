import torch
import os
from mcoplib.profiler import profiler
# from mcoplib.op import 
import mcoplib.sgl_kernel

# 使用简化的profiler装饰器
# @profiler(output_dir="./profiles", warmup=2, repeat=3)
def per_token_cast_to_fp8(x: torch.Tensor):
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, n
    ), (x_amax / 448.0).view(m, -1)

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

def per_token_cast_to_fp8_ref(input):
    return per_token_cast_to_fp8(input)

def test_per_token_cast_to_fp8(m, n, dtype,warmup=1, rep=1):
    torch.manual_seed(42)
    input = torch.randn(m, n, dtype=dtype, device='cuda', requires_grad=True)
    out_fp8_ref, out_scale = per_token_cast_to_fp8_ref(input)
    out = torch.empty(m, n, dtype=torch.float8_e4m3fn, device='cuda')
    scale = torch.empty(m,n//128, dtype=torch.float32, device='cuda')
    torch.ops.sgl_kernel.per_token_cast_to_fp8.default(out, scale, input)
    # torch.testing.assert_close(out_fp8_ref, out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_scale, scale, atol=1e-5, rtol=1e-5)
    print("done")
    


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("使用设备:", device)
    test_per_token_cast_to_fp8(4096,4096,torch.bfloat16)

    print("\n=== 简化Profiler使用说明 ===")
    print("1. 装饰器已简化，移除了tensorboard相关参数")
    print("2. 按照test.py格式输出详细的trace信息")
    print("3. 只使用export_chrome_trace保存trace文件")
    print("4. 性能数据和trace文件已保存到 ./profiles/ 目录")
# 使用简化的profiler装饰器
