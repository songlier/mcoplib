import torch
import os
from mcoplib.profiler import profiler
from mcoplib.op import rms_norm


# 使用简化的profiler装饰器
@profiler(output_dir="./profiles", warmup=2, repeat=3)
def rms_norm_ref(
    hidden_states: torch.Tensor,    # 输入张量 [batch_size, hidden_size]
    residual: torch.Tensor,  # 可选残差张量 [batch_size, hidden_size]
    weight: torch.Tensor,           # RMS归一化权重 [hidden_size]
    after_res:torch.Tensor,
    eps: float = 1e-6,              # 数值稳定性参数
):
    """
    Add + RMS Norm + Dynamic Quant 融合算子
    
    参数:
        hidden_states: 输入张量
        residual: 可选的残差连接输入
        weight: RMS归一化的权重参数
        eps: 防止除零的小常数
    返回:
        after_res: 残差连接后的结果 [batch_size, hidden_size]
        output_quant: 量化后的输出 [batch_size, hidden_size] (int8)
        scale: 每个token的缩放因子 [batch_size] (float32)
    """
    # 1. 残差连接: hidden_states + residual
    if residual is not None:
        after_res = hidden_states + residual
    else:
        after_res = hidden_states
    
    # 2. RMS归一化
    # 计算均方根
    mean_square = torch.mean(after_res * after_res, dim=-1, keepdim=True)
    rms = torch.rsqrt(mean_square + eps)
    
    # 应用归一化和权重
    normalized = after_res / rms * weight
    
    return normalized.to(hidden_states.dtype), after_res

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

def fwd_flash(input, weight, var_epsilon, residual,out, after_res):
    rms_norm(out, input, weight, var_epsilon, after_res, residual, True)

def test_rms_norm(m, n, dtype,warmup=1, rep=1):
    torch.manual_seed(42)
    hidden_states = torch.randn(m, n, dtype=dtype, device='cuda', requires_grad=True)
    residual = torch.randn(m,n,dtype=dtype,device='cuda', requires_grad=True)
    
    # bfloat16 or float32
    weight = torch.randn(n,dtype=dtype,device='cuda', requires_grad=True)
    after_res_ref = torch.empty(m, n, dtype=dtype, device='cuda')
    after_res = torch.empty(m, n, dtype=dtype, device='cuda')
    norm_ref = torch.empty(m, n, dtype=dtype, device='cuda')
    norm = torch.empty(m, n, dtype=dtype, device='cuda')
    norm_ref, after_res_ref = rms_norm_ref(hidden_states, residual, weight, after_res_ref, 1e-5)
    fwd_flash(hidden_states, weight, 1e-5, residual, norm, after_res)
    
    torch.testing.assert_close(norm_ref, norm, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(after_res_ref.reshape(-1), after_res.reshape(-1), atol=1e-5, rtol=1e-5)
    dur_infi = benchmark(fwd_flash, (hidden_states, weight, 1e-5, residual, norm, after_res), warmup, rep)
    print(f"m={m}, n={n}, dtype={dtype}, infi={dur_infi.mean().item()}")


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("使用设备:", device)
    test_rms_norm(4096,4096,torch.float32)

    print("\n=== 简化Profiler使用说明 ===")
    print("1. 装饰器已简化，移除了tensorboard相关参数")
    print("2. 按照test.py格式输出详细的trace信息")
    print("3. 只使用export_chrome_trace保存trace文件")
    print("4. 性能数据和trace文件已保存到 ./profiles/ 目录")
