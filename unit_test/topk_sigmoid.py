import torch
import torch.nn.functional as F
import time
import math

# 尝试导入自定义的 CUDA op
try:
    import mcoplib._moe_C
except ImportError:
    print("Warning: 无法导入 vllm._moe_C, 请确保算子已正确编译安装在环境中。")
    print("测试脚本将继续，但在调用 torch.ops._moe_C.topk_sigmoid 时可能会报错。")

def cpu_topk_sigmoid_reference(gating_output, bias, topk, renormalize):
    """
    完全对齐 C++ 源码逻辑的 CPU 参考实现
    """
    # 强制转换到 float32 以保证计算精度
    gating_fp32 = gating_output.float()
    bias_fp32 = bias.float() if bias is not None else None

    # 1. 计算 Sigmoid
    sigmoid_scores = torch.sigmoid(gating_fp32)

    # 2. 加上 bias 用于做 Top-K 决策 (对应源码中的 row_chunk_for_choice)
    if bias_fp32 is not None:
        routing_scores = sigmoid_scores + bias_fp32
    else:
        routing_scores = sigmoid_scores

    # 3. 选出 Top-K 的专家索引
    _, topk_indices = torch.topk(routing_scores, k=topk, dim=-1)

    # 4. 根据索引去原 sigmoid_scores (未加 bias) 中提取最终权重
    topk_weights = torch.gather(sigmoid_scores, dim=-1, index=topk_indices)

    # 5. 可选：重归一化
    if renormalize:
        # 为了防止除以 0，源码中有一句：denom = selected_sum > 0.f ? selected_sum : 1.f;
        row_sum = topk_weights.sum(dim=-1, keepdim=True)
        row_sum = torch.where(row_sum > 0.0, row_sum, torch.tensor(1.0, device=row_sum.device))
        topk_weights = topk_weights / row_sum

    return topk_weights, topk_indices

def run_test():
    # --- 1. Step3.5 模型配置参数 ---
    NUM_EXPERTS = 288        # moe_num_experts
    TOP_K = 8                # moe_top_k
    DTYPE = torch.bfloat16   # torch_dtype
    NUM_TOKENS = 4096        # 模拟一次典型的 prefill token 数量
    RENORMALIZE = True      # 测试参数：是否归一化 (可以改为 True 测试另一个分支)

    print(f"=== 开始测试 topk_sigmoid ===")
    print(f"配置: Tokens={NUM_TOKENS}, Experts={NUM_EXPERTS}, TopK={TOP_K}, Dtype={DTYPE}")

    # --- 2. 构造输入 ---
    torch.manual_seed(42)  # 固定随机数种子保证可复现
    # gating_output: [num_tokens, num_experts]
    gating_output = torch.randn((NUM_TOKENS, NUM_EXPERTS), dtype=DTYPE, device='cuda')
    # bias 必须是一维 float32
    bias = torch.randn(NUM_EXPERTS, dtype=torch.float32, device='cuda')

    # 分配 CUDA op 的输出张量
    cuda_topk_weights = torch.empty((NUM_TOKENS, TOP_K), dtype=torch.float32, device='cuda')
    cuda_topk_indices = torch.empty((NUM_TOKENS, TOP_K), dtype=torch.int32, device='cuda')
    # token_expert_indices 在 C++ 源码主要用于后续排序，这里我们也按要求分配
    cuda_token_expert_indices = torch.empty((NUM_TOKENS, TOP_K), dtype=torch.int32, device='cuda')

    # --- 3. 精度对比 (Accuracy Test) ---
    print("\n[1/2] 正在进行精度校验...")

    # 3.1 获取 CPU 结果
    cpu_weights, cpu_indices = cpu_topk_sigmoid_reference(gating_output, bias, TOP_K, RENORMALIZE)

    # 3.2 获取 CUDA 结果
    torch.ops._moe_C.topk_sigmoid(
        cuda_topk_weights,
        cuda_topk_indices,
        cuda_token_expert_indices,
        gating_output,
        RENORMALIZE,
        bias
    )

    # 强制同步以确保 CUDA Kernel 执行完毕
    torch.cuda.synchronize()

    # 3.3 比较 Indices 是否完全一致
    indices_match = torch.equal(cpu_indices.to(torch.int32), cuda_topk_indices)
    assert indices_match, "❌ 精度测试失败：CUDA 算子与 CPU 选出的 Top-K 专家索引不一致！"
    print("✅ Top-K 专家索引 (Indices) 对比完全一致。")

    # 3.4 比较 Weights (要求精度误差小于 0.00001)
    cos_sim = F.cosine_similarity(cpu_weights.flatten(), cuda_topk_weights.flatten(), dim=0)
    cos_sim_val = cos_sim.item()
    precision_error = 1.0 - cos_sim_val

    max_diff = torch.max(torch.abs(cpu_weights - cuda_topk_weights)).item()

    print(f"📊 权重余弦相似度: {cos_sim_val:.8f}")
    print(f"📊 权重余弦相似度误差 (1 - cos_sim): {precision_error:.8e}")
    print(f"📊 权重最大绝对误差 (Max Diff): {max_diff:.8e}")

    assert precision_error < 0.00001, f"❌ 精度测试失败：余弦相似度误差 {precision_error} >= 0.00001"
    assert math.isnan(precision_error) == False, "❌ 精度测试失败：遇到 NaN"
    print("✅ 权重精度校验通过 (误差 < 0.00001)。")


    # --- 4. 耗时统计 (Performance Test) ---
    print("\n[2/2] 正在进行耗时统计分析 (Profile CPU vs CUDA)...")
    WARMUP = 10
    RUNS = 100

    # 4.1 Profile CPU
    for _ in range(WARMUP):
        _ = cpu_topk_sigmoid_reference(gating_output, bias, TOP_K, RENORMALIZE)

    start_time = time.time()
    for _ in range(RUNS):
        _ = cpu_topk_sigmoid_reference(gating_output, bias, TOP_K, RENORMALIZE)
    cpu_avg_time = (time.time() - start_time) / RUNS * 1000  # 转换为毫秒(ms)

    # 4.2 Profile CUDA
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(WARMUP):
        torch.ops._moe_C.topk_sigmoid(
            cuda_topk_weights, cuda_topk_indices, cuda_token_expert_indices,
            gating_output, RENORMALIZE, bias
        )
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(RUNS):
        torch.ops._moe_C.topk_sigmoid(
            cuda_topk_weights, cuda_topk_indices, cuda_token_expert_indices,
            gating_output, RENORMALIZE, bias
        )
    end_event.record()
    torch.cuda.synchronize()
    cuda_avg_time = start_event.elapsed_time(end_event) / RUNS  # elapsed_time 默认单位为毫秒

    print(f"⏱️  CPU 平均耗时 (100次跑均): {cpu_avg_time:.4f} ms")
    print(f"⏱️  GPU CUDA 算子平均耗时: {cuda_avg_time:.4f} ms")
    print(f"🚀 CUDA 加速比: {cpu_avg_time / cuda_avg_time:.2f}x")
    print("\n=== 测试圆满结束 ===")

if __name__ == "__main__":
    run_test()
