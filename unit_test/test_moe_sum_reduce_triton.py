#!/usr/bin/env python3
"""
测试moe_sum_reduce_triton函数的单元测试
该测试专门针对fused_moe_triton_kernels.py中的moe_sum_reduce_triton函数

测试内容包括：
1. 精度测试（使用余弦相似度）
2. 性能测试（微秒级耗时，包含warmup）
3. CPU/GPU运行时间和算力强度测试
4. 基于DeepSeek-R1模型参数的测试
"""

import time
import math
import unittest
from typing import Tuple

import torch
import torch.nn.functional as F

# 导入要测试的函数
from mcoplib.triton_fused_moe import moe_sum_reduce_triton
class TestMoeSumReduceTriton(unittest.TestCase):
    """测试moe_sum_reduce_triton函数的单元测试类"""

    @classmethod
    def setUpClass(cls):
        """设置测试类，检查CUDA可用性"""
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.cuda_available = torch.cuda.is_available()

        if not cls.cuda_available:
            cls.skipTest("CUDA not available, skipping tests")

        # DeepSeek-R1 模型的典型参数
        cls.deepseek_hidden_size = 2048  # DeepSeek模型的hidden_size
        cls.deepseek_topk = 6           # DeepSeek的num_experts_per_tok
        cls.deepseek_n_routed_experts = 64  # DeepSeek的n_routed_experts

        print(f"Running on device: {cls.device}")
        print(f"DeepSeek-R1 parameters: hidden_size={cls.deepseek_hidden_size}, topk={cls.deepseek_topk}")

    def setUp(self):
        """每个测试方法前的设置"""
        if not self.cuda_available:
            self.skipTest("CUDA not available")
        torch.cuda.empty_cache()  # 清理GPU内存

    def _create_test_data(self, token_num: int, hidden_dim: int, topk_num: int, dtype: torch.dtype = torch.float16) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        创建测试数据

        Args:
            token_num: token数量
            hidden_dim: hidden维度
            topk_num: topk数量
            dtype: 数据类型

        Returns:
            input_tensor: 输入tensor (token_num, topk_num, hidden_dim)
            output_tensor: 输出tensor (token_num, hidden_dim)
            routed_scaling_factor: 缩放因子
        """
        # 创建输入数据 - 模拟MOE输出的中间结果
        input_tensor = torch.randn(token_num, topk_num, hidden_dim, dtype=dtype, device=self.device)

        # 创建输出数据
        output_tensor = torch.zeros(token_num, hidden_dim, dtype=dtype, device=self.device)

        # DeepSeek的缩放因子，通常为1.0
        routed_scaling_factor = 1.0

        return input_tensor, output_tensor, routed_scaling_factor

    def _reference_implementation(self, input_tensor: torch.Tensor, routed_scaling_factor: float) -> torch.Tensor:
        """
        参考实现（PyTorch原生）用于精度对比

        Args:
            input_tensor: 输入tensor (token_num, topk_num, hidden_dim)
            routed_scaling_factor: 缩放因子

        Returns:
            参考输出tensor (token_num, hidden_dim)
        """
        # 在topk维度上求和，然后乘以缩放因子
        result = torch.sum(input_tensor, dim=1) * routed_scaling_factor
        return result

    def _cosine_similarity(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        """
        计算两个tensor的余弦相似度

        Args:
            tensor1: 第一个tensor
            tensor2: 第二个tensor

        Returns:
            余弦相似度（0-1之间的值）
        """
        # 展平为一维向量
        flat1 = tensor1.flatten().float()
        flat2 = tensor2.flatten().float()

        # 计算余弦相似度
        similarity = F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0), dim=1)
        return similarity.item()

    def test_accuracy_deepseek_parameters(self):
        """测试基于DeepSeek参数的精度"""
        print("\n=== 测试精度（DeepSeek-R1参数）===")

        # 使用DeepSeek-R1的参数
        token_num = 32  # batch size
        hidden_dim = self.deepseek_hidden_size
        topk_num = self.deepseek_topk
        dtype = torch.float16

        # 创建测试数据
        input_tensor, output_tensor, routed_scaling_factor = self._create_test_data(
            token_num, hidden_dim, topk_num, dtype
        )

        # 运行待测试的函数
        moe_sum_reduce_triton(input_tensor, output_tensor, routed_scaling_factor)

        # 计算参考结果
        reference_output = self._reference_implementation(input_tensor, routed_scaling_factor)

        # 计算余弦相似度
        cosine_sim = self._cosine_similarity(output_tensor, reference_output)

        print(f"Token数量: {token_num}")
        print(f"Hidden维度: {hidden_dim}")
        print(f"TopK: {topk_num}")
        print(f"数据类型: {dtype}")
        print(f"余弦相似度: {cosine_sim:.6f}")

        # 断言：余弦相似度应该大于0.999
        self.assertGreater(cosine_sim, 0.999, f"余弦相似度 {cosine_sim} 低于阈值 0.999")

        # 断言：输出形状应该正确
        self.assertEqual(output_tensor.shape, (token_num, hidden_dim))

        print("✓ 精度测试通过")

    def test_accuracy_various_sizes(self):
        """测试不同输入尺寸的精度"""
        print("\n=== 测试不同尺寸的精度 ===")

        test_cases = [
            (1, 512, 2),    # 小尺寸
            (8, 1024, 4),   # 中等尺寸
            (64, 4096, 8),  # 大尺寸
            (16, 2048, 6),  # DeepSeek类似尺寸
        ]

        for token_num, hidden_dim, topk_num in test_cases:
            with self.subTest(token_num=token_num, hidden_dim=hidden_dim, topk_num=topk_num):
                print(f"测试尺寸: token_num={token_num}, hidden_dim={hidden_dim}, topk_num={topk_num}")

                # 创建测试数据
                input_tensor, output_tensor, routed_scaling_factor = self._create_test_data(
                    token_num, hidden_dim, topk_num
                )

                # 运行待测试的函数
                moe_sum_reduce_triton(input_tensor, output_tensor, routed_scaling_factor)

                # 计算参考结果
                reference_output = self._reference_implementation(input_tensor, routed_scaling_factor)

                # 计算余弦相似度
                cosine_sim = self._cosine_similarity(output_tensor, reference_output)

                print(f"  余弦相似度: {cosine_sim:.6f}")

                # 断言：余弦相似度应该大于0.999
                self.assertGreater(cosine_sim, 0.999,
                    f"尺寸 {token_num}x{hidden_dim}x{topk_num} 的余弦相似度 {cosine_sim} 低于阈值 0.999")

        print("✓ 不同尺寸精度测试通过")

    def test_performance_with_warmup(self):
        """测试性能（包含warmup）"""
        print("\n=== 测试性能（包含warmup）===")

        # 使用DeepSeek-R1的参数
        token_num = 128  # 较大的batch size测试性能
        hidden_dim = self.deepseek_hidden_size
        topk_num = self.deepseek_topk
        dtype = torch.float16
        warmup_iterations = 10
        test_iterations = 100

        # 创建测试数据
        input_tensor, output_tensor, routed_scaling_factor = self._create_test_data(
            token_num, hidden_dim, topk_num, dtype
        )

        # Warmup
        print(f"Warmup {warmup_iterations} 次迭代...")
        for i in range(warmup_iterations):
            moe_sum_reduce_triton(input_tensor, output_tensor, routed_scaling_factor)

        # 确保所有CUDA操作完成
        torch.cuda.synchronize()

        # 性能测试
        print(f"性能测试 {test_iterations} 次迭代...")
        start_time = time.perf_counter()

        for i in range(test_iterations):
            moe_sum_reduce_triton(input_tensor, output_tensor, routed_scaling_factor)

        torch.cuda.synchronize()  # 确保所有CUDA操作完成
        end_time = time.perf_counter()

        # 计算统计信息
        total_time = end_time - start_time
        avg_time_us = (total_time / test_iterations) * 1_000_000  # 转换为微秒
        throughput = (token_num * test_iterations) / total_time  # tokens per second

        print(f"输入尺寸: {token_num} x {topk_num} x {hidden_dim}")
        print(f"数据类型: {dtype}")
        print(f"总时间: {total_time:.6f}s")
        print(f"平均时间: {avg_time_us:.2f} μs")
        print(f"吞吐量: {throughput:.2f} tokens/s")

        # 性能断言：平均时间应该在合理范围内（这里设为1ms作为参考）
        self.assertLess(avg_time_us, 1000, f"平均时间 {avg_time_us:.2f} μs 超过阈值 1000 μs")

        print("✓ 性能测试通过")

    def test_compute_intensity(self):
        """测试算力强度"""
        print("\n=== 测试算力强度 ===")

        # 使用DeepSeek-R1的参数
        token_num = 64
        hidden_dim = self.deepseek_hidden_size
        topk_num = self.deepseek_topk
        dtype = torch.float16

        # 创建测试数据
        input_tensor, output_tensor, routed_scaling_factor = self._create_test_data(
            token_num, hidden_dim, topk_num, dtype
        )

        # 计算操作数量：每个token需要对topk个专家的hidden_dim进行求和
        # 总操作数 = token_num * topk_num * hidden_dim (加法操作)
        total_operations = token_num * topk_num * hidden_dim

        # 测试多次运行计算平均时间
        num_runs = 50
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(num_runs):
            moe_sum_reduce_triton(input_tensor, output_tensor, routed_scaling_factor)

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        avg_time_s = (end_time - start_time) / num_runs
        ops_per_second = total_operations / avg_time_s
        gflops = ops_per_second / 1e9

        print(f"输入尺寸: {token_num} x {topk_num} x {hidden_dim}")
        print(f"总操作数: {total_operations:,}")
        print(f"平均时间: {avg_time_s * 1000:.3f} ms")
        print(f"算力强度: {gflops:.2f} GFLOPS")

        # 算力强度断言：应该达到合理的GFLOPS水平
        self.assertGreater(gflops, 1.0, f"算力强度 {gflops:.2f} GFLOPS 低于阈值 1.0 GFLOPS")

        print("✓ 算力强度测试通过")

    def test_cpu_gpu_timing(self):
        """测试CPU和GPU运行时间"""
        print("\n=== 测试CPU/GPU运行时间 ===")

        # 使用DeepSeek-R1的参数
        token_num = 32
        hidden_dim = self.deepseek_hidden_size
        topk_num = self.deepseek_topk
        dtype = torch.float16

        # 创建测试数据
        input_tensor, output_tensor, routed_scaling_factor = self._create_test_data(
            token_num, hidden_dim, topk_num, dtype
        )

        # CPU时间测试（使用PyTorch参考实现）
        input_cpu = input_tensor.cpu()

        start_cpu = time.perf_counter()
        reference_output = self._reference_implementation(input_cpu, routed_scaling_factor)
        end_cpu = time.perf_counter()
        cpu_time_us = (end_cpu - start_cpu) * 1_000_000

        # GPU时间测试（使用Triton实现）
        torch.cuda.synchronize()
        start_gpu = time.perf_counter()
        moe_sum_reduce_triton(input_tensor, output_tensor, routed_scaling_factor)
        torch.cuda.synchronize()
        end_gpu = time.perf_counter()
        gpu_time_us = (end_gpu - start_gpu) * 1_000_000

        # 计算加速比
        speedup = cpu_time_us / gpu_time_us if gpu_time_us > 0 else float('inf')

        print(f"输入尺寸: {token_num} x {topk_num} x {hidden_dim}")
        print(f"CPU时间: {cpu_time_us:.2f} μs")
        print(f"GPU时间: {gpu_time_us:.2f} μs")
        print(f"加速比: {speedup:.2f}x")

        # 性能断言：GPU应该比CPU快（在合理的数据尺寸下）
        # 注意：对于小数据量，GPU可能没有优势
        if token_num * hidden_dim * topk_num > 100000:  # 对于较大的数据量
            self.assertGreater(speedup, 1.0, f"GPU加速比 {speedup:.2f}x 低于阈值 1.0x")

        print("✓ CPU/GPU时间测试通过")

    def test_edge_cases(self):
        """测试边界情况"""
        print("\n=== 测试边界情况 ===")

        # 测试最小尺寸
        print("测试最小尺寸...")
        input_tensor, output_tensor, routed_scaling_factor = self._create_test_data(1, 1, 1)
        moe_sum_reduce_triton(input_tensor, output_tensor, routed_scaling_factor)
        reference_output = self._reference_implementation(input_tensor, routed_scaling_factor)
        cosine_sim = self._cosine_similarity(output_tensor, reference_output)
        self.assertGreater(cosine_sim, 0.999, "最小尺寸测试失败")
        print("✓ 最小尺寸测试通过")

        # 测试不同的缩放因子
        print("测试不同缩放因子...")
        scaling_factors = [0.1, 1.0, 2.5, 10.0]
        for scale in scaling_factors:
            with self.subTest(scaling_factor=scale):
                input_tensor, output_tensor, routed_scaling_factor = self._create_test_data(8, 512, 4)
                moe_sum_reduce_triton(input_tensor, output_tensor, scale)
                reference_output = self._reference_implementation(input_tensor, scale)
                cosine_sim = self._cosine_similarity(output_tensor, reference_output)
                self.assertGreater(cosine_sim, 0.999, f"缩放因子 {scale} 测试失败")

        print("✓ 不同缩放因子测试通过")
        print("✓ 边界情况测试通过")

    def test_different_dtypes(self):
        """测试不同的数据类型"""
        print("\n=== 测试不同数据类型 ===")

        dtypes = [torch.float16, torch.bfloat16, torch.float32]
        token_num = 16
        hidden_dim = 1024
        topk_num = 4

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                print(f"测试数据类型: {dtype}")

                # 创建测试数据
                input_tensor = torch.randn(token_num, topk_num, hidden_dim, dtype=dtype, device=self.device)
                output_tensor = torch.zeros(token_num, hidden_dim, dtype=dtype, device=self.device)
                routed_scaling_factor = 1.0

                # 运行测试
                moe_sum_reduce_triton(input_tensor, output_tensor, routed_scaling_factor)

                # 计算参考结果
                reference_output = self._reference_implementation(input_tensor, routed_scaling_factor)

                # 计算余弦相似度
                cosine_sim = self._cosine_similarity(output_tensor, reference_output)

                print(f"  余弦相似度: {cosine_sim:.6f}")

                # 断言：余弦相似度应该大于0.999
                self.assertGreater(cosine_sim, 0.999,
                    f"数据类型 {dtype} 的余弦相似度 {cosine_sim} 低于阈值 0.999")

        print("✓ 不同数据类型测试通过")


def run_performance_analysis():
    """运行详细的性能分析"""
    print("=" * 60)
    print("MOE Sum Reduce Triton 性能分析")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA不可用，跳过性能分析")
        return

    device = torch.device("cuda")

    # 测试不同尺寸的性能
    test_configs = [
        (32, 2048, 6),   # DeepSeek基础配置
        (64, 2048, 6),   # 更大batch
        (32, 4096, 6),   # 更大hidden
        (64, 4096, 8),   # 大配置
        (128, 2048, 6),  # 大batch
    ]

    for token_num, hidden_dim, topk_num in test_configs:
        print(f"\n配置: {token_num} x {topk_num} x {hidden_dim}")
        print("-" * 40)

        # 创建测试数据
        input_tensor = torch.randn(token_num, topk_num, hidden_dim, dtype=torch.float16, device=device)
        output_tensor = torch.zeros(token_num, hidden_dim, dtype=torch.float16, device=device)
        routed_scaling_factor = 1.0

        # Warmup
        for _ in range(20):
            moe_sum_reduce_triton(input_tensor, output_tensor, routed_scaling_factor)

        torch.cuda.synchronize()

        # 时间测试
        num_iterations = 100
        start_time = time.perf_counter()

        for _ in range(num_iterations):
            moe_sum_reduce_triton(input_tensor, output_tensor, routed_scaling_factor)

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        # 计算性能指标
        avg_time_ms = ((end_time - start_time) / num_iterations) * 1000
        throughput_tokens_per_sec = token_num / ((end_time - start_time) / num_iterations)
        memory_gb = (input_tensor.numel() + output_tensor.numel()) * 2 / (1024**3)  # FP16

        # 计算带宽
        data_bytes = (input_tensor.numel() + output_tensor.numel()) * 2  # 2 bytes for FP16
        bandwidth_gb_per_sec = data_bytes / ((end_time - start_time) / num_iterations) / (1024**3)

        print(f"平均时间: {avg_time_ms:.3f} ms")
        print(f"吞吐量: {throughput_tokens_per_sec:.2f} tokens/s")
        print(f"内存使用: {memory_gb:.2f} GB")
        print(f"带宽: {bandwidth_gb_per_sec:.2f} GB/s")


if __name__ == "__main__":
    # 运行单元测试
    print("运行 MOE Sum Reduce Triton 单元测试")
    unittest.main(argv=[''], exit=False, verbosity=2)

    # 运行性能分析
    run_performance_analysis()