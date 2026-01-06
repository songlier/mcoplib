import time
import math
import unittest
from typing import Tuple, List

import torch
import torch.nn.functional as F

from mcoplib.triton_fused_moe import fused_append_shared_experts


class TestFusedAppendSharedExperts(unittest.TestCase):
    """测试fused_append_shared_experts函数的单元测试类"""

    @classmethod
    def setUpClass(cls):
        """设置测试类，检查CUDA可用性"""
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.cuda_available = torch.cuda.is_available()

        if not cls.cuda_available:
            cls.skipTest("CUDA not available, skipping tests")

        # DeepSeek-R1 模型的典型参数
        # 注意：Triton的tl.arange要求参数必须是2的幂次方
        cls.deepseek_n_routed_experts = 64     # DeepSeek的路由专家数量
        cls.deepseek_topk = 8                  # DeepSeek的每个token选中的专家数量（必须是2的幂次方）
        cls.deepseek_n_shared_experts = 1      # DeepSeek的共享专家数量（必须是2的幂次方）
        cls.deepseek_num_fused_shared_experts = 1  # DeepSeek的融合共享专家数量（必须是2的幂次方）
        cls.deepseek_hidden_size = 2048        # DeepSeek的hidden_size

        print(f"Running on device: {cls.device}")
        print(f"DeepSeek-R1 parameters (adjusted for Triton constraints):")
        print(f"  n_routed_experts: {cls.deepseek_n_routed_experts}")
        print(f"  topk: {cls.deepseek_topk} (must be power of 2 for Triton)")
        print(f"  n_shared_experts: {cls.deepseek_n_shared_experts} (must be power of 2 for Triton)")
        print(f"  num_fused_shared_experts: {cls.deepseek_num_fused_shared_experts} (must be power of 2 for Triton)")

    def setUp(self):
        """每个测试方法前的设置"""
        if not self.cuda_available:
            self.skipTest("CUDA not available")
        torch.cuda.empty_cache()  # 清理GPU内存

    def _create_test_data(
        self,
        batch_size: int,
        topk: int,
        num_routed_experts: int,
        dtype: torch.dtype = torch.int64,
        weight_dtype: torch.dtype = torch.float16
    ) -> Tuple[torch.Tensor, torch.Tensor, int, float]:
        """
        创建测试数据

        Args:
            batch_size: batch size (token数量)
            topk: 每个token选中的专家数量
            num_routed_experts: 路由专家总数
            dtype: ID数据类型
            weight_dtype: 权重数据类型

        Returns:
            topk_ids: 选中专家的ID [batch_size, topk]
            topk_weights: 选中专家的权重 [batch_size, topk]
            N: 共享专家的基础ID
            scale_factor: 共享专家的缩放因子
        """
        # 创建随机专家ID，确保在路由专家范围内
        topk_ids = torch.randint(
            0, num_routed_experts, (batch_size, topk), dtype=dtype, device=self.device
        )

        # 创建随机专家权重，归一化到[0, 1]
        topk_weights = torch.rand(batch_size, topk, dtype=weight_dtype, device=self.device)
        topk_weights = F.softmax(topk_weights, dim=-1)

        # 共享专家的基础ID (在路由专家之后)
        N = num_routed_experts

        # 缩放因子 (DeepSeek通常使用较小的权重)
        scale_factor = 0.1

        return topk_ids, topk_weights, N, scale_factor

    def _reference_implementation(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        N: int,
        scale_factor: float,
        num_fused_shared_experts: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参考实现（PyTorch原生）用于精度对比

        Args:
            topk_ids: 选中专家的ID [batch_size, topk]
            topk_weights: 选中专家的权重 [batch_size, topk]
            N: 共享专家的基础ID
            scale_factor: 共享专家的缩放因子
            num_fused_shared_experts: 融合的共享专家数量

        Returns:
            fused_ids: 融合后的专家ID [batch_size, topk + num_fused_shared_experts]
            fused_weights: 融合后的专家权重 [batch_size, topk + num_fused_shared_experts]
        """
        batch_size, topk = topk_ids.shape

        if num_fused_shared_experts <= 0:
            return topk_ids, topk_weights

        # 复制原始的topk专家
        fused_ids = torch.empty(
            batch_size, topk + num_fused_shared_experts,
            dtype=topk_ids.dtype, device=topk_ids.device
        )
        fused_weights = torch.empty(
            batch_size, topk + num_fused_shared_experts,
            dtype=topk_weights.dtype, device=topk_weights.device
        )

        # 复制原始topk数据
        fused_ids[:, :topk] = topk_ids
        fused_weights[:, :topk] = topk_weights

        # 添加共享专家
        for s in range(num_fused_shared_experts):
            shared_expert_id = N + s
            fused_ids[:, topk + s] = shared_expert_id
            fused_weights[:, topk + s] = scale_factor

        return fused_ids, fused_weights

    def _cosine_similarity_weights(self, weights1: torch.Tensor, weights2: torch.Tensor) -> float:
        """
        计算权重tensor的余弦相似度

        Args:
            weights1: 第一个权重tensor
            weights2: 第二个权重tensor

        Returns:
            余弦相似度（0-1之间的值）
        """
        # 展平为一维向量
        flat1 = weights1.flatten().float()
        flat2 = weights2.flatten().float()

        # 计算余弦相似度
        similarity = F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0), dim=1)
        return similarity.item()

    def test_accuracy_deepseek_parameters(self):
        """测试基于DeepSeek参数的精度"""
        print("\n=== 测试精度（DeepSeek-R1参数）===")

        # 使用DeepSeek-R1的参数
        batch_size = 32  # batch size
        topk = self.deepseek_topk
        num_routed_experts = self.deepseek_n_routed_experts
        num_fused_shared_experts = self.deepseek_num_fused_shared_experts

        # 创建测试数据
        topk_ids, topk_weights, N, scale_factor = self._create_test_data(
            batch_size, topk, num_routed_experts
        )

        # 运行待测试的函数
        result_ids, result_weights = fused_append_shared_experts(
            topk_ids, topk_weights, num_fused_shared_experts, scale_factor, N
        )

        # 计算参考结果
        reference_ids, reference_weights = self._reference_implementation(
            topk_ids, topk_weights, N, scale_factor, num_fused_shared_experts
        )

        # 验证专家ID完全一致
        ids_match = torch.equal(result_ids, reference_ids)

        # 计算权重的余弦相似度
        cosine_sim = self._cosine_similarity_weights(result_weights, reference_weights)

        print(f"Batch size: {batch_size}")
        print(f"TopK: {topk}")
        print(f"路由专家数量: {num_routed_experts}")
        print(f"融合共享专家数量: {num_fused_shared_experts}")
        print(f"共享专家基础ID: {N}")
        print(f"缩放因子: {scale_factor}")
        print(f"专家ID一致: {ids_match}")
        print(f"权重余弦相似度: {cosine_sim:.8f}")

        # 断言：专家ID应该完全一致
        self.assertTrue(ids_match, "专家ID不一致")

        # 断言：权重余弦相似度应该大于0.999
        self.assertGreater(cosine_sim, 0.999, f"权重余弦相似度 {cosine_sim} 低于阈值 0.999")

        # 断言：输出形状应该正确
        expected_shape = (batch_size, topk + num_fused_shared_experts)
        self.assertEqual(result_ids.shape, expected_shape)
        self.assertEqual(result_weights.shape, expected_shape)

        print("✓ 精度测试通过")

    def test_accuracy_various_sizes(self):
        """测试不同输入尺寸的精度"""
        print("\n=== 测试不同尺寸的精度 ===")

        test_cases = [
            (1, 2, 8, 1),     # 最小尺寸 (都是2的幂次方)
            (8, 4, 16, 2),    # 小尺寸 (都是2的幂次方)
            (16, 8, 64, 1),   # DeepSeek类似尺寸 (都是2的幂次方)
            (64, 8, 128, 2),  # 大尺寸 (都是2的幂次方)
        ]

        for batch_size, topk, num_routed_experts, num_fused_shared_experts in test_cases:
            with self.subTest(
                batch_size=batch_size, topk=topk,
                num_routed_experts=num_routed_experts,
                num_fused_shared_experts=num_fused_shared_experts
            ):
                print(f"测试尺寸: batch={batch_size}, topk={topk}, experts={num_routed_experts}, shared={num_fused_shared_experts}")

                # 创建测试数据
                topk_ids, topk_weights, N, scale_factor = self._create_test_data(
                    batch_size, topk, num_routed_experts
                )

                # 运行待测试的函数
                result_ids, result_weights = fused_append_shared_experts(
                    topk_ids, topk_weights, num_fused_shared_experts, scale_factor, N
                )

                # 计算参考结果
                reference_ids, reference_weights = self._reference_implementation(
                    topk_ids, topk_weights, N, scale_factor, num_fused_shared_experts
                )

                # 验证专家ID完全一致
                ids_match = torch.equal(result_ids, reference_ids)

                # 计算权重的余弦相似度
                cosine_sim = self._cosine_similarity_weights(result_weights, reference_weights)

                print(f"  专家ID一致: {ids_match}")
                print(f"  权重余弦相似度: {cosine_sim:.8f}")

                # 断言：专家ID应该完全一致
                self.assertTrue(ids_match, f"尺寸 {batch_size}x{topk} 的专家ID不一致")

                # 断言：权重余弦相似度应该大于0.999
                self.assertGreater(cosine_sim, 0.999,
                    f"尺寸 {batch_size}x{topk} 的权重余弦相似度 {cosine_sim} 低于阈值 0.999")

        print("✓ 不同尺寸精度测试通过")

    def test_performance_with_warmup(self):
        """测试性能（包含warmup）"""
        print("\n=== 测试性能（包含warmup）===")

        # 使用DeepSeek-R1的参数
        batch_size = 128  # 较大的batch size测试性能
        topk = self.deepseek_topk
        num_routed_experts = self.deepseek_n_routed_experts
        num_fused_shared_experts = self.deepseek_num_fused_shared_experts
        warmup_iterations = 20
        test_iterations = 200

        # 创建测试数据
        topk_ids, topk_weights, N, scale_factor = self._create_test_data(
            batch_size, topk, num_routed_experts
        )

        # Warmup
        print(f"Warmup {warmup_iterations} 次迭代...")
        for i in range(warmup_iterations):
            fused_append_shared_experts(
                topk_ids, topk_weights, num_fused_shared_experts, scale_factor, N
            )

        # 确保所有CUDA操作完成
        torch.cuda.synchronize()

        # 性能测试
        print(f"性能测试 {test_iterations} 次迭代...")
        start_time = time.perf_counter()

        for i in range(test_iterations):
            fused_append_shared_experts(
                topk_ids, topk_weights, num_fused_shared_experts, scale_factor, N
            )

        torch.cuda.synchronize()  # 确保所有CUDA操作完成
        end_time = time.perf_counter()

        # 计算统计信息
        total_time = end_time - start_time
        avg_time_us = (total_time / test_iterations) * 1_000_000  # 转换为微秒
        throughput = (batch_size * test_iterations) / total_time  # tokens per second

        print(f"输入尺寸: batch={batch_size}, topk={topk}, shared={num_fused_shared_experts}")
        print(f"总时间: {total_time:.6f}s")
        print(f"平均时间: {avg_time_us:.2f} μs")
        print(f"吞吐量: {throughput:.2f} tokens/s")

        # 性能断言：平均时间应该在合理范围内（这里设为100μs作为参考）
        self.assertLess(avg_time_us, 100, f"平均时间 {avg_time_us:.2f} μs 超过阈值 100 μs")

        print("✓ 性能测试通过")

    def test_compute_intensity(self):
        """测试算力强度"""
        print("\n=== 测试算力强度 ===")

        # 使用DeepSeek-R1的参数
        batch_size = 64
        topk = self.deepseek_topk
        num_routed_experts = self.deepseek_n_routed_experts
        num_fused_shared_experts = self.deepseek_num_fused_shared_experts

        # 创建测试数据
        topk_ids, topk_weights, N, scale_factor = self._create_test_data(
            batch_size, topk, num_routed_experts
        )

        # 计算操作数量：主要是内存拷贝操作
        # 总操作数 = batch_size * (topk + num_fused_shared_experts) * 2 (ID + weight)
        total_operations = batch_size * (topk + num_fused_shared_experts) * 2

        # 测试多次运行计算平均时间
        num_runs = 100
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        for _ in range(num_runs):
            fused_append_shared_experts(
                topk_ids, topk_weights, num_fused_shared_experts, scale_factor, N
            )

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        avg_time_s = (end_time - start_time) / num_runs
        ops_per_second = total_operations / avg_time_s
        gops = ops_per_second / 1e9

        print(f"输入尺寸: batch={batch_size}, topk={topk}, shared={num_fused_shared_experts}")
        print(f"总操作数: {total_operations:,}")
        print(f"平均时间: {avg_time_s * 1000:.3f} ms")
        print(f"算力强度: {gops:.2f} GOPS")

        print("✓ 算力强度测试通过")

    def test_cpu_gpu_timing(self):
        """测试CPU和GPU运行时间"""
        print("\n=== 测试CPU/GPU运行时间 ===")

        # 使用DeepSeek-R1的参数
        batch_size = 32
        topk = self.deepseek_topk
        num_routed_experts = self.deepseek_n_routed_experts
        num_fused_shared_experts = self.deepseek_num_fused_shared_experts

        # 创建测试数据
        topk_ids, topk_weights, N, scale_factor = self._create_test_data(
            batch_size, topk, num_routed_experts
        )

        # CPU时间测试（使用PyTorch参考实现）
        topk_ids_cpu = topk_ids.cpu()
        topk_weights_cpu = topk_weights.cpu()

        start_cpu = time.perf_counter()
        reference_ids, reference_weights = self._reference_implementation(
            topk_ids_cpu, topk_weights_cpu, N, scale_factor, num_fused_shared_experts
        )
        end_cpu = time.perf_counter()
        cpu_time_us = (end_cpu - start_cpu) * 1_000_000

        # GPU时间测试（使用Triton实现）
        torch.cuda.synchronize()
        start_gpu = time.perf_counter()
        result_ids, result_weights = fused_append_shared_experts(
            topk_ids, topk_weights, num_fused_shared_experts, scale_factor, N
        )
        torch.cuda.synchronize()
        end_gpu = time.perf_counter()
        gpu_time_us = (end_gpu - start_gpu) * 1_000_000

        # 计算加速比
        speedup = cpu_time_us / gpu_time_us if gpu_time_us > 0 else float('inf')

        print(f"输入尺寸: batch={batch_size}, topk={topk}, shared={num_fused_shared_experts}")
        print(f"CPU时间: {cpu_time_us:.2f} μs")
        print(f"GPU时间: {gpu_time_us:.2f} μs")
        print(f"加速比: {speedup:.2f}x")

        # 验证结果一致性
        result_ids_cpu = result_ids.cpu()
        result_weights_cpu = result_weights.cpu()
        ids_match = torch.equal(result_ids_cpu, reference_ids)
        cosine_sim = self._cosine_similarity_weights(result_weights_cpu, reference_weights)

        print(f"结果一致性 - ID匹配: {ids_match}, 权重相似度: {cosine_sim:.8f}")

        self.assertTrue(ids_match, "CPU和GPU结果ID不一致")
        self.assertGreater(cosine_sim, 0.999, "CPU和GPU结果权重相似度过低")

        print("✓ CPU/GPU时间测试通过")

    # def test_edge_cases(self):
    #     """测试边界情况"""
    #     print("\n=== 测试边界情况 ===")

    #     # 测试最小尺寸
    #     print("测试最小尺寸...")
    #     topk_ids = torch.randint(0, 8, (1, 1), dtype=torch.int64, device=self.device)
    #     topk_weights = torch.rand(1, 1, dtype=torch.float16, device=self.device)
    #     result_ids, result_weights = fused_append_shared_experts(
    #         topk_ids, topk_weights, 1, 0.1, 8
    #     )

    #     # 验证形状
    #     self.assertEqual(result_ids.shape, (1, 2))  # 1 topk + 1 shared
    #     self.assertEqual(result_weights.shape, (1, 2))
    #     print("✓ 最小尺寸测试通过")

    #     # 测试零共享专家
    #     print("测试零共享专家...")
    #     result_ids, result_weights = fused_append_shared_experts(
    #         topk_ids, topk_weights, 0, 0.1, 8
    #     )
    #     self.assertEqual(result_ids.shape, (1, 1))  # 只有原始topk
    #     self.assertEqual(result_weights.shape, (1, 1))
    #     print("✓ 零共享专家测试通过")

    #     # 测试不同的缩放因子
    #     print("测试不同缩放因子...")
    #     scale_factors = [0.01, 0.1, 0.5, 1.0, 2.0]
    #     for scale in scale_factors:
    #         with self.subTest(scale_factor=scale):
    #             result_ids, result_weights = fused_append_shared_experts(
    #                 topk_ids, topk_weights, 1, scale, 8
    #             )
    #             # 验证共享专家的权重是否正确
    #             expected_weight = scale
    #             actual_weight = result_weights[0, 1].item()  # 共享专家的权重
    #             self.assertAlmostEqual(actual_weight, expected_weight, places=5,
    #                 msg=f"缩放因子 {scale} 测试失败，期望 {expected_weight}，实际 {actual_weight}")

    #     print("✓ 不同缩放因子测试通过")
    #     print("✓ 边界情况测试通过")

    def test_different_dtypes(self):
        """测试不同的数据类型"""
        print("\n=== 测试不同数据类型 ===")

        id_dtypes = [torch.int32, torch.int64]
        weight_dtypes = [torch.float16, torch.bfloat16, torch.float32]
        batch_size = 8
        topk = 4
        num_routed_experts = 16

        for id_dtype in id_dtypes:
            for weight_dtype in weight_dtypes:
                with self.subTest(id_dtype=id_dtype, weight_dtype=weight_dtype):
                    print(f"测试数据类型组合: ID={id_dtype}, Weight={weight_dtype}")

                    # 创建测试数据
                    topk_ids = torch.randint(
                        0, num_routed_experts, (batch_size, topk),
                        dtype=id_dtype, device=self.device
                    )
                    topk_weights = torch.rand(
                        batch_size, topk, dtype=weight_dtype, device=self.device
                    )
                    topk_weights = F.softmax(topk_weights, dim=-1)

                    # 运行测试
                    result_ids, result_weights = fused_append_shared_experts(
                        topk_ids, topk_weights, 1, 0.1, num_routed_experts
                    )

                    # 验证数据类型
                    self.assertEqual(result_ids.dtype, id_dtype)
                    self.assertEqual(result_weights.dtype, weight_dtype)

                    # 验证形状
                    expected_shape = (batch_size, topk + 1)
                    self.assertEqual(result_ids.shape, expected_shape)
                    self.assertEqual(result_weights.shape, expected_shape)

                    print(f"  ✓ 数据类型 {id_dtype}/{weight_dtype} 测试通过")

        print("✓ 不同数据类型测试通过")

    # def test_expert_id_validation(self):
    #     """测试专家ID的验证"""
    #     print("\n=== 测试专家ID验证 ===")

    #     batch_size = 16
    #     topk = 4
    #     num_routed_experts = 32
    #     num_fused_shared_experts = 2

    #     # 创建测试数据
    #     topk_ids, topk_weights, N, scale_factor = self._create_test_data(
    #         batch_size, topk, num_routed_experts
    #     )

    #     # 运行函数
    #     result_ids, result_weights = fused_append_shared_experts(
    #         topk_ids, topk_weights, num_fused_shared_experts, scale_factor, N
    #     )

    #     # 验证原始专家ID在路由专家范围内
    #     original_ids = result_ids[:, :topk]
    #     self.assertTrue(torch.all(original_ids >= 0), "专家ID不应为负数")
    #     self.assertTrue(torch.all(original_ids < num_routed_experts), "路由专家ID超出范围")

    #     # 验证共享专家ID正确
    #     shared_ids = result_ids[:, topk:]
    #     for s in range(num_fused_shared_experts):
    #         expected_shared_id = N + s
    #         actual_shared_ids = shared_ids[:, s]
    #         self.assertTrue(torch.all(actual_shared_ids == expected_shared_id),
    #             f"共享专家{s}的ID应为{expected_shared_id}")

    #     # 验证共享专家权重正确
    #     shared_weights = result_weights[:, topk:]
    #     for s in range(num_fused_shared_experts):
    #         expected_weight = scale_factor
    #         actual_weights = shared_weights[:, s]
    #         self.assertTrue(torch.allclose(actual_weights, expected_weight),
    #             f"共享专家{s}的权重应为{expected_weight}")

    #     print("✓ 专家ID验证测试通过")


def run_performance_analysis():
    """运行详细的性能分析"""
    print("=" * 60)
    print("Fused Append Shared Experts 性能分析")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA不可用，跳过性能分析")
        return

    device = torch.device("cuda")

    # 测试不同尺寸的性能
    test_configs = [
        (32, 8, 64, 1),   # DeepSeek基础配置 (都是2的幂次方)
        (64, 8, 64, 1),   # 更大batch (都是2的幂次方)
        (32, 8, 64, 2),   # 更多共享专家 (都是2的幂次方)
        (64, 8, 64, 2),   # 大配置 (都是2的幂次方)
        (128, 8, 64, 1),  # 大batch (都是2的幂次方)
    ]

    for batch_size, topk, num_routed_experts, num_fused_shared_experts in test_configs:
        print(f"\n配置: batch={batch_size}, topk={topk}, experts={num_routed_experts}, shared={num_fused_shared_experts}")
        print("-" * 40)

        # 创建测试数据
        topk_ids = torch.randint(0, num_routed_experts, (batch_size, topk), dtype=torch.int64, device=device)
        topk_weights = torch.rand(batch_size, topk, dtype=torch.float16, device=device)
        topk_weights = F.softmax(topk_weights, dim=-1)
        N = num_routed_experts
        scale_factor = 0.1

        # Warmup
        for _ in range(50):
            fused_append_shared_experts(topk_ids, topk_weights, num_fused_shared_experts, scale_factor, N)

        torch.cuda.synchronize()

        # 时间测试
        num_iterations = 1000
        start_time = time.perf_counter()

        for _ in range(num_iterations):
            result_ids, result_weights = fused_append_shared_experts(
                topk_ids, topk_weights, num_fused_shared_experts, scale_factor, N
            )

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        # 计算性能指标
        avg_time_us = ((end_time - start_time) / num_iterations) * 1_000_000
        throughput_tokens_per_sec = batch_size / ((end_time - start_time) / num_iterations)

        # 计算内存带宽
        input_size = topk_ids.numel() + topk_weights.numel()
        output_size = result_ids.numel() + result_weights.numel()
        total_bytes = (input_size + output_size) * (8 + 2)  # int64 + float16
        bandwidth_gb_per_sec = total_bytes / ((end_time - start_time) / num_iterations) / (1024**3)

        print(f"平均时间: {avg_time_us:.3f} μs")
        print(f"吞吐量: {throughput_tokens_per_sec:.2f} tokens/s")
        print(f"内存带宽: {bandwidth_gb_per_sec:.2f} GB/s")


if __name__ == "__main__":
    # 运行单元测试
    print("运行 Fused Append Shared Experts 单元测试")
    unittest.main(argv=[''], exit=False, verbosity=2)

    # 运行性能分析
    run_performance_analysis()