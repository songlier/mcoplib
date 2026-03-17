#!/usr/bin/env python3
"""
VLLM 算子 NVBench 性能测试统一框架

该框架提供了统一的接口来测试 mcoplib/op/vllm 和 mcoplib/op/vllm/moe 目录下的所有 CUDA 算子。

使用方法:
    python vllm_ops_benchmark.py --help
    python vllm_ops_benchmark.py --list
    python vllm_ops_benchmark.py --devices 0
"""

import sys
import os
from typing import Dict, List, Tuple, Optional, Callable

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cuda.bench as bench
import torch


# 尝试导入 mcoplib
try:
    import mcoplib._C
except ImportError as e:
    print(f"警告: 无法导入 mcoplib._C: {e}")
    print("请确保 mcoplib 已正确安装")


def as_torch_cuda_stream(cs: bench.CudaStream, dev: int | None) -> torch.cuda.ExternalStream:
    """将 nvbench 的 CudaStream 转换为 PyTorch 的 ExternalStream"""
    return torch.cuda.ExternalStream(
        stream_ptr=cs.addressof(), device=torch.cuda.device(dev)
    )


def get_dtype_info(dtype_id: int) -> Tuple[torch.dtype, str, int]:
    """
    获取数据类型信息

    Args:
        dtype_id: 数据类型 ID (0=float16, 1=float32, 2=bfloat16)

    Returns:
        (dtype, name, element_size)
    """
    dtype_map = {
        0: (torch.float16, "float16", 2),
        1: (torch.float32, "float32", 4),
        2: (torch.bfloat16, "bfloat16", 2)
    }
    return dtype_map.get(dtype_id, (torch.float16, "float16", 2))


class VLLMBenchmarkBase:
    """VLLM 算子 Benchmark 基类"""

    def __init__(self, op_name: str):
        self.op_name = op_name

    def setup_tensors(self, state: bench.State) -> Dict[str, torch.Tensor]:
        """设置测试所需的张量，子类需要实现"""
        raise NotImplementedError

    def execute_op(self, launch: bench.Launch, tensors: Dict[str, torch.Tensor]):
        """执行算子，子类需要实现"""
        raise NotImplementedError

    def run_benchmark(self, state: bench.State) -> None:
        """运行 benchmark 的主函数"""
        # 设置节流阈值
        state.set_throttle_threshold(0.25)

        # 获取设备和流
        dev_id = state.get_device()
        tc_s = as_torch_cuda_stream(state.get_stream(), dev_id)

        # 设置张量
        tensors = self.setup_tensors(state)

        # 定义 launcher
        def launcher(launch: bench.Launch) -> None:
            launch_tc_s = as_torch_cuda_stream(launch.get_stream(), dev_id)
            with torch.cuda.stream(launch_tc_s):
                self.execute_op(launch, tensors)

        # 执行 benchmark
        state.exec(launcher, sync=True)


# ============================================================================
# 激活函数算子测试
# ============================================================================

class SiluAndMulBenchmark(VLLMBenchmarkBase):
    """silu_and_mul 算子测试"""

    def setup_tensors(self, state: bench.State) -> Dict[str, torch.Tensor]:
        num_tokens = state.get_int64_or_default("NumTokens", 1024)
        hidden_size = state.get_int64_or_default("HiddenSize", 512)
        dtype_id = state.get_int64_or_default("DType", 0)

        dtype, dtype_name, element_size = get_dtype_info(dtype_id)
        dev_id = state.get_device()
        tc_s = as_torch_cuda_stream(state.get_stream(), dev_id)

        state.add_summary("DType", dtype_name)
        state.add_summary("NumTokens", str(num_tokens))
        state.add_summary("HiddenSize", str(hidden_size))

        # silu_and_mul 需要输入是偶数维度
        input_size = hidden_size * 2
        total_elements = num_tokens * input_size
        state.add_element_count(total_elements)

        # 计算内存访问
        read_bytes = total_elements * element_size
        write_bytes = num_tokens * hidden_size * element_size
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

        with torch.cuda.stream(tc_s):
            input_tensor = torch.randn((num_tokens, input_size), dtype=dtype, device=f'cuda:{dev_id}')
            output_tensor = torch.empty((num_tokens, hidden_size), dtype=dtype, device=f'cuda:{dev_id}')

        return {"input": input_tensor, "output": output_tensor}

    def execute_op(self, launch: bench.Launch, tensors: Dict[str, torch.Tensor]):
        torch.ops._C.silu_and_mul(tensors["output"], tensors["input"])


class GeluAndMulBenchmark(VLLMBenchmarkBase):
    """gelu_and_mul 算子测试"""

    def setup_tensors(self, state: bench.State) -> Dict[str, torch.Tensor]:
        num_tokens = state.get_int64_or_default("NumTokens", 1024)
        hidden_size = state.get_int64_or_default("HiddenSize", 512)
        dtype_id = state.get_int64_or_default("DType", 0)

        dtype, dtype_name, element_size = get_dtype_info(dtype_id)
        dev_id = state.get_device()
        tc_s = as_torch_cuda_stream(state.get_stream(), dev_id)

        state.add_summary("DType", dtype_name)
        state.add_summary("NumTokens", str(num_tokens))
        state.add_summary("HiddenSize", str(hidden_size))

        input_size = hidden_size * 2
        total_elements = num_tokens * input_size
        state.add_element_count(total_elements)

        read_bytes = total_elements * element_size
        write_bytes = num_tokens * hidden_size * element_size
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

        with torch.cuda.stream(tc_s):
            input_tensor = torch.randn((num_tokens, input_size), dtype=dtype, device=f'cuda:{dev_id}')
            output_tensor = torch.empty((num_tokens, hidden_size), dtype=dtype, device=f'cuda:{dev_id}')

        return {"input": input_tensor, "output": output_tensor}

    def execute_op(self, launch: bench.Launch, tensors: Dict[str, torch.Tensor]):
        torch.ops._C.gelu_and_mul(tensors["output"], tensors["input"])


# ============================================================================
# 归一化算子测试
# ============================================================================

class RmsNormBenchmark(VLLMBenchmarkBase):
    """rms_norm 算子测试"""

    def setup_tensors(self, state: bench.State) -> Dict[str, torch.Tensor]:
        num_tokens = state.get_int64_or_default("NumTokens", 1024)
        hidden_size = state.get_int64_or_default("HiddenSize", 512)
        dtype_id = state.get_int64_or_default("DType", 0)
        epsilon = 1e-6

        dtype, dtype_name, element_size = get_dtype_info(dtype_id)
        dev_id = state.get_device()
        tc_s = as_torch_cuda_stream(state.get_stream(), dev_id)

        state.add_summary("DType", dtype_name)
        state.add_summary("NumTokens", str(num_tokens))
        state.add_summary("HiddenSize", str(hidden_size))

        total_elements = num_tokens * hidden_size
        state.add_element_count(total_elements)

        read_bytes = total_elements * element_size + hidden_size * element_size
        write_bytes = total_elements * element_size
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

        with torch.cuda.stream(tc_s):
            input_tensor = torch.randn((num_tokens, hidden_size), dtype=dtype, device=f'cuda:{dev_id}')
            weight = torch.randn((hidden_size,), dtype=dtype, device=f'cuda:{dev_id}')
            output_tensor = torch.empty((num_tokens, hidden_size), dtype=dtype, device=f'cuda:{dev_id}')

        return {"input": input_tensor, "weight": weight, "output": output_tensor, "epsilon": epsilon}

    def execute_op(self, launch: bench.Launch, tensors: Dict[str, torch.Tensor]):
        torch.ops._C.rms_norm(
            tensors["output"],
            tensors["input"],
            tensors["weight"],
            tensors["epsilon"]
        )


# ============================================================================
# MOE 算子测试
# ============================================================================

class TopkSoftmaxBenchmark(VLLMBenchmarkBase):
    """topk_softmax MOE 算子测试"""

    def setup_tensors(self, state: bench.State) -> Dict[str, torch.Tensor]:
        num_tokens = state.get_int64_or_default("NumTokens", 1024)
        num_experts = state.get_int64_or_default("NumExperts", 8)
        topk = state.get_int64_or_default("TopK", 2)
        renormalize = False

        dev_id = state.get_device()
        tc_s = as_torch_cuda_stream(state.get_stream(), dev_id)

        state.add_summary("NumTokens", str(num_tokens))
        state.add_summary("NumExperts", str(num_experts))
        state.add_summary("TopK", str(topk))

        total_elements = num_tokens * num_experts
        state.add_element_count(total_elements)

        element_size = 4  # float32
        read_bytes = total_elements * element_size
        write_bytes = num_tokens * topk * element_size * 2  # topk_weights + topk_indices
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

        with torch.cuda.stream(tc_s):
            gating_output = torch.randn((num_tokens, num_experts), dtype=torch.float32, device=f'cuda:{dev_id}')
            topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device=f'cuda:{dev_id}')
            topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device=f'cuda:{dev_id}')
            token_expert_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device=f'cuda:{dev_id}')

        return {
            "gating_output": gating_output,
            "topk_weights": topk_weights,
            "topk_indices": topk_indices,
            "token_expert_indices": token_expert_indices,
            "renormalize": renormalize
        }

    def execute_op(self, launch: bench.Launch, tensors: Dict[str, torch.Tensor]):
        torch.ops._C.topk_softmax(
            tensors["topk_weights"],
            tensors["topk_indices"],
            tensors["token_expert_indices"],
            tensors["gating_output"],
            tensors["renormalize"]
        )


class MoeSumBenchmark(VLLMBenchmarkBase):
    """moe_sum MOE 算子测试"""

    def setup_tensors(self, state: bench.State) -> Dict[str, torch.Tensor]:
        num_tokens = state.get_int64_or_default("NumTokens", 1024)
        hidden_size = state.get_int64_or_default("HiddenSize", 512)
        topk = state.get_int64_or_default("TopK", 2)
        dtype_id = state.get_int64_or_default("DType", 0)

        dtype, dtype_name, element_size = get_dtype_info(dtype_id)
        dev_id = state.get_device()
        tc_s = as_torch_cuda_stream(state.get_stream(), dev_id)

        state.add_summary("DType", dtype_name)
        state.add_summary("NumTokens", str(num_tokens))
        state.add_summary("HiddenSize", str(hidden_size))
        state.add_summary("TopK", str(topk))

        total_elements = num_tokens * topk * hidden_size
        state.add_element_count(total_elements)

        read_bytes = total_elements * element_size
        write_bytes = num_tokens * hidden_size * element_size
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

        with torch.cuda.stream(tc_s):
            input_tensor = torch.randn((num_tokens, topk, hidden_size), dtype=dtype, device=f'cuda:{dev_id}')
            output_tensor = torch.empty((num_tokens, hidden_size), dtype=dtype, device=f'cuda:{dev_id}')

        return {"input": input_tensor, "output": output_tensor}

    def execute_op(self, launch: bench.Launch, tensors: Dict[str, torch.Tensor]):
        torch.ops._C.moe_sum(tensors["input"], tensors["output"])


# ============================================================================
# 注册所有 benchmarks
# ============================================================================

def register_all_benchmarks() -> Dict[str, bench.Benchmark]:
    """注册所有算子的 benchmark"""

    benchmarks = {}

    # 激活函数算子
    silu_and_mul_bench = bench.register(SiluAndMulBenchmark("silu_and_mul").run_benchmark)
    silu_and_mul_bench.add_int64_axis("DType", [0, 1])  # float16, float32
    silu_and_mul_bench.add_int64_axis("NumTokens", [512, 1024, 2048])
    silu_and_mul_bench.add_int64_axis("HiddenSize", [512, 1024, 2048])
    benchmarks["silu_and_mul"] = silu_and_mul_bench

    gelu_and_mul_bench = bench.register(GeluAndMulBenchmark("gelu_and_mul").run_benchmark)
    gelu_and_mul_bench.add_int64_axis("DType", [0, 1])
    gelu_and_mul_bench.add_int64_axis("NumTokens", [512, 1024, 2048])
    gelu_and_mul_bench.add_int64_axis("HiddenSize", [512, 1024, 2048])
    benchmarks["gelu_and_mul"] = gelu_and_mul_bench

    # 归一化算子
    rms_norm_bench = bench.register(RmsNormBenchmark("rms_norm").run_benchmark)
    rms_norm_bench.add_int64_axis("DType", [0, 1])
    rms_norm_bench.add_int64_axis("NumTokens", [512, 1024, 2048])
    rms_norm_bench.add_int64_axis("HiddenSize", [512, 1024, 2048])
    benchmarks["rms_norm"] = rms_norm_bench

    # MOE 算子
    topk_softmax_bench = bench.register(TopkSoftmaxBenchmark("topk_softmax").run_benchmark)
    topk_softmax_bench.add_int64_axis("NumTokens", [512, 1024, 2048])
    topk_softmax_bench.add_int64_axis("NumExperts", [4, 8, 16])
    topk_softmax_bench.add_int64_axis("TopK", [2, 4])
    benchmarks["topk_softmax"] = topk_softmax_bench

    moe_sum_bench = bench.register(MoeSumBenchmark("moe_sum").run_benchmark)
    moe_sum_bench.add_int64_axis("DType", [0, 1])
    moe_sum_bench.add_int64_axis("NumTokens", [512, 1024, 2048])
    moe_sum_bench.add_int64_axis("HiddenSize", [512, 1024, 2048])
    moe_sum_bench.add_int64_axis("TopK", [2, 4])
    benchmarks["moe_sum"] = moe_sum_bench

    return benchmarks


if __name__ == "__main__":
    # 注册所有 benchmarks
    benchmarks = register_all_benchmarks()

    print("=" * 60)
    print("VLLM 算子 NVBench 性能测试")
    print("=" * 60)
    print(f"已注册 {len(benchmarks)} 个算子测试:")
    for name in benchmarks.keys():
        print(f"  - {name}")
    print("=" * 60)

    # 运行所有 benchmarks
    bench.run_all_benchmarks(sys.argv)
