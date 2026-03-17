#!/usr/bin/env python3
"""
VLLM 扩展算子 NVBench 性能测试

该文件包含了更多算子的性能测试，包括：
- RoPE (旋转位置编码) 算子
- 量化算子
- 其他激活函数
- 更多的 MOE 算子
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cuda.bench as bench
import torch
from vllm_ops_benchmark import VLLMBenchmarkBase, as_torch_cuda_stream, get_dtype_info


# ============================================================================
# RoPE (旋转位置编码) 算子测试
# ============================================================================

class RotaryEmbeddingBenchmark(VLLMBenchmarkBase):
    """rotary_embedding 算子测试"""

    def setup_tensors(self, state: bench.State) -> dict:
        num_tokens = state.get_int64_or_default("NumTokens", 1024)
        num_heads = state.get_int64_or_default("NumHeads", 32)
        head_size = state.get_int64_or_default("HeadSize", 128)
        dtype_id = state.get_int64_or_default("DType", 0)
        is_neox = True

        dtype, dtype_name, element_size = get_dtype_info(dtype_id)
        dev_id = state.get_device()
        tc_s = as_torch_cuda_stream(state.get_stream(), dev_id)

        state.add_summary("DType", dtype_name)
        state.add_summary("NumTokens", str(num_tokens))
        state.add_summary("NumHeads", str(num_heads))
        state.add_summary("HeadSize", str(head_size))

        total_elements = num_tokens * num_heads * head_size
        state.add_element_count(total_elements * 2)  # query + key

        # 计算内存访问
        cos_sin_cache_size = head_size // 2
        read_bytes = (total_elements * 2 + cos_sin_cache_size) * element_size
        write_bytes = total_elements * 2 * element_size
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

        with torch.cuda.stream(tc_s):
            positions = torch.randint(0, 2048, (num_tokens,), dtype=torch.int64, device=f'cuda:{dev_id}')
            query = torch.randn((num_tokens, num_heads, head_size), dtype=dtype, device=f'cuda:{dev_id}')
            key = torch.randn((num_tokens, num_heads, head_size), dtype=dtype, device=f'cuda:{dev_id}')
            cos_sin_cache = torch.randn((2048, cos_sin_cache_size * 2), dtype=dtype, device=f'cuda:{dev_id}')

        return {
            "positions": positions,
            "query": query,
            "key": key,
            "head_size": head_size,
            "cos_sin_cache": cos_sin_cache,
            "is_neox": is_neox
        }

    def execute_op(self, launch: bench.Launch, tensors: dict):
        torch.ops._C.rotary_embedding(
            tensors["positions"],
            tensors["query"],
            tensors["key"],
            tensors["head_size"],
            tensors["cos_sin_cache"],
            tensors["is_neox"]
        )


# ============================================================================
# 其他激活函数算子测试
# ============================================================================

class MulAndSiluBenchmark(VLLMBenchmarkBase):
    """mul_and_silu 算子测试"""

    def setup_tensors(self, state: bench.State) -> dict:
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

    def execute_op(self, launch: bench.Launch, tensors: dict):
        torch.ops._C.mul_and_silu(tensors["output"], tensors["input"])


class GeluNewBenchmark(VLLMBenchmarkBase):
    """gelu_new 算子测试"""

    def setup_tensors(self, state: bench.State) -> dict:
        num_tokens = state.get_int64_or_default("NumTokens", 1024)
        hidden_size = state.get_int64_or_default("HiddenSize", 512)
        dtype_id = state.get_int64_or_default("DType", 0)

        dtype, dtype_name, element_size = get_dtype_info(dtype_id)
        dev_id = state.get_device()
        tc_s = as_torch_cuda_stream(state.get_stream(), dev_id)

        state.add_summary("DType", dtype_name)
        state.add_summary("NumTokens", str(num_tokens))
        state.add_summary("HiddenSize", str(hidden_size))

        total_elements = num_tokens * hidden_size
        state.add_element_count(total_elements)

        read_bytes = total_elements * element_size
        write_bytes = total_elements * element_size
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

        with torch.cuda.stream(tc_s):
            input_tensor = torch.randn((num_tokens, hidden_size), dtype=dtype, device=f'cuda:{dev_id}')
            output_tensor = torch.empty((num_tokens, hidden_size), dtype=dtype, device=f'cuda:{dev_id}')

        return {"input": input_tensor, "output": output_tensor}

    def execute_op(self, launch: bench.Launch, tensors: dict):
        torch.ops._C.gelu_new(tensors["output"], tensors["input"])


# ============================================================================
# 量化算子测试
# ============================================================================

class StaticScaledFp8QuantBenchmark(VLLMBenchmarkBase):
    """static_scaled_fp8_quant 算子测试"""

    def setup_tensors(self, state: bench.State) -> dict:
        num_tokens = state.get_int64_or_default("NumTokens", 1024)
        hidden_size = state.get_int64_or_default("HiddenSize", 512)
        dtype_id = state.get_int64_or_default("DType", 0)  # 输入类型

        input_dtype, input_dtype_name, input_element_size = get_dtype_info(dtype_id)
        dev_id = state.get_device()
        tc_s = as_torch_cuda_stream(state.get_stream(), dev_id)

        state.add_summary("InputDType", input_dtype_name)
        state.add_summary("NumTokens", str(num_tokens))
        state.add_summary("HiddenSize", str(hidden_size))

        total_elements = num_tokens * hidden_size
        state.add_element_count(total_elements)

        # FP8 输出
        output_element_size = 1

        read_bytes = total_elements * input_element_size
        write_bytes = total_elements * output_element_size
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

        with torch.cuda.stream(tc_s):
            input_tensor = torch.randn((num_tokens, hidden_size), dtype=input_dtype, device=f'cuda:{dev_id}')
            scale = torch.tensor(1.0, dtype=torch.float32, device=f'cuda:{dev_id}')
            # FP8 使用 float8_e4m3fn
            output_tensor = torch.empty((num_tokens, hidden_size), dtype=torch.float8_e4m3fn, device=f'cuda:{dev_id}')

        return {"input": input_tensor, "output": output_tensor, "scale": scale}

    def execute_op(self, launch: bench.Launch, tensors: dict):
        torch.ops._C.static_scaled_fp8_quant(
            tensors["output"],
            tensors["input"],
            tensors["scale"]
        )


# ============================================================================
# 更多 MOE 算子测试
# ============================================================================

class MoeAlignBlockSizeBenchmark(VLLMBenchmarkBase):
    """moe_align_block_size 算子测试"""

    def setup_tensors(self, state: bench.State) -> dict:
        num_tokens = state.get_int64_or_default("NumTokens", 1024)
        num_experts = state.get_int64_or_default("NumExperts", 8)
        block_size = state.get_int64_or_default("BlockSize", 64)
        topk = 2

        dev_id = state.get_device()
        tc_s = as_torch_cuda_stream(state.get_stream(), dev_id)

        state.add_summary("NumTokens", str(num_tokens))
        state.add_summary("NumExperts", str(num_experts))
        state.add_summary("BlockSize", str(block_size))

        total_elements = num_tokens * topk
        state.add_element_count(total_elements)

        element_size = 4  # int32
        read_bytes = total_elements * element_size
        write_bytes = (num_tokens + num_experts + 1) * element_size
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

        with torch.cuda.stream(tc_s):
            topk_ids = torch.randint(0, num_experts, (num_tokens, topk), dtype=torch.int32, device=f'cuda:{dev_id}')
            sorted_token_ids = torch.empty((num_tokens * topk,), dtype=torch.int32, device=f'cuda:{dev_id}')
            experts_ids = torch.empty((num_experts + 1,), dtype=torch.int32, device=f'cuda:{dev_id}')
            num_tokens_post_pad = torch.empty((num_experts,), dtype=torch.int32, device=f'cuda:{dev_id}')

        return {
            "topk_ids": topk_ids,
            "num_experts": num_experts,
            "block_size": block_size,
            "sorted_token_ids": sorted_token_ids,
            "experts_ids": experts_ids,
            "num_tokens_post_pad": num_tokens_post_pad
        }

    def execute_op(self, launch: bench.Launch, tensors: dict):
        torch.ops._C.moe_align_block_size(
            tensors["topk_ids"],
            tensors["num_experts"],
            tensors["block_size"],
            tensors["sorted_token_ids"],
            tensors["experts_ids"],
            tensors["num_tokens_post_pad"]
        )


# ============================================================================
# 注册所有扩展 benchmarks
# ============================================================================

def register_extended_benchmarks() -> dict:
    """注册所有扩展算子的 benchmark"""

    benchmarks = {}

    # RoPE 算子
    rotary_emb_bench = bench.register(RotaryEmbeddingBenchmark("rotary_embedding").run_benchmark)
    rotary_emb_bench.add_int64_axis("DType", [0, 1])
    rotary_emb_bench.add_int64_axis("NumTokens", [512, 1024, 2048])
    rotary_emb_bench.add_int64_axis("NumHeads", [16, 32])
    rotary_emb_bench.add_int64_axis("HeadSize", [64, 128])
    benchmarks["rotary_embedding"] = rotary_emb_bench

    # 其他激活函数
    mul_and_silu_bench = bench.register(MulAndSiluBenchmark("mul_and_silu").run_benchmark)
    mul_and_silu_bench.add_int64_axis("DType", [0, 1])
    mul_and_silu_bench.add_int64_axis("NumTokens", [512, 1024, 2048])
    mul_and_silu_bench.add_int64_axis("HiddenSize", [512, 1024, 2048])
    benchmarks["mul_and_silu"] = mul_and_silu_bench

    gelu_new_bench = bench.register(GeluNewBenchmark("gelu_new").run_benchmark)
    gelu_new_bench.add_int64_axis("DType", [0, 1])
    gelu_new_bench.add_int64_axis("NumTokens", [512, 1024, 2048])
    gelu_new_bench.add_int64_axis("HiddenSize", [512, 1024, 2048])
    benchmarks["gelu_new"] = gelu_new_bench

    # 量化算子
    static_scaled_fp8_quant_bench = bench.register(StaticScaledFp8QuantBenchmark("static_scaled_fp8_quant").run_benchmark)
    static_scaled_fp8_quant_bench.add_int64_axis("DType", [0, 1])
    static_scaled_fp8_quant_bench.add_int64_axis("NumTokens", [512, 1024, 2048])
    static_scaled_fp8_quant_bench.add_int64_axis("HiddenSize", [512, 1024, 2048])
    benchmarks["static_scaled_fp8_quant"] = static_scaled_fp8_quant_bench

    # 更多 MOE 算子
    moe_align_block_size_bench = bench.register(MoeAlignBlockSizeBenchmark("moe_align_block_size").run_benchmark)
    moe_align_block_size_bench.add_int64_axis("NumTokens", [512, 1024, 2048])
    moe_align_block_size_bench.add_int64_axis("NumExperts", [4, 8, 16])
    moe_align_block_size_bench.add_int64_axis("BlockSize", [32, 64, 128])
    benchmarks["moe_align_block_size"] = moe_align_block_size_bench

    return benchmarks


if __name__ == "__main__":
    # 注册所有扩展 benchmarks
    benchmarks = register_extended_benchmarks()

    print("=" * 60)
    print("VLLM 扩展算子 NVBench 性能测试")
    print("=" * 60)
    print(f"已注册 {len(benchmarks)} 个扩展算子测试:")
    for name in benchmarks.keys():
        print(f"  - {name}")
    print("=" * 60)

    # 运行所有 benchmarks
    bench.run_all_benchmarks(sys.argv)
