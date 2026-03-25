#!/usr/bin/env python3
"""
VLLM 算子 NVBench 性能测试 - 主入口

该脚本整合了所有 vllm 算子的性能测试，包括：
- 基础算子（激活函数、归一化等）
- MOE 算子
- RoPE 算子
- 量化算子

使用方法:
    python run_all_vllm_benchmarks.py --help
    python run_all_vllm_benchmarks.py --list
    python run_all_vllm_benchmarks.py --devices 0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cuda.bench as bench
from vllm_ops_benchmark import (
    SiluAndMulBenchmark,
    GeluAndMulBenchmark,
    RmsNormBenchmark,
    TopkSoftmaxBenchmark,
    MoeSumBenchmark
)
from vllm_extended_benchmark import (
    RotaryEmbeddingBenchmark,
    MulAndSiluBenchmark,
    GeluNewBenchmark,
    StaticScaledFp8QuantBenchmark,
    MoeAlignBlockSizeBenchmark
)


def register_all_benchmarks():
    """注册所有算子的 benchmark"""

    benchmarks = {}

    # ============================================================================
    # 基础算子
    # ============================================================================

    # 激活函数
    silu_and_mul_bench = bench.register(SiluAndMulBenchmark("silu_and_mul").run_benchmark)
    silu_and_mul_bench.add_int64_axis("DType", [0, 1])
    silu_and_mul_bench.add_int64_axis("NumTokens", [512, 1024, 2048, 4096])
    silu_and_mul_bench.add_int64_axis("HiddenSize", [512, 1024, 2048, 4096])
    benchmarks["silu_and_mul"] = silu_and_mul_bench

    mul_and_silu_bench = bench.register(MulAndSiluBenchmark("mul_and_silu").run_benchmark)
    mul_and_silu_bench.add_int64_axis("DType", [0, 1])
    mul_and_silu_bench.add_int64_axis("NumTokens", [512, 1024, 2048, 4096])
    mul_and_silu_bench.add_int64_axis("HiddenSize", [512, 1024, 2048, 4096])
    benchmarks["mul_and_silu"] = mul_and_silu_bench

    gelu_and_mul_bench = bench.register(GeluAndMulBenchmark("gelu_and_mul").run_benchmark)
    gelu_and_mul_bench.add_int64_axis("DType", [0, 1])
    gelu_and_mul_bench.add_int64_axis("NumTokens", [512, 1024, 2048, 4096])
    gelu_and_mul_bench.add_int64_axis("HiddenSize", [512, 1024, 2048, 4096])
    benchmarks["gelu_and_mul"] = gelu_and_mul_bench

    gelu_new_bench = bench.register(GeluNewBenchmark("gelu_new").run_benchmark)
    gelu_new_bench.add_int64_axis("DType", [0, 1])
    gelu_new_bench.add_int64_axis("NumTokens", [512, 1024, 2048, 4096])
    gelu_new_bench.add_int64_axis("HiddenSize", [512, 1024, 2048, 4096])
    benchmarks["gelu_new"] = gelu_new_bench

    # 归一化
    rms_norm_bench = bench.register(RmsNormBenchmark("rms_norm").run_benchmark)
    rms_norm_bench.add_int64_axis("DType", [0, 1])
    rms_norm_bench.add_int64_axis("NumTokens", [512, 1024, 2048, 4096])
    rms_norm_bench.add_int64_axis("HiddenSize", [512, 1024, 2048, 4096])
    benchmarks["rms_norm"] = rms_norm_bench

    # ============================================================================
    # RoPE 算子
    # ============================================================================

    rotary_emb_bench = bench.register(RotaryEmbeddingBenchmark("rotary_embedding").run_benchmark)
    rotary_emb_bench.add_int64_axis("DType", [0, 1])
    rotary_emb_bench.add_int64_axis("NumTokens", [512, 1024, 2048])
    rotary_emb_bench.add_int64_axis("NumHeads", [16, 32])
    rotary_emb_bench.add_int64_axis("HeadSize", [64, 128])
    benchmarks["rotary_embedding"] = rotary_emb_bench

    # ============================================================================
    # 量化算子
    # ============================================================================

    static_scaled_fp8_quant_bench = bench.register(
        StaticScaledFp8QuantBenchmark("static_scaled_fp8_quant").run_benchmark
    )
    static_scaled_fp8_quant_bench.add_int64_axis("DType", [0, 1])
    static_scaled_fp8_quant_bench.add_int64_axis("NumTokens", [512, 1024, 2048, 4096])
    static_scaled_fp8_quant_bench.add_int64_axis("HiddenSize", [512, 1024, 2048, 4096])
    benchmarks["static_scaled_fp8_quant"] = static_scaled_fp8_quant_bench

    # ============================================================================
    # MOE 算子
    # ============================================================================

    topk_softmax_bench = bench.register(TopkSoftmaxBenchmark("topk_softmax").run_benchmark)
    topk_softmax_bench.add_int64_axis("NumTokens", [512, 1024, 2048, 4096])
    topk_softmax_bench.add_int64_axis("NumExperts", [4, 8, 16, 32])
    topk_softmax_bench.add_int64_axis("TopK", [2, 4])
    benchmarks["topk_softmax"] = topk_softmax_bench

    moe_sum_bench = bench.register(MoeSumBenchmark("moe_sum").run_benchmark)
    moe_sum_bench.add_int64_axis("DType", [0, 1])
    moe_sum_bench.add_int64_axis("NumTokens", [512, 1024, 2048, 4096])
    moe_sum_bench.add_int64_axis("HiddenSize", [512, 1024, 2048, 4096])
    moe_sum_bench.add_int64_axis("TopK", [2, 4])
    benchmarks["moe_sum"] = moe_sum_bench

    moe_align_block_size_bench = bench.register(
        MoeAlignBlockSizeBenchmark("moe_align_block_size").run_benchmark
    )
    moe_align_block_size_bench.add_int64_axis("NumTokens", [512, 1024, 2048, 4096])
    moe_align_block_size_bench.add_int64_axis("NumExperts", [4, 8, 16, 32])
    moe_align_block_size_bench.add_int64_axis("BlockSize", [32, 64, 128])
    benchmarks["moe_align_block_size"] = moe_align_block_size_bench

    return benchmarks


def main():
    """主函数"""

    # 注册所有 benchmarks
    benchmarks = register_all_benchmarks()

    # 打印信息
    print("=" * 80)
    print("VLLM 算子 NVBench 性能测试 - 完整版本")
    print("=" * 80)
    print(f"已注册 {len(benchmarks)} 个算子测试:\n")

    # 按类别分组显示
    categories = {
        "激活函数": ["silu_and_mul", "mul_and_silu", "gelu_and_mul", "gelu_new"],
        "归一化": ["rms_norm"],
        "RoPE": ["rotary_embedding"],
        "量化": ["static_scaled_fp8_quant"],
        "MOE": ["topk_softmax", "moe_sum", "moe_align_block_size"]
    }

    for category, ops in categories.items():
        print(f"{category}:")
        for op in ops:
            if op in benchmarks:
                print(f"  ✓ {op}")
        print()

    print("=" * 80)
    print("\n使用方法:")
    print("  列出所有测试:   python run_all_vllm_benchmarks.py --list")
    print("  运行所有测试:   python run_all_vllm_benchmarks.py --devices 0")
    print("  运行特定算子:   python run_all_vllm_benchmarks.py -b silu_and_mul -a NumTokens=1024")
    print("  输出到 JSON:    python run_all_vllm_benchmarks.py --json -o results.json")
    print("  输出到 Markdown: python run_all_vllm_benchmarks.py --markdown -o results.md")
    print("=" * 80)

    # 运行所有 benchmarks
    bench.run_all_benchmarks(sys.argv)


if __name__ == "__main__":
    main()
