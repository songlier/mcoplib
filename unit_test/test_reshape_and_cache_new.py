#!/usr/bin/env python3
"""
OP算子性能测试benchmark代码
用于对比mcoplib_ops和maca_extension_ops两个算子库的性能和精度
"""

import os
import sys
import time
import torch
import math
import numpy as np
from typing import Dict, List, Tuple, Any, Callable, Optional
import argparse
from functools import partial
import mcoplib.lmdeploy
# 添加项目路径到sys.path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置随机种子保证结果可复现
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class OpsBenchmark:
    """算子性能测试基类"""
    
    def __init__(self):

        print("starting to setting device and dtype")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float16  # 默认使用半精度

        try:
            from mcoplib import lmdeploy as mcoplib_ops
            self.mcoplib_ops = mcoplib_ops
            print("✓ Successfully imported mcoplib ops")
        except ImportError as e:
            print(f"✗ Failed to import mcoplib ops: {e}")
            self.mcoplib_ops = None
        try:
            from mcoplib import op as op_rms_norm
            self.op_rms_norm = op_rms_norm
            print("✓ Successfully imported mcoplib rms_norm")
        except ImportError as e:
            print(f"✗ Failed to import mcoplib rms_norm: {e}")
            self.op_rms_norm = None

    
    def measure_time(self, func: Callable, *args, warmup: int = 3, repeat: int = 10, **kwargs) -> Tuple[float, Any]:
        """测量函数执行时间"""
        # 预热
        for _ in range(warmup):
            _ = func(*args, **kwargs)
        
        # 正式测量
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        for _ in range(repeat):
            result = func(*args, **kwargs)
            
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / repeat * 1e6  # 转换为微秒
        return avg_time, result
    
    def compare_ops(self, op_name: str, op_func1: Callable, input_data: Dict[str, Any], shapes_info: str = "") -> Dict[str, Any]:
        """对比两个算子的性能和精度"""
        if self.mcoplib_ops is None:
            return {"error": "One or both ops libraries not available"}
        
        print(f"\n{'='*60}")
        print(f"Testing {op_name} - {shapes_info}")
        print(f"{'='*60}")
        
        try:
            # 测试maca_ext_ops
            time1, result1 = self.measure_time(op_func1, **input_data)
            print(f"mcoplib_time {op_name}: {time1:.2f} μs ")
            

            return {
                "op_name": op_name,
                "shapes_info": shapes_info,
                "mcoplib_time": time1,
                "shapes": {k: v.shape if hasattr(v, 'shape') else str(v) for k, v in input_data.items()}
            }

                
        except Exception as e:
            print(f"Error testing {op_name}: {e}")
            return {"op_name": op_name, "error": str(e)}
    
    def generate_test_data(self, shape: Tuple[int, ...], dtype: torch.dtype = None) -> torch.Tensor:
        """生成测试数据"""
        if dtype is None:
            dtype = self.dtype
        
        # 使用固定随机种子生成数据
        generator = torch.Generator(device=self.device)
        generator.manual_seed(42)
        
        return torch.randn(shape, dtype=dtype, device=self.device, generator=generator)
    
    def run_benchmark(self) -> List[Dict[str, Any]]:
        """运行所有算子的benchmark测试"""
        results = []
        
        # 需要测试的算子列表
        ops_to_test = [
            "reshape_and_cache_new",
        ]
        
        for op_name in ops_to_test:
            try:
                op_results = self.test_single_op(op_name)
                results.extend(op_results)
            except Exception as e:
                print(f"Failed to test {op_name}: {e}")
                results.append({
                    "op_name": op_name,
                    "error": str(e)
                })
        
        return results
    
    def test_single_op(self, op_name: str) -> List[Dict[str, Any]]:
        """测试单个算子"""
        results = []
        
        # 定义不同算子的测试shape组合
        test_shapes = {
            "reshape_and_cache_new": [
                ((128, 8, 64), (256, 8, 8, 64, 8)),    # 小shape: key_cache需要5D [num_blocks, num_heads, head_size/x, block_size, x], value_cache需要4D [num_blocks, num_heads, head_size, block_size]
                ((1024, 32, 128), (2048, 32, 16, 128, 8)),  # 中等shape
            ]
        }
        
        # 获取当前算子的测试shapes
        shapes = test_shapes.get(op_name, [(128, 512)])  # 默认使用小shape
        
        for shape in shapes:
            try:
                if op_name == "reshape_and_cache_new":
                    result = self.test_reshape_and_cache_new(shape)

                else:
                    print(f"Unknown op: {op_name}")
                    continue
                
                results.append(result)
                
            except Exception as e:
                print(f"Failed to test {op_name} with shape {shape}: {e}")
                results.append({
                    "op_name": op_name,
                    "shapes_info": str(shape),
                    "error": str(e)
                })
        
        return results
    
    def test_rms_norm(self, shape: Tuple[int, ...]) -> Dict[str, Any]:
        """测试rms_norm算子"""
        # 生成测试数据
        hidden_states = self.generate_test_data(shape)
        weight = self.generate_test_data((shape[-1],))
        epsilon = 1e-6
        
        # 创建输出张量
        output = torch.empty_like(hidden_states)
        
        # 测试数据
        input_data = {
            "hidden_states": hidden_states,
            "weight": weight,
            "epsilon": epsilon,
            "output": output
        }
        
        def test_mcoplib(**kwargs):
            self.op_rms_norm.rms_norm(
                kwargs["output"],
                kwargs["hidden_states"], 
                kwargs["weight"], 
                kwargs["epsilon"],None, None, False
            )
            return kwargs["output"]
        
        return self.compare_ops("rms_norm", test_mcoplib, input_data, str(shape))
    
    def test_reshape_and_cache_new(self, shape: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Dict[str, Any]:
        key_shape, key_cache_shape = shape
    
        
        # 生成测试数据 - 保存原始数据用于每次测试
        original_key = self.generate_test_data(key_shape)
        original_value = self.generate_test_data(key_shape)
        

        num_blocks = key_cache_shape[0]
        num_heads = key_cache_shape[1] 
        head_size = key_shape[2]  # 使用原始key的head_size
        block_size = key_cache_shape[3]
        
        value_cache_shape = (num_blocks, num_heads, head_size, block_size)
        
        original_key_cache = self.generate_test_data(key_cache_shape)
        original_value_cache = self.generate_test_data(value_cache_shape)
        kv_indices = torch.randint(0, num_blocks, (key_shape[0], 1), device=self.device)
        
        # 测试数据模板（不包含会被修改的缓存数据）
        input_data_template = {
            "kv_indices": kv_indices,
        }

        
        def test_mcoplib(**kwargs):
            # 每次测试都使用原始数据的副本
            key_copy = original_key.clone()
            value_copy = original_value.clone()
            key_cache_copy = original_key_cache.clone()
            value_cache_copy = original_value_cache.clone()
            
            # 按照fill_kv_cache函数的调用方式修正参数
            kv_indices_squeezed = kwargs["kv_indices"].squeeze(-1)
            self.mcoplib_ops.reshape_and_cache_new(
                key_copy, 
                value_copy, 
                key_cache_copy, 
                value_cache_copy, 
                kv_indices_squeezed, 
                "auto", 1.0, 1.0
            )
            return key_cache_copy  # 返回key_cache用于对比
        
        return self.compare_ops("reshape_and_cache_new", test_mcoplib, input_data_template, str(shape))
    


def main():
    """主函数"""
    
    # 创建benchmark实例
    benchmark = OpsBenchmark()
    results = benchmark.run_benchmark()
    
    # 打印汇总结果
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*80}")
    
    successful_tests = [r for r in results if 'error' not in r]
    failed_tests = [r for r in results if 'error' in r]
    
    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")

if __name__ == "__main__":
    main()