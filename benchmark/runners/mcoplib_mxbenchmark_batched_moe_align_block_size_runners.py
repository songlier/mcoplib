import torch
import math
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._moe_C
except ImportError:
    pass

class Batched_moe_align_block_size_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.max_tokens_per_batch = config.get("max_tokens_per_batch", 65536)
        self.block_size = config.get("block_size", 128)
        self.num_experts = config.get("num_experts", 8)
        self.num_tokens = config.get("num_tokens", 32768)
        
        # 保持 int32 以匹配 C++ Kernel 接口
        self.index_dtype = torch.int32 
        
    def define_metrics(self, state):
        # --- 1. 通用列定义 (Standardization) ---
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "int32")
        
        # 将关键维度合并为 Shape 字符串
        # 格式: (MaxTokens Experts BlockSize)
        shape_str = f"({self.max_tokens_per_batch} {self.num_experts} {self.block_size})"
        state.add_summary("Shape", shape_str)
        
        # --- 2. 吞吐量指标 ---
        # [关键修复] 添加元素计数，NVBench 依赖此项计算 Elem/s
        # 对于此算子，有效负载是输入的总 Token 数
        state.add_element_count(int(self.num_tokens))
        
        # 显存计算逻辑
        aligned_capacity = (self.max_tokens_per_batch + self.block_size - 1) // self.block_size * self.block_size
        rect_sorted_size = self.num_experts * aligned_capacity
        rect_block_size = rect_sorted_size // self.block_size 
        
        # int32 = 4 bytes
        element_size = 4 
        total_in = self.num_experts * element_size
        total_out = (rect_sorted_size + rect_block_size + 1) * element_size
        
        state.add_global_memory_reads(total_in)
        state.add_global_memory_writes(total_out)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            
            # 1. 构造输入
            probs = torch.rand(self.num_experts, device=dev)
            probs = probs / probs.sum()
            expert_num_tokens = (probs * self.num_tokens).to(dtype=self.index_dtype)
            
            # 2. 计算 Output Shapes
            max_capacity_arg = self.max_tokens_per_batch
            aligned_capacity = (max_capacity_arg + self.block_size - 1) // self.block_size * self.block_size
            rect_sorted_size = self.num_experts * aligned_capacity
            rect_block_size = rect_sorted_size // self.block_size
            
            # 3. 分配 Outputs
            sorted_ids = torch.empty(rect_sorted_size, dtype=self.index_dtype, device=dev)
            block_ids_out = torch.empty(rect_block_size, dtype=self.index_dtype, device=dev)
            num_tokens_post_pad = torch.empty(1, dtype=self.index_dtype, device=dev)

        return self.make_launcher(
            dev_id, 
            torch.ops._moe_C.batched_moe_align_block_size,
            max_capacity_arg,
            self.block_size,
            expert_num_tokens,
            sorted_ids,
            block_ids_out,
            num_tokens_post_pad
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        torch.manual_seed(42)
        
        # 1. 准备输入
        probs = torch.rand(self.num_experts, device="cpu")
        probs = probs / probs.sum()
        expert_num_tokens_cpu = (probs * self.num_tokens).to(dtype=self.index_dtype)
        expert_num_tokens = expert_num_tokens_cpu.to(dev)

        # 2. 准备输出
        max_capacity_arg = self.max_tokens_per_batch
        aligned_capacity = (max_capacity_arg + self.block_size - 1) // self.block_size * self.block_size
        rect_sorted_size = self.num_experts * aligned_capacity
        rect_block_size = rect_sorted_size // self.block_size
        
        # 使用 C++ Kernel 的 Sentinel 值进行填充
        fill_val = self.num_experts * max_capacity_arg
        
        sorted_ids = torch.full((rect_sorted_size,), fill_val, dtype=self.index_dtype, device=dev)
        block_ids_out = torch.full((rect_block_size,), -1, dtype=self.index_dtype, device=dev)
        num_tokens_post_pad = torch.zeros(1, dtype=self.index_dtype, device=dev)

        # 3. 运行算子
        torch.ops._moe_C.batched_moe_align_block_size(
            max_capacity_arg,
            self.block_size,
            expert_num_tokens,
            sorted_ids,
            block_ids_out,
            num_tokens_post_pad
        )
        
        # 4. CPU Reference
        counts = expert_num_tokens_cpu.tolist()
        bs = self.block_size
        
        ref_sorted_ids = torch.full((rect_sorted_size,), fill_val, dtype=self.index_dtype)
        ref_block_ids = torch.full((rect_block_size,), -1, dtype=self.index_dtype)
        
        cursor = 0          
        total_post_pad = 0
        
        for e, count in enumerate(counts):
            padded_count = math.ceil(count / bs) * bs
            num_blocks = padded_count // bs
            
            # Fill Block IDs
            start_blk_idx = cursor // bs
            if num_blocks > 0:
                ref_block_ids[start_blk_idx : start_blk_idx + num_blocks] = e
            
            # Fill Sorted IDs (with batch offset logic)
            batch_offset = e * max_capacity_arg
            for k in range(count):
                ref_sorted_ids[cursor + k] = batch_offset + k
            
            cursor += padded_count
            total_post_pad += padded_count
            
        ref_num_tokens_post_pad = torch.tensor([total_post_pad], dtype=self.index_dtype)

        # 5. 验证
        passed_count, diff_count = self.check_diff(num_tokens_post_pad, ref_num_tokens_post_pad.to(dev))
        if not passed_count: return False, diff_count
            
        passed_blk, diff_blk = self.check_diff(block_ids_out, ref_block_ids.to(dev))
        if not passed_blk: return False, diff_blk

        passed_ids, diff_ids = self.check_diff(sorted_ids, ref_sorted_ids.to(dev))
        return passed_ids, diff_ids