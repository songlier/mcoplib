import torch
import math
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._moe_C
except ImportError:
    pass

class Moe_lora_align_block_size_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 16384)
        self.num_experts = config.get("num_experts", 128)
        self.block_size = config.get("block_size", 32)
        self.max_loras = config.get("max_loras", 4)
        self.top_k = config.get("top_k", 1)
        # C++ Kernel 内部使用 int32_t 指针，必须使用 int32 防止内存错位
        self.index_dtype = torch.int32

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "int32")
        shape_str = f"({self.num_tokens} {self.num_experts} {self.max_loras})"
        state.add_summary("Shape", shape_str)
        
        # 元素总数 = Token总数 * TopK
        total_elements = self.num_tokens * self.top_k
        state.add_element_count(total_elements)
        
        # --- 显存访问估算 ---
        # 计算每个 LoRA 的最大容量 (假设极端情况：所有 Token 都去同一个 LoRA)
        capacity_per_lora = (total_elements + self.block_size - 1) // self.block_size * self.block_size
        max_blocks_per_lora = capacity_per_lora // self.block_size
        
        # Read: topk_ids, mapping, adapter_enabled, lora_ids
        read_bytes = (total_elements * 4) + (self.num_tokens * 4) + (self.max_loras * 4 * 2)
        
        # Write: sorted_token_ids, expert_ids, num_tokens_post_pad
        # sorted_token_ids: [max_loras * capacity]
        # expert_ids: [max_loras * max_blocks] (C++ 逻辑)
        # num_tokens_post_pad: [max_loras]
        write_bytes = (self.max_loras * capacity_per_lora * 4) + \
                      (self.max_loras * max_blocks_per_lora * 4) + \
                      (self.max_loras * 4)
        
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            
            # 1. 准备输入
            # topk_ids: [N, K]
            topk_ids = torch.randint(0, self.num_experts, (self.num_tokens, self.top_k), dtype=self.index_dtype, device=dev)
            # token_lora_mapping: [N] (映射到 0 ~ max_loras-1)
            token_lora_mapping = torch.randint(0, self.max_loras, (self.num_tokens,), dtype=self.index_dtype, device=dev)
            # 启用所有 adapter
            adapter_enabled = torch.ones(self.max_loras, dtype=self.index_dtype, device=dev)
            lora_ids = torch.arange(self.max_loras, dtype=self.index_dtype, device=dev)
            
            # 2. 计算输出维度
            total_elements = self.num_tokens * self.top_k
            # max_num_tokens_padded: 单个 LoRA 的最大 Buffer 大小
            max_num_tokens_padded = (total_elements + self.block_size - 1) // self.block_size * self.block_size
            max_num_m_blocks = max_num_tokens_padded // self.block_size
            
            # 3. 准备输出
            sorted_token_ids = torch.empty(self.max_loras * max_num_tokens_padded, dtype=self.index_dtype, device=dev)
            expert_ids = torch.empty(self.max_loras * max_num_m_blocks, dtype=self.index_dtype, device=dev)
            num_tokens_post_pad = torch.empty(self.max_loras, dtype=self.index_dtype, device=dev)

        # 4. 调用算子 (最后传入 None 对应 C++ 的 std::optional maybe_expert_map)
        return self.make_launcher(
            dev_id, torch.ops._moe_C.moe_lora_align_block_size,
            topk_ids, token_lora_mapping,
            self.num_experts, self.block_size, self.max_loras,
            max_num_tokens_padded, max_num_m_blocks,
            sorted_token_ids, expert_ids, num_tokens_post_pad,
            adapter_enabled, lora_ids, None
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        torch.manual_seed(42)
        
        # --- 1. 准备数据 ---
        topk_ids = torch.randint(0, self.num_experts, (self.num_tokens, self.top_k), dtype=self.index_dtype, device=dev)
        token_lora_mapping = torch.randint(0, self.max_loras, (self.num_tokens,), dtype=self.index_dtype, device=dev)
        adapter_enabled = torch.ones(self.max_loras, dtype=self.index_dtype, device=dev)
        lora_ids = torch.arange(self.max_loras, dtype=self.index_dtype, device=dev)
        
        total_elements = self.num_tokens * self.top_k
        max_num_tokens_padded = (total_elements + self.block_size - 1) // self.block_size * self.block_size
        max_num_m_blocks = max_num_tokens_padded // self.block_size
        
        # 初始化输出 (匹配 C++ Kernel 的初始化行为)
        # sorted_token_ids 初始化为 numel (total_elements)
        sorted_token_ids = torch.full((self.max_loras * max_num_tokens_padded,), total_elements, dtype=self.index_dtype, device=dev)
        # expert_ids 初始化为 -1
        expert_ids = torch.full((self.max_loras * max_num_m_blocks,), -1, dtype=self.index_dtype, device=dev)
        # num_tokens_post_pad 初始化为 0
        num_tokens_post_pad = torch.zeros(self.max_loras, dtype=self.index_dtype, device=dev)

        # --- 2. 运行算子 ---
        torch.ops._moe_C.moe_lora_align_block_size(
            topk_ids, token_lora_mapping,
            self.num_experts, self.block_size, self.max_loras,
            max_num_tokens_padded, max_num_m_blocks,
            sorted_token_ids, expert_ids, num_tokens_post_pad,
            adapter_enabled, lora_ids, None
        )
        
        # --- 3. CPU Reference ---
        topk_flat = topk_ids.cpu().flatten()
        mapping_cpu = token_lora_mapping.cpu()
        # 注意: mapping 是对 token 的映射，topk_ids 展平后每个元素对应的 mapping 索引需除以 top_k
        mapping_expanded = mapping_cpu.repeat_interleave(self.top_k)
        
        ref_sorted = torch.full((self.max_loras * max_num_tokens_padded,), total_elements, dtype=self.index_dtype)
        ref_experts = torch.full((self.max_loras * max_num_m_blocks,), -1, dtype=self.index_dtype)
        ref_post_pad = torch.zeros(self.max_loras, dtype=self.index_dtype)
        
        bs = self.block_size
        
        # 模拟 Kernel 逻辑：按 LoRA ID 并行处理
        for lora_idx in range(self.max_loras):
            lid = lora_ids[lora_idx].item()
            if adapter_enabled[lid] == 0: continue
            
            # 筛选属于当前 LoRA 的 Token
            mask = (mapping_expanded == lid)
            relevant_indices = mask.nonzero(as_tuple=True)[0] # 全局展平后的索引
            if len(relevant_indices) == 0: continue
            
            sub_topk = topk_flat[relevant_indices]
            
            # 统计每个 Expert 的数量
            counts = torch.bincount(sub_topk, minlength=self.num_experts)
            
            curr_sorted_offset = lid * max_num_tokens_padded
            curr_expert_offset = lid * max_num_m_blocks
            
            lora_total_padded = 0
            
            for eid in range(self.num_experts):
                cnt = counts[eid].item()
                padded_cnt = math.ceil(cnt / bs) * bs
                
                # Fill Expert IDs
                num_blks = padded_cnt // bs
                if num_blks > 0:
                    start_blk = lora_total_padded // bs
                    ref_experts[curr_expert_offset + start_blk : curr_expert_offset + start_blk + num_blks] = eid
                
                # Fill Sorted IDs (找到属于该 Expert 且属于该 LoRA 的原始索引)
                if cnt > 0:
                    # 在 relevant_indices 中找到 topk 值为 eid 的那些
                    e_mask = (sub_topk == eid)
                    original_indices = relevant_indices[e_mask]
                    ref_sorted[curr_sorted_offset + lora_total_padded : curr_sorted_offset + lora_total_padded + cnt] = original_indices
                
                lora_total_padded += padded_cnt
            
            ref_post_pad[lid] = lora_total_padded

        # --- 4. 验证对比 ---
        # 4.1 验证 num_tokens_post_pad
        passed_pad, diff_pad = self.check_diff(num_tokens_post_pad, ref_post_pad.to(dev))
        if not passed_pad:
            print(f"Post Pad mismatch: GPU={num_tokens_post_pad}, Ref={ref_post_pad}")
            return False, diff_pad
            
        # 4.2 验证 expert_ids
        passed_exp, diff_exp = self.check_diff(expert_ids, ref_experts.to(dev))
        if not passed_exp:
            print("Expert IDs mismatch")
            return False, diff_exp
            
        # 4.3 验证 sorted_token_ids (需排序)
        # GPU 结果拉回 CPU
        gpu_sorted = sorted_token_ids.cpu()
        
        # 对 GPU 结果进行分段排序，因为 atomicAdd 导致同一 Block 内顺序随机
        # 参考 Ref 的生成逻辑，我们知道每个段的长度和位置
        for lora_idx in range(self.max_loras):
            lid = lora_ids[lora_idx].item()
            if adapter_enabled[lid] == 0: continue
            
            # 获取该 LoRA 的总 Padding 后长度
            total_len = ref_post_pad[lid].item()
            if total_len == 0: continue
            
            base_offset = lid * max_num_tokens_padded
            
            # 还需要按 Expert 分段排序，因为 Ref 是按 Expert 顺序填充的
            # 重新计算 counts (为了获取每个 Expert 的段长度)
            mask = (mapping_expanded == lid)
            sub_topk = topk_flat[mask]
            counts = torch.bincount(sub_topk, minlength=self.num_experts)
            
            current_local_offset = 0
            for eid in range(self.num_experts):
                cnt = counts[eid].item()
                padded_cnt = math.ceil(cnt / bs) * bs
                
                if padded_cnt > 0:
                    # 取出 GPU 对应的段
                    start = base_offset + current_local_offset
                    end = start + padded_cnt
                    segment = gpu_sorted[start:end]
                    
                    # 排序：有效 ID ( < total_elements) 在前，Padding (total_elements) 在后
                    sorted_seg, _ = torch.sort(segment)
                    gpu_sorted[start:end] = sorted_seg
                    
                current_local_offset += padded_cnt

        passed_sort, diff_sort = self.check_diff(gpu_sorted, ref_sorted)
        if not passed_sort:
            print("Sorted Token IDs mismatch")
            
        return passed_sort, diff_sort