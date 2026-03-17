import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass

class Top_k_per_row_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_rows = config.get("num_rows", 128)
        self.vocab_size = config.get("vocab_size", 32000)
        # C++ sampler.cu 中模板硬编码 kTopK = 2048
        self.top_k = 2048 
        self.dtype = torch.float32 

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "float32/int32")
        state.add_summary("Shape", f"Rows={self.num_rows} Vocab={self.vocab_size} K={self.top_k}")
        
        # 元素总数 (Logits 中的元素)
        total_elements = self.num_rows * self.vocab_size
        state.add_element_count(total_elements)
        
        # 显存读写计算
        # Read: Logits (float32) + RowStarts(int32) + RowEnds(int32)
        # Write: Indices (int32)
        read_bytes = (self.num_rows * self.vocab_size * 4) + (self.num_rows * 4 * 2)
        write_bytes = (self.num_rows * self.top_k * 4)
        
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            
            # 1. 准备输入
            logits = torch.randn(self.num_rows, self.vocab_size, dtype=self.dtype, device=dev)
            row_starts = torch.zeros(self.num_rows, dtype=torch.int32, device=dev)
            row_ends = torch.full((self.num_rows,), self.vocab_size, dtype=torch.int32, device=dev)
            
            # 2. 准备输出
            indices = torch.empty((self.num_rows, self.top_k), dtype=torch.int32, device=dev)
            
            stride0 = logits.stride(0)
            stride1 = logits.stride(1)

        def launcher(launch):
            stream = self.as_torch_stream(launch.get_stream(), dev_id)
            with torch.cuda.stream(stream):
                torch.ops._C.top_k_per_row(
                    logits,
                    row_starts,
                    row_ends,
                    indices,
                    self.num_rows,
                    stride0,
                    stride1
                )
        return launcher

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        
        # 1. 准备数据
        logits = torch.randn(self.num_rows, self.vocab_size, dtype=self.dtype, device=dev)
        row_starts = torch.zeros(self.num_rows, dtype=torch.int32, device=dev)
        row_ends = torch.full((self.num_rows,), self.vocab_size, dtype=torch.int32, device=dev)
        indices_op = torch.empty((self.num_rows, self.top_k), dtype=torch.int32, device=dev)
        
        stride0 = logits.stride(0)
        stride1 = logits.stride(1)

        # 2. 运行算子
        torch.ops._C.top_k_per_row(
            logits,
            row_starts,
            row_ends,
            indices_op,
            self.num_rows,
            stride0,
            stride1
        )

        # 3. 运行 Python Reference (torch.topk)
        current_k = min(self.top_k, self.vocab_size)
        ref_values, ref_indices = torch.topk(logits, k=current_k, dim=1, sorted=True)
        
        # 4. 验证逻辑修改：对比数值而不是对比索引
        # Top-K 的核心是选出最大的 K 个值。如果存在相同数值，索引可能不同，但数值应该一致。
        
        # 使用算子输出的 indices 从 logits 中取值 (Gather)
        # indices_op shape: [rows, top_k]
        # values_op shape: [rows, top_k]
        values_op = torch.gather(logits, 1, indices_op.long())
        
        # 将 Gather 出来的数值与 Reference 的数值进行对比
        # 同样需要排序后再对比，因为 TopK 输出并未严格保证行内有序性（虽然通常是有序的）
        values_op_sorted, _ = torch.sort(values_op, dim=1, descending=True)
        ref_values_sorted, _ = torch.sort(ref_values, dim=1, descending=True)
        
        # 如果 TopK 大于 VocabSize，需要截断有效部分
        if values_op_sorted.size(1) > ref_values_sorted.size(1):
             values_op_sorted = values_op_sorted[:, :ref_values_sorted.size(1)]
        
        # 使用 check_diff 对比数值相似度
        # 数值完全一致则余弦相似度为 1.0
        return self.check_diff(values_op_sorted, ref_values_sorted)