import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass

class Top_k_per_row_decode_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_rows = config.get("num_rows", 128)
        self.vocab_size = config.get("vocab_size", 32000)
        self.next_n = config.get("next_n", 1)
        # C++ Kernel 硬编码 kTopK = 2048
        self.top_k = 2048
        self.dtype = torch.float32

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "float32/int32")
        state.add_summary("Shape", f"Rows={self.num_rows} Vocab={self.vocab_size} NextN={self.next_n}")
        
        # 元素总数
        total_elements = self.num_rows * self.vocab_size
        state.add_element_count(total_elements)
        
        # 估算显存读写
        # Read: Logits + SeqLens
        # Write: Indices
        read_bytes = (self.num_rows * self.vocab_size * 4) + (self.num_rows * 4)
        write_bytes = (self.num_rows * self.top_k * 4)
        
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            
            logits = torch.randn(self.num_rows, self.vocab_size, dtype=self.dtype, device=dev)
            
            # 构造 seq_lens: 必须确保 seq_len - next_n + (r%next_n) + 1 > 0 且 <= vocab_size
            # Kernel逻辑: rowEnd = seq_len - next_n + (rowIdx % next_n) + 1
            # 简化起见，让 seq_len = vocab_size，这样 rowEnd 可能会略小于 vocab_size
            # 注意: seqLens 的索引是 rowIdx / next_n。这意味着 seqLens 的长度应该是 num_rows / next_n (向上取整)
            # 但提供的 C++ 签名只接受 numRows，内部也是用 rowIdx 访问。
            # Kernel 源码: seq_len = seqLens[rowIdx / next_n];
            # 这意味着 seqLens 长度只需要是 batch_size (即 num_rows / next_n)
            
            # 为了兼容各种 next_n 设置，我们生成一个足够长的 seqLens
            num_seqs = (self.num_rows + self.next_n - 1) // self.next_n
            # 设为 vocab_size - next_n，保证计算出的 rowEnd 不会越界
            base_len = max(self.next_n + 10, self.vocab_size - self.next_n)
            seq_lens = torch.full((num_seqs,), base_len, dtype=torch.int32, device=dev)

            # 输出 Buffer (TopK = 2048)
            indices = torch.empty((self.num_rows, self.top_k), dtype=torch.int32, device=dev)
            
            stride0 = logits.stride(0)
            stride1 = logits.stride(1)

        def launcher(launch):
            stream = self.as_torch_stream(launch.get_stream(), dev_id)
            with torch.cuda.stream(stream):
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    self.next_n,
                    seq_lens,
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
        
        num_seqs = (self.num_rows + self.next_n - 1) // self.next_n
        # 构造 SeqLens，使 rowEnd 在合理范围内
        # rowEnd = seq_len - next_n + (r % next_n) + 1
        # 我们希望 rowEnd <= vocab_size
        safe_len = self.vocab_size - self.next_n - 1
        if safe_len < self.next_n: safe_len = self.vocab_size # Fallback
        
        seq_lens = torch.full((num_seqs,), safe_len, dtype=torch.int32, device=dev)
        
        indices_op = torch.empty((self.num_rows, self.top_k), dtype=torch.int32, device=dev)
        
        stride0 = logits.stride(0)
        stride1 = logits.stride(1)

        # 2. 运行算子
        torch.ops._C.top_k_per_row_decode(
            logits,
            self.next_n,
            seq_lens,
            indices_op,
            self.num_rows,
            stride0,
            stride1
        )

        # 3. Python Reference
        # 必须模拟 Kernel 中的切片逻辑
        ref_values_list = []
        op_values_list = []

        for r in range(self.num_rows):
            # C++ Logic:
            # int seq_len = seqLens[rowIdx / next_n];
            # int rowEnd = seq_len - next_n + (rowIdx % next_n) + 1;
            seq_idx = r // self.next_n
            # 防止 seq_lens 越界 (虽然 num_seqs 计算正确，但防万一)
            s_len = seq_lens[min(seq_idx, len(seq_lens)-1)].item()
            
            row_end = s_len - self.next_n + (r % self.next_n) + 1
            row_end = min(max(0, row_end), self.vocab_size)
            
            # Slice: [0, row_end]
            row_logits = logits[r, :row_end]
            
            # TopK
            cur_k = min(self.top_k, row_logits.size(0))
            if cur_k == 0:
                # Empty slice handling
                ref_v = torch.zeros(self.top_k, device=dev, dtype=self.dtype)
                op_v = torch.zeros(self.top_k, device=dev, dtype=self.dtype)
            else:
                ref_v, _ = torch.topk(row_logits, k=cur_k, sorted=True)
                
                # Gather OP values using indices
                # Indices might be -1 or out of range if not fully populated
                row_indices = indices_op[r, :cur_k].long()
                # Clamp to avoid gather error on invalid indices (if any)
                row_indices = row_indices.clamp(min=0, max=self.vocab_size-1)
                
                op_v = logits[r, row_indices]
                
                # Sort for value comparison
                op_v, _ = torch.sort(op_v, descending=True)
                
                # Padding if needed (though we compare sliced length)
                
            ref_values_list.append(ref_v)
            op_values_list.append(op_v)

        # 拼接所有行的结果进行批量对比
        # 注意：各行长度可能不同 (如果 k 很大)，这里简化处理：
        # 假设 row_end >= top_k，则长度一致。
        # 如果长度不一致，check_diff 扁平化后可能对不齐。
        # 为保险起见，我们只对比那些 row_end >= top_k 的行，或者 pad 到 top_k
        
        flat_ref = torch.cat([t.flatten() for t in ref_values_list])
        flat_op = torch.cat([t.flatten() for t in op_values_list])
        
        # 截断到相同长度 (以防万一)
        min_len = min(flat_ref.numel(), flat_op.numel())
        flat_ref = flat_ref[:min_len]
        flat_op = flat_op[:min_len]

        return self.check_diff(flat_op, flat_ref)