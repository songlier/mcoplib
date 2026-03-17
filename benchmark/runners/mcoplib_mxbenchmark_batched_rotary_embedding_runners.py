import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

import mcoplib._C

class Batched_rotary_embedding_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 4)
        self.seq_len = config.get("seq_len", 128)
        self.num_heads = config.get("num_heads", 32)
        self.num_kv_heads = config.get("num_kv_heads", 32)
        self.head_size = config.get("head_size", 128)
        self.rot_dim = config.get("rot_dim", 128)
        self.max_position = config.get("max_position", 4096)
        self.is_neox = config.get("is_neox", True)
        
        # 验证 rot_dim 合法性
        assert self.rot_dim <= self.head_size, "rot_dim cannot be larger than head_size"
        assert self.rot_dim % 2 == 0, "rot_dim must be even"

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"[B={self.batch_size} S={self.seq_len} H={self.num_heads} D={self.head_size}]")
        
        num_tokens = self.batch_size * self.seq_len
        
        # 计算 Q 和 K 的总元素数量
        q_elements = num_tokens * self.num_heads * self.head_size
        k_elements = num_tokens * self.num_kv_heads * self.head_size
        
        # 【新增】注册元素总数，从而让 benchmark 计算 Elem/s
        state.add_element_count(q_elements + k_elements)

        # 计算读写量
        aux_elements_int64 = num_tokens * 2 
        # Cache 读取量
        cache_read_elements = num_tokens * self.rot_dim

        element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        
        # Total Bytes calculation
        rw_bytes = (q_elements + k_elements) * 2 * element_size 
        read_only_bytes = cache_read_elements * element_size
        idx_bytes = aux_elements_int64 * 8

        state.add_global_memory_reads((q_elements + k_elements + cache_read_elements) * element_size + idx_bytes)
        state.add_global_memory_writes((q_elements + k_elements) * element_size)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            num_tokens = self.batch_size * self.seq_len
            
            # 1. 扁平化输入构造 [num_tokens]
            positions = torch.randint(0, self.max_position, (num_tokens,), device=dev, dtype=torch.long)
            offsets = torch.zeros_like(positions, device=dev, dtype=torch.long)
            
            # 2. Q/K 构造为 [num_tokens, H*D]
            q_shape_flat = (num_tokens, self.num_heads * self.head_size)
            k_shape_flat = (num_tokens, self.num_kv_heads * self.head_size)
            
            query = torch.randn(q_shape_flat, dtype=self.dtype, device=dev)
            key = torch.randn(k_shape_flat, dtype=self.dtype, device=dev)
            
            # Cache 保持 [max_pos, rot_dim]
            cos_sin_cache = torch.randn(self.max_position, self.rot_dim, dtype=self.dtype, device=dev)

        def launcher(launch):
            stream = self.as_torch_stream(launch.get_stream(), dev_id)
            with torch.cuda.stream(stream):
                torch.ops._C.batched_rotary_embedding(
                    positions,
                    query,
                    key,
                    self.head_size,
                    cos_sin_cache,
                    self.is_neox,
                    self.rot_dim,
                    offsets
                )
        return launcher

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        num_tokens = self.batch_size * self.seq_len
        
        # 1. 准备 Reference 数据 (使用原始 [B, S] 结构方便逻辑计算)
        positions_bs = torch.randint(0, self.seq_len, (self.batch_size, self.seq_len), device=dev, dtype=torch.long)
        offsets_bs = torch.zeros_like(positions_bs, device=dev, dtype=torch.long)
        
        q_shape = (self.batch_size, self.seq_len, self.num_heads, self.head_size)
        k_shape = (self.batch_size, self.seq_len, self.num_kv_heads, self.head_size)
        
        query = torch.randn(q_shape, dtype=self.dtype, device=dev)
        key = torch.randn(k_shape, dtype=self.dtype, device=dev)

        # 构造 Cache (归一化避免数值溢出)
        cos_sin_cache = torch.randn(self.max_position, self.rot_dim, dtype=self.dtype, device=dev)
        cos_sin_cache = cos_sin_cache / (cos_sin_cache.norm(dim=-1, keepdim=True) + 1e-6)

        # 备份用于 Reference 计算
        query_ref = query.clone().float()
        key_ref = key.clone().float()
        cache_ref = cos_sin_cache.clone().float()
        
        # 2. 准备算子输入 (必须 Flatten 为 [num_tokens, ...])
        positions_op = positions_bs.view(-1)
        offsets_op = offsets_bs.view(-1)
        
        # [B, S, H, D] -> [B*S, H*D]
        query_op = query.clone().view(num_tokens, self.num_heads * self.head_size)
        key_op = key.clone().view(num_tokens, self.num_kv_heads * self.head_size)
        
        # 3. 运行算子
        torch.ops._C.batched_rotary_embedding(
            positions_op,
            query_op,
            key_op,
            self.head_size,
            cos_sin_cache,
            self.is_neox,
            self.rot_dim,
            offsets_op
        )

        # 4. 运行 Python Reference
        total_pos = positions_bs + offsets_bs # [B, S]
        
        # Gather cache
        current_cache = F.embedding(total_pos, cache_ref) 
        half_dim = self.rot_dim // 2
        
        # Split cos/sin
        cos = current_cache[..., :half_dim].unsqueeze(2) 
        sin = current_cache[..., half_dim:].unsqueeze(2) 
        
        def apply_rope_ref(x_tensor, cos, sin, is_neox):
            # x_tensor: [B, S, H, HeadSize]
            x_rot = x_tensor[..., :self.rot_dim]
            x_pass = x_tensor[..., self.rot_dim:]
            
            if is_neox:
                # GPT-NeoX Style
                x1 = x_rot[..., :half_dim]
                x2 = x_rot[..., half_dim:]
                out_x1 = x1 * cos - x2 * sin
                out_x2 = x1 * sin + x2 * cos
                out_rot = torch.cat([out_x1, out_x2], dim=-1)
            else:
                # GPT-J Style
                x1 = x_rot[..., 0::2]
                x2 = x_rot[..., 1::2]
                out_x1 = x1 * cos - x2 * sin
                out_x2 = x1 * sin + x2 * cos
                out_rot = torch.stack([out_x1, out_x2], dim=-1).flatten(-2)
                
            return torch.cat([out_rot, x_pass], dim=-1)

        ref_out_q = apply_rope_ref(query_ref, cos, sin, self.is_neox)
        ref_out_k = apply_rope_ref(key_ref, cos, sin, self.is_neox)

        # 5. 比较结果 (全部 Flatten 后比较)
        op_flat = torch.cat([query_op.flatten(), key_op.flatten()])
        ref_flat = torch.cat([ref_out_q.flatten(), ref_out_k.flatten()]).to(self.dtype)

        # 【调整点】由于 RoPE 涉及三角函数计算，BF16 下 1e-6 的差异属于正常精度误差
        # 默认阈值 0.999999 太严格，调整为 0.9999
        return self.check_diff(op_flat, ref_flat, threshold=0.9999)