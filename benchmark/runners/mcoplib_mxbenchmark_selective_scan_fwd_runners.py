import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase
import mcoplib._C

class Selective_scan_fwd_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch = config.get("batch", 2)
        self.dim = config.get("dim", 64)
        self.seqlen = config.get("seqlen", 128)
        self.dstate = config.get("dstate", 16)
        self.n_groups = config.get("n_groups", 1)
        self.delta_softplus = config.get("delta_softplus", True)
        self.pad_slot_id = config.get("pad_slot_id", -1)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"(B={self.batch} D={self.dim} L={self.seqlen})")
        
        # 估算元素总数 (主要关注 Input/Output 的 feature map 大小)
        total = self.batch * self.dim * self.seqlen
        state.add_element_count(total)
        
        # 显存读写估算 (粗略计算 u, delta, z, B, C, ssm_states 的读写)
        element_size = 2 if self.dtype == torch.bfloat16 or self.dtype == torch.float16 else 4
        # Reads: u, delta, A, B, C, D, z, bias
        # Writes: z, ssm_states
        # 简化计算，主要 IO 吞吐
        rw_volume = (total * 3) + (self.batch * self.n_groups * self.dstate * self.seqlen * 2) 
        state.add_global_memory_reads(rw_volume * element_size)
        state.add_global_memory_writes(total * element_size)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            
            # 初始化 Tensor
            u = torch.randn(self.batch, self.dim, self.seqlen, dtype=self.dtype, device=dev)
            delta = torch.randn(self.batch, self.dim, self.seqlen, dtype=self.dtype, device=dev)
            A = -torch.rand(self.dim, self.dstate, dtype=torch.float32, device=dev)
            B = torch.randn(self.batch, self.n_groups, self.dstate, self.seqlen, dtype=self.dtype, device=dev)
            C = torch.randn(self.batch, self.n_groups, self.dstate, self.seqlen, dtype=self.dtype, device=dev)
            D = torch.randn(self.dim, dtype=torch.float32, device=dev)
            z = torch.randn(self.batch, self.dim, self.seqlen, dtype=self.dtype, device=dev)
            delta_bias = torch.randn(self.dim, dtype=torch.float32, device=dev)
            ssm_states = torch.zeros(self.batch, self.dim, self.dstate, dtype=self.dtype, device=dev)
            
            # None 参数
            query_start_loc = None
            cache_indices = None
            has_initial_state = None

            # 封装 operator 调用
            def op_closure():
                return torch.ops._C.selective_scan_fwd(
                    u, delta, A, B, C, D, z, delta_bias, 
                    self.delta_softplus, query_start_loc, cache_indices, 
                    has_initial_state, ssm_states, self.pad_slot_id
                )

        return self.make_launcher(dev_id, op_closure)

    def selective_scan_ref(self, u, delta, A, B, C, D, z, delta_bias):
        # Python 参考实现 (Float32 精度)
        B_sz, D_sz, L_sz = u.shape
        N_sz = A.shape[1]
        G_sz = B.shape[1]
        
        # 1. Delta 预处理
        delta = delta.float()
        delta_bias = delta_bias.float()
        u = u.float()
        z = z.float()
        A = A.float()
        B = B.float()
        C = C.float()
        D = D.float()
        
        if self.delta_softplus:
            delta = F.softplus(delta + delta_bias.view(1, -1, 1))
        
        # 2. 离散化
        # A: (D, N) -> (B, D, N, L)
        deltaA = torch.exp(torch.einsum('bdl,dn->bdnl', delta, A))
        
        # B: (B, G, N, L) -> (B, D, N, L) via repeat if necessary
        if G_sz == 1:
            B = B.repeat(1, D_sz, 1, 1) # (B, D, N, L)
            C = C.repeat(1, D_sz, 1, 1) # (B, D, N, L)
        else:
            # 简单处理 Group > 1 的情况，假设 D 可被 G 整除
            ratio = D_sz // G_sz
            B = B.repeat_interleave(ratio, dim=1)
            C = C.repeat_interleave(ratio, dim=1)

        deltaB_u = torch.einsum('bdl,bdnl,bdl->bdnl', delta, B, u)
        
        # 3. 扫描 (Scan)
        x = torch.zeros(B_sz, D_sz, N_sz, device=u.device)
        ys = []
        for i in range(L_sz):
            x = x * deltaA[:, :, :, i] + deltaB_u[:, :, :, i]
            y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            ys.append(y)
            
        y = torch.stack(ys, dim=2) # (B, D, L)
        
        # 4. 最终输出
        out = y + u * D.view(1, -1, 1)
        out = out * F.silu(z)
        
        return out

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        
        # 创建验证数据
        u = torch.randn(self.batch, self.dim, self.seqlen, dtype=self.dtype, device=dev)
        delta = torch.randn(self.batch, self.dim, self.seqlen, dtype=self.dtype, device=dev)
        A = -torch.rand(self.dim, self.dstate, dtype=torch.float32, device=dev)
        B = torch.randn(self.batch, self.n_groups, self.dstate, self.seqlen, dtype=self.dtype, device=dev)
        C = torch.randn(self.batch, self.n_groups, self.dstate, self.seqlen, dtype=self.dtype, device=dev)
        D = torch.randn(self.dim, dtype=torch.float32, device=dev)
        z = torch.randn(self.batch, self.dim, self.seqlen, dtype=self.dtype, device=dev)
        delta_bias = torch.randn(self.dim, dtype=torch.float32, device=dev)
        ssm_states = torch.zeros(self.batch, self.dim, self.dstate, dtype=self.dtype, device=dev)
        
        # 备份 z 用于 Ref
        z_clone = z.clone() 
        
        # 算子输出 (In-place 修改 z)
        torch.ops._C.selective_scan_fwd(
            u, delta, A, B, C, D, z, delta_bias, 
            self.delta_softplus, None, None, None, ssm_states, self.pad_slot_id
        )
        op_out = z 

        # 参考输出
        ref_out = self.selective_scan_ref(u, delta, A, B, C, D, z_clone, delta_bias)
        
        # 转换回 dtype 进行比较
        ref_out = ref_out.to(self.dtype)
        
        # 检查差异 (传入单一误差阈值 1e-2，修复参数过多的问题)
        return self.check_diff(op_out, ref_out, 1e-2)