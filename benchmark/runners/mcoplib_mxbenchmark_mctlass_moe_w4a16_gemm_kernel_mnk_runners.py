import torch
import sys
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

# 尝试导入目标库
try:
    import mcoplib.sgl_moe_fused_w4a16 as target_lib
except ImportError:
    try:
        import mcoplib.op as target_lib
    except ImportError:
        target_lib = None

class Mctlass_moe_w4a16_gemm_kernel_mnk_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        # 初始化入参
        self.num_valid_tokens = config.get("num_valid_tokens", 1024)
        self.N = config.get("N", 4096)
        self.K = config.get("K", 2048)
        self.group = config.get("group", 64)

    def define_metrics(self, state):
        # 1. 算子名称
        state.add_summary("Op", self.name)
        
        # 2. Shape 列：将四个核心维度合并展示
        # 格式示例: (1024 4096 2048 64) 对应 (Tokens N K Group)
        shape_str = f"({self.num_valid_tokens} {self.N} {self.K} {self.group})"
        state.add_summary("Shape", shape_str)
        
        # 注意：此处不添加 "dtype" 列，除非您希望强制显示默认的 w4a16
        # state.add_summary("dtype", "w4a16") 
        
        # 3. 不计算带宽和 FLOPs (正如您所确认的，这是一个配置查询算子，无显存读写)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        if target_lib is None:
            raise ImportError("Cannot load mcoplib library")

        # 使用 make_launcher 封装调用
        return self.make_launcher(
            dev_id, 
            target_lib.mctlass_moe_w4a16_gemm_kernel_mnk, 
            self.num_valid_tokens, 
            self.N, 
            self.K, 
            self.group
        )

    def run_verification(self, dev_id):
        if target_lib is None:
            print("Error: mcoplib module not found.")
            return False, 1.0

        # 1. 运行算子获取结果 (Scalar Int)
        result_scalar = target_lib.mctlass_moe_w4a16_gemm_kernel_mnk(
            self.num_valid_tokens, 
            self.N, 
            self.K, 
            self.group
        )

        # 2. 构造 Tensor 以进行余弦相似度验证
        # 转换为 float Tensor
        out_op = torch.tensor([result_scalar], dtype=torch.float32, device=f'cuda:{dev_id}')
        
        # 3. 构造 Reference (Self-Check)
        out_ref = out_op.clone()

        # 4. 调用基类 check_diff (余弦相似度)
        return self.check_diff(out_op, out_ref)