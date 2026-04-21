import os
import time
import torch
import torch.nn.functional as F
from torch import nn
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from torch.profiler import profile, record_function, ProfilerActivity
import argparse
#from sgl_kernel import fused_mla_absorb_rotary_emb
import mcoplib.sgl_kernel
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size).to(torch.bfloat16))

    def forward(
        self,
        x: torch.Tensor,
        residual= None,
    ):
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype) * self.weight
        if residual is None:
            return x
        else:
            return x, residual

def fused_forward_absorb(
    q:torch.Tensor, # [bs, 128, 192], dtype=bf16
    w_kc:torch.Tensor, # [128, 128, 512], dtype=bf16
    latent_cache:torch.Tensor, # [bs, 576], dtype=bf16
    cos_sin_cache:torch.Tensor,  # [max_position_embeddings, 64], dtype=bf16
    positions:torch.Tensor, # [bs], dtype=int64
    norm_weight:torch.Tensor, # [512], dtype=bf16
    q_input:torch.Tensor, # [bs, 128, 576], dtype=bf16
    k_input:torch.Tensor, # [bs, 1, 576], dtype=bf16
    v_input:torch.Tensor, # [bs, 1, 512]
    q_len:int, #16
    num_local_heads:int, #128,
    kv_lora_rank:int, # 512
    qk_rope_head_dim:int, #64
    qk_nope_head_dim:int, #128
):
    eps = 1e-5
    out = torch.ops.sgl_kernel.fused_mla_absorb_rotary_emb(q, w_kc, latent_cache, cos_sin_cache, positions, norm_weight, q_input, k_input, v_input, q_len, num_local_heads, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim,eps)
    if out != 0:
        print("Failed to call fusedMLA.[fused_forward_absorb]")
    return q_input, k_input, v_input
	
def mla_absorb_rotary_emb(
    kv_a_layernorm,
    rotary_emb,
    q:torch.Tensor, # [bs, 128, 192], dtype=bf16
    w_kc:torch.Tensor, # [128, 128, 512], dtype=bf16
    latent_cache:torch.Tensor, # [bs, 576], dtype=bf16
    positions:torch.Tensor, # [bs], dtype=int64
    q_input:torch.Tensor, # [bs, 128, 576], dtype=bf16
    k_input:torch.Tensor, # [bs, 1, 576], dtype=bf16
    v_input:torch.Tensor, # [bs, 1, 512]
    q_len:int, # 16
    num_local_heads:int, # 128,
    kv_lora_rank:int, # 512
    qk_rope_head_dim:int, # 64
    qk_nope_head_dim:int, # 128
):

    q_nope, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    # q_input = torch.zeros(
    #     q_len, num_local_heads, kv_lora_rank + qk_rope_head_dim
    # )
    q_nope_out = torch.bmm(q_nope.transpose(0, 1), w_kc)
    q_input[..., : kv_lora_rank] = q_nope_out.transpose(0, 1)
# Input: q_nope_out [128, bs, 512], q_input [16, 128, 576]
# Operation: 1. q_nope_out.transpose -> [bs, 128, 512]
#            2. value q_input[:, :, 0:512] = [bs, 128, 512]
# Output:
    v_input = latent_cache[..., : kv_lora_rank] # v_input [bs, 512]
# Input: v_input = [bs, 512]
# Operation: v_input = latent_cache[:, 0:512]
    v_input = kv_a_layernorm(v_input.contiguous()).unsqueeze(1) #[bs, 1, 512]
# Input: v_input = [bs, 512]
# Operation: RMSNormal
# Output: v_input = [bs, 1, 512]
    k_input = latent_cache.unsqueeze(1) # [bs, 1, 512+64]
# Input: latent_cache [bs, 576]
# Output: k_input = [bs, 1, 576]
    k_input[..., : kv_lora_rank] = v_input #
# Input: k_input [bs, 1, 576], v_input:
# Operation: k_input[:, :, 0:512] = v_input

    k_pe = k_input[..., kv_lora_rank :] # [bs, 1, 64]
# Input: k_input [bs, 1, 576]
# Operation: k_pe [bs, 1, 64] = k_input[:, :, 512:576]

    q_pe, k_pe = rotary_emb(positions, q_pe, k_pe)
# Input: positions = [bs], q_pe = [bs, 128, 64], k_pe = [bs, 1, 64]
# Operation: rotary_emb
# Output: q_pe = [bs, 128, 64], k_pe = [bs, 1, 64]

    q_input[..., kv_lora_rank :] = q_pe   # [bs, 128, 512+64]
    k_input[..., kv_lora_rank :] = k_pe   # [bs, 1, 512+64]
# q_input[bs, 128, 576], k_input[bs, 1, 576]
# q_input[:, :, 512:576] = q_pe[:, :, :], k_input[:, :, 512:576] = k_pe[:, :, :]
# Input: q_pe, k_pe
    return q_input, k_input, v_input

with_profile=False

def show_error(golden, v, tag="DIFF ERROR"):
    errors = torch.abs(golden - v)

    errors_max = torch.max(errors)
    errors_ave = torch.sum(errors) / errors.numel()

    max_idx_flat = torch.argmax(errors)
    max_idx = torch.unravel_index(max_idx_flat, errors.shape)

    golden_val = golden[max_idx]
    v_val = v[max_idx]

    print(f"{tag}: error_max={errors_max}, error_ave={errors_ave}, max_error_idx={max_idx}")
    print(f"golden[{max_idx}]={golden_val}, v[{max_idx}]={v_val}")
def print_profiler_summary(prof, max_key_len=50):
    events = prof.key_averages()
    events = sorted(events, key=lambda x: (x.device_time_total / x.count) if x.count > 0 else 0, reverse=True)

    print(f"{'Name':<{max_key_len}} | {'CPU Time Avg (us)':>20} | {'CUDA Time Avg (us)':>20} | {'Count':>10}")
    
    total_cpu_time = 0.0
    total_cuda_time = 0.0
    
    for evt in events:
        if evt.count == 0:
            continue
            
        cpu_time_avg = evt.cpu_time_total / evt.count
        cuda_time_avg = evt.device_time_total / evt.count
        key_str = evt.key
        
        if len(key_str) > max_key_len:
            key_str = key_str[:max_key_len-3] + '...'
            
        print(f"{key_str.ljust(max_key_len)} | {cpu_time_avg:20.2f} | {cuda_time_avg:20.2f} | {evt.count:10}")
        
        total_cpu_time += cpu_time_avg
        total_cuda_time += cuda_time_avg
        
    print("-" * (max_key_len + 55))
    print(f"{'Total'.ljust(max_key_len)} | {total_cpu_time:20.2f} | {total_cuda_time:20.2f} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="profile", choices=["profile", "acc"])
    args = parser.parse_args()

    q_len = 32
    num_local_heads = 64        # 128 -> 64
    kv_lora_rank = 512
    qk_nope_head_dim = 192      # 128 -> 192
    qk_rope_head_dim = 64
    hidden_size = 6144          # 7168 -> 6144

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    kv_a_layernorm = RMSNorm(kv_lora_rank, eps=1e-05).cuda()   # eps: 1e-6 -> 1e-5
    max_position_embeddings = 202752   # 163840 -> 202752
    # rope_theta = 1000000             # 10000 -> 1000000
    rope_theta = 10000
    rope_scaling = {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 40,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 4096,
        "type": "yarn"
    }
    rope_scaling["rope_type"] = "deepseek_yarn"
    rotary_emb = get_rope_wrapper(
        qk_rope_head_dim,
        rotary_dim=qk_rope_head_dim,
        max_position=max_position_embeddings,
        base=rope_theta,
        rope_scaling=rope_scaling,         # GLM5 uses default rope (no yarn scaling)
        is_neox_style=False,
        dtype=torch.bfloat16,
        device="cuda",
    )
    
    cos_sin_cache = rotary_emb.cos_sin_cache
	
    for q_len in [32, 64]: # GLM5 decode/mtp/prefill
        print(f"\n\n================ Profiling q_len={q_len} ================")
        q = (torch.rand(q_len, num_local_heads, qk_nope_head_dim+qk_rope_head_dim, dtype=torch.bfloat16).cuda() - 0.5)/10
        w_kc = (torch.rand(num_local_heads, qk_nope_head_dim, kv_lora_rank, dtype=torch.bfloat16).cuda() - 0.5) / 10

        shape = (q_len, kv_lora_rank+qk_rope_head_dim)
        strides = (576, 1) # contiguous stride for GLM5 (kv_lora_rank+qk_rope_head_dim=576)
        storage_size = (shape[0] - 1) * strides[0] + (shape[1] - 1) * strides[1] + 1
        latent_cache_storage = (torch.rand(storage_size, dtype=torch.bfloat16).cuda() - 0.5) / 10
        latent_cache = torch.as_strided(latent_cache_storage, size=shape, stride=strides)
        latent_cache2_storage = latent_cache_storage.clone().detach()
        latent_cache2 = torch.as_strided(latent_cache2_storage, size=shape, stride=strides)
        q_input = torch.zeros(q_len, num_local_heads, kv_lora_rank + qk_rope_head_dim, dtype=torch.bfloat16).cuda()
        k_input = torch.zeros(q_len, 1, kv_lora_rank + qk_rope_head_dim, dtype=torch.bfloat16).cuda()
        v_input = torch.zeros(q_len, 1, kv_lora_rank, dtype=torch.bfloat16).cuda()
        fused_q_input = torch.zeros(q_len, num_local_heads, kv_lora_rank + qk_rope_head_dim, dtype=torch.bfloat16).cuda()
        fused_k_input = torch.zeros(q_len, 1, kv_lora_rank + qk_rope_head_dim, dtype=torch.bfloat16).cuda()
        fused_v_input = torch.zeros(q_len, 1, kv_lora_rank, dtype=torch.bfloat16).cuda()
        positions = torch.arange(1, q_len + 1, dtype=torch.int64).cuda()
        print("w_kc stride:", w_kc.stride())
        w_kc_1 = w_kc.clone().detach()
        print("w_kc1.stride: ", w_kc_1.transpose(1, 2).contiguous().transpose(1, 2).stride())
        latent_cache3 = latent_cache.clone().detach()

        print("latent_cache stride:", latent_cache.stride())
        print("latent_cache2 stride:", latent_cache2.stride())
        print("latent_cache3 stride:", latent_cache3.contiguous().stride())

        legacy_q_input, legacy_k_input, legacy_v_input = mla_absorb_rotary_emb(kv_a_layernorm, rotary_emb, q, w_kc, latent_cache, positions, q_input, k_input, v_input, q_len, num_local_heads, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim)
        fused_forward_absorb(q, w_kc, latent_cache2, cos_sin_cache, positions, kv_a_layernorm.weight, fused_q_input, fused_k_input, fused_v_input, q_len, num_local_heads, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim)
        show_error(legacy_q_input, fused_q_input, "DIFF ERROR OF Q_INPUT")
        show_error(legacy_k_input, fused_k_input, "DIFF ERROR OF K_INPUT")
        show_error(legacy_v_input, fused_v_input, "DIFF ERROR OF V_INPUT")