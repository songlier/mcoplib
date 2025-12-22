import os
import sys
import time
import torch
import torch.nn as nn
import unittest

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from mcoplib.fused_mla import fused_mla_normal_rotary_emb as fused_mla_normal_rotary_emb
from measure_cuda import measure_cuda

def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


# Inverse dim formula to find dim based on number of rotations
def _yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def _yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> Tuple[int, int]:
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(
    low: float, high: float, dim: int, dtype: torch.dtype
) -> torch.Tensor:
    if low == high:
        high += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

class CustomOp(nn.Module):
    def __init__(self):
        super().__init__()
        self._forward_method = self.dispatch_forward()

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args, **kwargs):
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)

    def forward_xpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_hpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def dispatch_forward(self):
        return self.forward_native

def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


class RotaryEmbedding(CustomOp):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        # NOTE(ByronHsu): cache needs to be in FP32 for numerical stability
        cache = cache.to(dtype)
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation of forward()."""
        if offsets is not None:
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

class DeepseekScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
        device: Optional[str] = "cuda",
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation.
        self.mscale = float(
            yarn_get_mscale(self.scaling_factor, float(mscale))
            / yarn_get_mscale(self.scaling_factor, float(mscale_all_dim))
            * attn_factor
        )
        self.device = device
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device=self.device)
            / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (
            1
            - _yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, dtype=torch.float)
        ) * self.extrapolation_factor
        inv_freq_mask = inv_freq_mask.to(self.device)
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(
            self.max_position_embeddings * self.scaling_factor,
            device=self.device,
            dtype=torch.float32,
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.mscale
        sin = freqs.sin() * self.mscale
        cache = torch.cat((cos, sin), dim=-1)
        print("Cache shape", cache.shape)
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim :]
            key_pass = key[..., self.rotary_dim :]

        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(positions.device)
        cos_sin = self.cos_sin_cache[
            torch.add(positions, offsets) if offsets is not None else positions
        ]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            # NOTE(woosuk): Here we assume that the positions tensor has the
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
        # print("cos", cos.shape, cos.dtype, cos)
        # print("sin", sin.shape, sin.dtype, sin)
        # print("query_rot", query_rot.shape, query_rot)
        # print("rotate_fn(query_rot)", rotate_fn(query_rot).shape, rotate_fn(query_rot))
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query, key

_ROPE_DICT: Dict[Tuple, RotaryEmbedding] = {}

def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
) -> RotaryEmbedding:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_scaling is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_scaling_tuple = {
            k: tuple(v) if isinstance(v, list) else v for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None
    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)
    key = (
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling_args,
        dtype,
    )
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    if "rope_type" in rope_scaling:
        scaling_type = rope_scaling["rope_type"]
    elif "type" in rope_scaling:
        scaling_type = rope_scaling["type"]
    else:
        raise ValueError("Unknown RoPE scaling type")

    if scaling_type == "deepseek_yarn":
        scaling_factor = rope_scaling["factor"]
        original_max_position = rope_scaling["original_max_position_embeddings"]
        # assert max_position == original_max_position * scaling_factor
        extra_kwargs = {
            k: v
            for k, v in rope_scaling.items()
            if k
            in (
                "extrapolation_factor",
                "attn_factor",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            )
        }
        rotary_emb = DeepseekScalingRotaryEmbedding(
            head_size,
            rotary_dim,
            original_max_position,
            base,
            is_neox_style,
            scaling_factor,
            dtype,
            **extra_kwargs,
        )
    else:
        raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps, dtype = torch.bfloat16):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size).to(dtype))

    def forward(
            self,
            x: torch.Tensor,
            residual= None,
        ) :
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

def get_rope_wrapper(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
    device: Optional[str] = None,
):
    if device != "cpu":
        return get_rope(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            rope_scaling,
            dtype,
            partial_rotary_factor,
        )

    
def normal_absorb_rotary_emb(
    kv_b_proj,
    kv_a_layernorm,
    rotary_emb,
    q:torch.tensor, # [bs, 128, 192], dtype=bf16
    latent_cache:torch.tensor, # [bs, 576], dtype=bf16
    positions:torch.tensor, # [bs], dtype=int64
    qk_nope_head_dim:int, #128
    qk_rope_head_dim:int, #64
    kv_lora_rank:int, #512
    v_head_dim:int, #128
    num_local_heads:int , #128
    k:torch.tensor
):
    _, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
# Input: q [3201, 128, 192]
# Operation: q_pe = q [ :, : , 0:64]
# Output:
    kv_a, _ = latent_cache.split([kv_lora_rank, qk_rope_head_dim], dim=-1)
# Input: latent_cache = [bs, 576]
# Operation: kv_a = latent_cache[:, 0:512]
    latent_cache = latent_cache.unsqueeze(1)
# Input: latent_cache = [bs, 576]
# Operation: unsqueeze(1)
# Output: latent_cache = [bs, 1, 576]
    kv_a = kv_a_layernorm(kv_a.contiguous())
    # print("kv_a:", kv_a.dtype)
    # print("kv_a:", kv_a.shape)
# Input: kv_a [bs, 512]
# Operation: RMSNorm
# Output: kv_a = [bs, 512]
    kv = kv_b_proj(kv_a)
# Input: kv_a [bs, 512], v_input:
# Operation: kv_b_proj [32768, 512]^T
# Output: kv [bs, 32768]
    kv = kv.view(-1, num_local_heads, qk_nope_head_dim + v_head_dim)
# Input: kv [ba, 32768]
# Operation: kv [bs, 128, 256] = kv [bs, 32768]
    k_nope = kv[..., : qk_nope_head_dim]
# Input: kv [bs, 128, 256]
# Output: k_nope [bs, 128, 128] = kv [bs, 128, 0:128]
    v = kv[..., qk_nope_head_dim :]
# Input: kv [bs, 128, 256]
# Output: v [bs, 128, 128] = kv [bs, 128, 128:256]
    k_pe = latent_cache[:, :, kv_lora_rank :]
# Input: latent_cache [bs, 1, 576]
# Output: k_pe [bs, 1, 64] = latent_cache [bs, 1, 512:576]
    q_pe, k_pe = rotary_emb(positions, q_pe, k_pe)
# Input: q_pe [bs, 128, 64] , k_pe [bs, 1, 64]
# Output: q_pe [bs, 128, 64] , k_pe [bs, 1, 64]
    q[..., qk_nope_head_dim :] = q_pe
    k[..., : qk_nope_head_dim] = k_nope
    k[..., qk_nope_head_dim :] = k_pe
    latent_cache[:, :, : kv_lora_rank] = kv_a.unsqueeze(1)
    latent_cache[:, :, kv_lora_rank :] = k_pe
    return q, k, v, latent_cache

def run_fused_mla_normal(q_len, num_local_heads, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, hidden_size, test_dtype=torch.bfloat16, test_atol=1e-5):
    # q_len = 3201
    # num_local_heads = 128
    # kv_lora_rank = 512
    # qk_rope_head_dim = 64
    # qk_nope_head_dim = 128
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    # v_head_dim = 128
    # num_heads = 128
    # hidden_size = 7168

    kv_a_layernorm = RMSNorm(kv_lora_rank, eps=1e-06, dtype=test_dtype).cuda()
    kv_b_proj = nn.Linear(kv_lora_rank , num_local_heads * (qk_nope_head_dim + v_head_dim), dtype=test_dtype, bias=False).cuda()
    kv_a = torch.zeros(q_len, kv_lora_rank).to(test_dtype).cuda()
    max_position_embeddings = 163840
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
        rope_scaling=rope_scaling,
        is_neox_style=False,
        dtype=torch.float32,
        device="cuda",
    )

    cos_sin_cache = rotary_emb.cos_sin_cache
    print(f"cos_sin_cache.dtype: {cos_sin_cache.dtype}")

    print(f"\n\n================= Profiling q_len={q_len} num_local_heads={num_local_heads} dtype={test_dtype}=================")
    q = torch.randn(q_len, num_local_heads, qk_head_dim).to(test_dtype).cuda()
    q2 = q.clone().detach()
    k = torch.empty_like(q).to(test_dtype).cuda()
    # k = torch.zeros(3201,128, 192).to(test_dtype).cuda()
    k1 = k.clone().detach()
    v = torch.zeros(q_len, num_local_heads, v_head_dim).to(test_dtype).cuda()
    latent_cache = torch.rand(q_len, kv_lora_rank + qk_rope_head_dim, dtype=test_dtype).cuda()
    latent_cache2 = latent_cache.clone().detach()
    positions = torch.arange(1, q_len + 1, dtype=torch.int64).cuda()

    legacy_q_input, legacy_k_input, legacy_v_input, legacy_latent_cache = normal_absorb_rotary_emb(kv_b_proj, kv_a_layernorm, rotary_emb, q, latent_cache, positions, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, num_local_heads, k)
    #   print("legacy_latent_cache : ", legacy_latent_cache.shape)
    device = 0
    torch.cuda.set_device(device)
    def kernel():

        fused_q_input, fused_k_input, fused_v_input, fused_latent_cache = fused_mla_normal_rotary_emb(kv_a, kv_b_proj, q2, latent_cache2, positions, cos_sin_cache, kv_a_layernorm.weight, k1, v, q_len, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, num_local_heads)
    stats = measure_cuda(kernel, iters=200, warmup=20, device=device)
    print("PyTorch matmul stats (microseconds):")
    print(f"mean={stats['mean_us']:.3f}µs median={stats['median_us']:.3f}µs min={stats['min_us']:.3f}µs stdev={stats['stdev_us']:.3f}µs")
    #   print("fused_latent_cache : ", fused_latent_cache.shape)
    # show_error(legacy_q_input, fused_q_input, "DIFF ERROR OF Q_INPUT")
    # show_error(legacy_k_input, fused_k_input, "DIFF ERROR OF K_INPUT")
    # show_error(legacy_v_input, fused_v_input, "DIFF ERROR OF V_INPUT")
    # show_error(legacy_latent_cache, fused_latent_cache.unsqueeze(1), "DIFF ERROR OF LATENT_CACHE")
    #   show_error(legacy_kv_a, fused_kv_a, "DIFF ERROR OF KV_A")
    # assert torch.allclose(legacy_q_input, fused_q_input, rtol=1e-05, atol=1e-05, equal_nan=False)
    # assert torch.allclose(legacy_k_input, fused_k_input, rtol=1e-05, atol=test_atol, equal_nan=False)
    # assert torch.allclose(legacy_v_input, fused_v_input, rtol=1e-05, atol=test_atol, equal_nan=False)
    # assert torch.allclose(legacy_latent_cache, fused_latent_cache.unsqueeze(1), rtol=1e-05, atol=test_atol, equal_nan=False)

def fused_mla_normal_func(dtype, test_atol):
    #for q_len in (1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,1017,1024,2039,2041,2048, 3201):
    #    for num_local_heads in (8, 16,32,64,128):
    run_fused_mla_normal(2048, num_local_heads=32, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, hidden_size=7168, test_dtype=dtype, test_atol=test_atol)

class TestFusedMlaNormal(unittest.TestCase):
    def test_fused_mla_normal_bfloat16(self):
        fused_mla_normal_func(torch.bfloat16, 1e-5)

    # def test_fused_mla_normal_float16(self):
    #     fused_mla_normal_func(torch.float16, 1e-2)

#只支持bfloat16
if __name__ == '__main__':
    unittest.main()
