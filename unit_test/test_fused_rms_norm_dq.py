# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Union

import pytest
import torch

import vllm._custom_ops as ops
from vllm.model_executor.layers.layernorm import RMSNorm
import mcoplib._C

DTYPES = [torch.bfloat16, torch.float]
QUANT_DTYPES = [torch.int8]
VEC_HIDDEN_SIZES = range(1024, 1030)
# Avoid combinatorial explosion with full Cartesian product
NUM_TOKENS_HIDDEN_SIZES = [
    *[(1, i) for i in [1, 64, *VEC_HIDDEN_SIZES, 5120, 5137]],
    *[(83, i) for i in [1, 1033, 2048, 5120]],
    *[(2048, i) for i in [1, 64, *VEC_HIDDEN_SIZES, 5137]],
    *[(4096, i) for i in [1, 64, 5137]],
]

ADD_RESIDUAL = [False, True]
SCALE_UBS = [True, False]
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]

EPS = 1e-6

## Helpers
def scaled_int8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    azp: Optional[torch.Tensor] = None,
    symmetric: bool = True
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantize the input tensor to int8 and return the quantized tensor and scale, and maybe azp.

    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.
        azp: Optional zero-point for the int8 quantization.
            Must be provided for asymmetric quantization if `scale` is provided.
        symmetric: Whether to use symmetric quantization (scale only, azp ignored).

    Returns:
      tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]] : Output int8 tensor, scales, and optionally azp.
    """
    output = torch.empty_like(input, dtype=torch.int8)
    if scale is not None:
        # static-per-tensor quantization.
        assert symmetric == (
            azp
            is None), "azp must only be provided for asymmetric quantization."
        torch.ops._C.static_scaled_int8_quant(output, input, scale, azp)
        return output, scale, azp

    # dynamic-per-token quantization.
    input_scales = torch.empty((input.numel() // input.shape[-1], 1),
                               device=input.device,
                               dtype=torch.float32)
    input_azp = None if symmetric else torch.empty_like(input_scales,
                                                        dtype=torch.int32)
    torch.ops._C.dynamic_scaled_int8_quant(output, input.contiguous(),
                                           input_scales, input_azp)
    return output, input_scales, input_azp

def as_float32_tensor(x: Union[float, torch.tensor]) -> torch.tensor:
    return torch.as_tensor(x, dtype=torch.float32, device='cuda')


def ref_rms_norm(rms_norm_layer: RMSNorm,
                 x: torch.Tensor,
                 residual: Optional[torch.Tensor]) \
        -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if residual is not None:
        residual = residual.clone()
        out, residual = rms_norm_layer.forward_native(x, residual)
    else:
        out = rms_norm_layer.forward_native(x)

    return out, residual


def ref_dynamic_per_token_quant(rms_norm_layer: RMSNorm,
                                x: torch.Tensor,
                                quant_dtype: torch.dtype,
                                residual: Optional[torch.Tensor],
                                scale_ub: Optional[torch.Tensor]) \
        -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if scale_ub is not None:
        assert quant_dtype == torch.float8_e4m3fn

    # Norm
    torch_out, residual = ref_rms_norm(rms_norm_layer, x, residual)



    assert quant_dtype == torch.int8
    torch_out, scales = scaled_int8_quant(torch_out)

    return torch_out, scales, residual


def ref_impl(rms_norm_layer: RMSNorm,
             x: torch.Tensor,
             quant_dtype: torch.dtype,
             residual: Optional[torch.Tensor],
             scale_ub: Optional[torch.Tensor]) \
        -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    return ref_dynamic_per_token_quant(rms_norm_layer, x, quant_dtype,
                                       residual, scale_ub)


def ops_dynamic_per_token_quant(weight: torch.Tensor,
                                x: torch.Tensor,
                                quant_dtype: torch.dtype,
                                residual: Optional[torch.Tensor],
                                scale_ub: Optional[torch.Tensor]) \
        -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if residual is not None:
        residual = residual.clone()

            # def rms_norm_dynamic_per_token_quant_gpu(x, weight, eps, quant_dtype, scale_ub, residual):
            # """Wrapper for the actual CUDA implementation."""
        
        output = torch.empty_like(x, dtype=quant_dtype)
        scales = torch.empty((x.numel() // x.shape[-1], 1),
                            device=x.device,
                            dtype=torch.float32)

        out, scales =  torch.ops._C.rms_norm_dynamic_per_token_quant(output, x, weight,
                                                    scales, EPS, scale_ub,
                                                    residual)
    # out, scales = ops.rms_norm_dynamic_per_token_quant(x, weight, EPS,
    #                                                    quant_dtype, scale_ub,
    #                                                    residual)
    return out, scales, residual


def ops_impl(weight: torch.Tensor,
             x: torch.Tensor,
             quant_dtype: torch.dtype,
             residual: Optional[torch.Tensor],
             scale_ub: Optional[torch.Tensor]) \
        -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    return ops_dynamic_per_token_quant(weight, x, quant_dtype, residual,
                                       scale_ub)


@pytest.mark.parametrize("num_tokens, hidden_size", NUM_TOKENS_HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("scale_ub", SCALE_UBS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_rms_norm(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    scale_ub: bool,
    dtype: torch.dtype,
    quant_dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)

    if scale_ub is not None and quant_dtype != torch.float8_e4m3fn:
        # skip
        return

    layer = RMSNorm(hidden_size, EPS).to(dtype=dtype)

    # Make weights
    layer.weight.data.normal_(mean=1.0, std=0.1)

    # Make inputs
    scale = 1 / (hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
    residual = torch.randn_like(x) * scale if add_residual else None
    if scale_ub is not None:
        rms_x, _ = ref_rms_norm(layer, x, residual)
        scale_ub = torch.mean(rms_x).to(dtype=torch.float32, device='cuda')

    ref_out, ref_scales, ref_residual = \
        ref_impl(layer, x, quant_dtype, residual, scale_ub)
    ops_out, ops_scales, ops_residual = \
        ops_impl(layer.weight, x, quant_dtype, residual, scale_ub)

    assert ref_out.dtype == quant_dtype
    assert ops_out.dtype == quant_dtype
    assert torch.allclose(ref_scales, ops_scales)
    if quant_dtype == torch.int8:
        # big atol to account for round-off errors.
        assert torch.allclose(ref_out, ops_out, atol=1)
    else:
        assert torch.allclose(ref_out.to(dtype=torch.float32),
                              ops_out.to(dtype=torch.float32))
    if add_residual:
        assert torch.allclose(ref_residual, ops_residual)

    output = torch.empty_like(x, dtype=quant_dtype)
    scales = torch.empty((x.numel() // x.shape[-1], 1),
                         device=x.device,
                         dtype=torch.float32)
