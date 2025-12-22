from typing import Tuple, Union
import torch
import mcoplib.op  as api

class FusedBiasGelu(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:

        output = api.fused_gelu_fwd(input,bias)
        ctx.save_for_backward(input,bias)
        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> torch.Tensor:

        (input, bias) = ctx.saved_tensors

        grad_input = api.fused_gelu_bwd(input, grad_output, bias)
        return grad_input


def fused_apply_bias_gelu(
    t: torch.Tensor
) -> torch.Tensor:
    return FusedBiasGelu.apply(t)
