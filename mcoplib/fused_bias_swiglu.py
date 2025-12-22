from typing import Tuple, Union
import torch
import mcoplib.op  as api

class FusedBiasSwiglu(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor
    ) -> torch.Tensor:

        output = api.fused_bias_swiglu_fwd(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> torch.Tensor:

        (input,) = ctx.saved_tensors

        grad_input = api.fused_bias_swiglu_bwd(input, grad_output)
        return grad_input


def fused_apply_bias_swiglu(
    t: torch.Tensor
) -> torch.Tensor:
    return FusedBiasSwiglu.apply(t)
