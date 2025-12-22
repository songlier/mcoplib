from typing import Tuple, Union
import torch
import mcoplib.op as api

class FusedBiasDropout(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, residual, dropout_prob = 0.0):
        output = api.fused_bias_dropout(input, residual, dropout_prob)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output, None