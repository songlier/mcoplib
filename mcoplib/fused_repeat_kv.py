from typing import Tuple, Union
import torch
import mcoplib.op  as api


class FusedRepeatKV(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, q_num_head, kv_num_head, head_dim):
        output = api.fused_repeat_kv_fwd(input, q_num_head, kv_num_head, head_dim)
        ctx.q_num_head = q_num_head
        ctx.kv_num_head = kv_num_head
        ctx.head_dim = head_dim
        ctx.partition = input.shape[2]
        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> torch.Tensor:
        
        # clark_note: This might cause wasted mem
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        
        q_num_head = ctx.q_num_head
        kv_num_head = ctx.kv_num_head
        partition = ctx.partition
        grad = api.fused_repeat_kv_bwd(grad_output, q_num_head, kv_num_head, partition)
        return grad, None, None, None
