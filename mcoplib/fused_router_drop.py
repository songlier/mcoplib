import torch
import triton
import triton.language as tl
from triton.runtime import driver
import triton.language.core as core
from triton.language.standard import _log2, sum, zeros_like
from copy import deepcopy

device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
fwd_kernels_branch1_step1 = {}
bwd_kernels_branch1_step1 = {}

@triton.jit
def compare_and_swap(x, ids, flip, i: core.constexpr, n_dims: core.constexpr):
    n_outer: core.constexpr = x.numel >> n_dims
    shape: core.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    y = core.reshape(x, shape)
    mask = core.arange(0, 2)[None, :, None]
    left = core.broadcast_to(sum(y * (1 - mask), 1)[:, None, :], shape)
    right = core.broadcast_to(sum(y * mask, 1)[:, None, :], shape)
    left = core.reshape(left, x.shape)
    right = core.reshape(right, x.shape)

    # idx
    y_idx = core.reshape(ids, shape)
    left_idx = core.broadcast_to(sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = core.broadcast_to(sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = core.reshape(left_idx, x.shape)
    right_idx = core.reshape(right_idx, x.shape)

    # actual compare-and-swap
    idtype = core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth,
                                signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = ((left > right) or ((left == right) and (left_idx < right_idx))) ^ flip
    ret = ix ^ core.where(cond, ileft ^ iright, zeros_like(ix))
    new_ids = ids ^ core.where(cond, left_idx ^ right_idx, zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids

@triton.jit
def _bitonic_merge(x, ids, stage: core.constexpr, order: core.constexpr,
                   n_dims: core.constexpr):
    n_outer: core.constexpr = x.numel >> n_dims
    core.static_assert(stage <= n_dims)
    if order == 2:
        shape: core.constexpr = [
            n_outer * 2**(n_dims - 1 - stage), 2, 2**stage
        ]
        flip = core.reshape(
            core.broadcast_to(core.arange(0, 2)[None, :, None], shape),
            x.shape)
    else:
        flip = order
    for i in core.static_range(stage):
        x, ids = compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids

@triton.jit
def argsort(x, ids, dim: core.constexpr = None, descending: core.constexpr = True):
    _dim: core.constexpr = len(x.shape) - 1 if dim is None else dim
    core.static_assert(_dim == len(x.shape) - 1,
                       "only minor dimension is currently supported")
    n_dims: core.constexpr = _log2(x.shape[_dim])
    pid = tl.program_id(0)

    for i in core.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending, n_dims)
    return x, ids

@triton.jit
def fwd_kernel_branch1_step1(input_ptr, output_ptr, indices_ptr, tokens_per_expert_ptr, n_rows, topk, N_COLS:tl.constexpr,
                   NUM_PROGRAMS:tl.constexpr, NUM_ROWS_PER_BLOCK:tl.constexpr,
                   BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, NUM_PROGRAMS, row_step, num_stages=num_stages):
        #load data:
        row_start_ptr = input_ptr + row_idx * BLOCK_SIZE
        col_offsets = tl.arange(0, BLOCK_SIZE)
        col_offsets_cols = tl.arange(0, N_COLS)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < BLOCK_SIZE
        row = tl.load(input_ptrs, mask=mask, other=-float('inf')).reshape(NUM_ROWS_PER_BLOCK, N_COLS)

        #topk:
        ids = tl.broadcast_to(tl.arange(0, N_COLS)[None, :], (NUM_ROWS_PER_BLOCK, N_COLS))
        probs, ids = argsort(row, ids, 1, True)
        mask_ = col_offsets_cols < topk
        probs = tl.where(mask_, probs, -float('inf'))

        #softmax:
        row_ = probs - tl.max(probs, axis=1)[:, None]
        numerator = tl.exp(row_)
        denominator = tl.sum(numerator, axis=1)
        softmax_output = numerator / denominator[:,None]

        #histc:
        ids = tl.cast(ids, tl.int32)
        ids = tl.where(mask_, ids, -int('1'))
        test_tensor = tl.histogram(ids.reshape(BLOCK_SIZE), N_COLS)
        tl.atomic_add(tokens_per_expert_ptr + col_offsets_cols, test_tensor)

        # output:
        mask_ = mask_[None,:].broadcast_to(NUM_ROWS_PER_BLOCK, N_COLS).reshape(BLOCK_SIZE)
        output_start_ptr = output_ptr + row_idx * BLOCK_SIZE
        indices_start_ptr = indices_ptr + row_idx * BLOCK_SIZE
        output_ptrs = output_start_ptr + col_offsets
        indices_ptrs = indices_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output.reshape(BLOCK_SIZE),mask=mask_)
        tl.store(indices_ptrs, ids.reshape(BLOCK_SIZE),mask=mask_)


@triton.jit
def bwd_kernel_branch1_step1(loss_ptr, y_topk_softmax_ptr, indices_ptr, bwd_val_ptr, topk:tl.constexpr, N_COLS:tl.constexpr, NUM_PROGRAMS:tl.constexpr,
               NUM_ROWS_PER_BLOCK:tl.constexpr, BLOCK_SIZE:tl.constexpr, BLOCK_SIZE_FINAL:tl.constexpr,num_stages:tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    offset = tl.arange(0, BLOCK_SIZE)
    test = tl.load(y_topk_softmax_ptr + 3)

    for row_idx in tl.range(row_start, NUM_PROGRAMS, row_step, num_stages=num_stages):
        #load data
        loss_start_ptr = loss_ptr + row_idx * BLOCK_SIZE
        y_topk_softmax_start_ptr = y_topk_softmax_ptr + row_idx * BLOCK_SIZE
        indices_start_ptr = indices_ptr + row_idx * BLOCK_SIZE
        loss_ptrs = loss_start_ptr + offset
        y_topk_softmax_ptrs = y_topk_softmax_start_ptr + offset
        indices_ptrs = indices_start_ptr + offset

        loss_vec = tl.load(loss_ptrs).reshape(NUM_ROWS_PER_BLOCK, topk)
        # if row_idx == 0:
        #     tl.device_print('loss_vec', loss_vec)
        y_topk_softmax = tl.load(y_topk_softmax_ptrs).reshape(NUM_ROWS_PER_BLOCK, topk)
        # if row_idx == 0:
        #     tl.device_print('y_topk_softmax', y_topk_softmax)
        indices = tl.load(indices_ptrs).reshape(NUM_ROWS_PER_BLOCK, topk)

        #softmax bwd
        J1 = loss_vec[:,None,:]
        y_topk_softmax_expand_dim = y_topk_softmax[:,None,:]
        J2_bef = -y_topk_softmax_expand_dim.permute(0,2,1) * y_topk_softmax_expand_dim
        softmax_bwd = tl.sum(J1.permute(0, 2, 1) * J2_bef, axis=-2) + (J1 * y_topk_softmax_expand_dim).reshape(NUM_ROWS_PER_BLOCK, topk)
        #topk_bwd
        indices_ = indices + (tl.arange(0, NUM_ROWS_PER_BLOCK) * N_COLS)[:,None]

        #store data
        bwd_val_start_ptr = bwd_val_ptr + row_idx * BLOCK_SIZE_FINAL
        # tl.device_print('softmax_bwd',softmax_bwd)
        # tl.device_print('indices_', indices_)
        tl.store(bwd_val_start_ptr + indices_, softmax_bwd)


class FusedRouterDrop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, topk):
        n_rows, n_cols = logits.shape

        probs = torch.zeros_like(logits)[:,:topk]
        top_indices = torch.zeros_like(logits, dtype=torch.int64)[:,:topk]
        tokens_per_expert = torch.zeros(n_cols, device=logits.device, dtype=torch.int64)

        ctx.save_for_backward(logits, probs, top_indices)
        ctx.topk = topk
        ctx.n_cols = n_cols

        BLOCK_SIZE = 32

        kernel, num_programs, NUM_PROGRAMS = fwd_kernels_branch1_step1.get(BLOCK_SIZE, (None, 0, 1))
        if kernel is None:
            num_warps = 1
            assert(n_cols == triton.next_power_of_2(n_cols))
            NUM_ROWS_PER_BLOCK = BLOCK_SIZE // n_cols
            NUM_PROGRAMS = n_rows // NUM_ROWS_PER_BLOCK
            num_stages = 4 if SIZE_SMEM > 200000 else 2
            kernel = fwd_kernel_branch1_step1.warmup(logits, probs, top_indices, tokens_per_expert, n_rows, topk, N_COLS=n_cols,
                                        BLOCK_SIZE=BLOCK_SIZE, NUM_ROWS_PER_BLOCK=NUM_ROWS_PER_BLOCK,
                                        NUM_PROGRAMS=NUM_PROGRAMS, num_stages=1, num_warps=num_warps, grid=(1, ))
            kernel._init_handles()
            n_regs = kernel.n_regs
            if n_regs == 0:
                n_regs = 1
            size_smem = kernel.metadata.shared
            if size_smem == 0:
                size_smem = 1
            occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
            occupancy = min(occupancy, SIZE_SMEM // size_smem)
            num_programs = NUM_SM * occupancy
            fwd_kernels_branch1_step1[BLOCK_SIZE] = (kernel, num_programs, NUM_PROGRAMS)

        num_programs = min(num_programs, NUM_PROGRAMS)
        # kernel, num_programs, NUM_PROGRAMS = fwd_kernels_branch1_step1.get(BLOCK_SIZE, (None, 0, 1))
        kernel[(num_programs, 1, 1)](
            logits,
            probs,
            top_indices,
            tokens_per_expert,
            n_rows,
            topk
        )
        return probs, top_indices, tokens_per_expert

    @staticmethod
    def backward(ctx, grad_probs, grad_top_indices, grad_tokens_per_expert):
        grad_top_indices = None
        grad_tokens_per_expert = None
        
        BLOCK_SIZE = 4
        logits, probs, top_indices = ctx.saved_tensors
        topk = ctx.topk
        n_cols = ctx.n_cols
        
        probs = probs.contiguous()
        top_indices = top_indices.contiguous()

        n_rows = logits.shape[0]

        bwd_output = torch.zeros_like(logits)

        kernel, num_programs, NUM_PROGRAMS = bwd_kernels_branch1_step1.get(BLOCK_SIZE, (None, 0, 1))
        if kernel is None:
            num_warps = 1
            assert(n_cols == triton.next_power_of_2(n_cols))
            NUM_ROWS_PER_BLOCK = BLOCK_SIZE // topk
            NUM_PROGRAMS = n_rows // NUM_ROWS_PER_BLOCK
            BLOCK_SIZE_FINAL = NUM_ROWS_PER_BLOCK * n_cols
            num_stages = 4 if SIZE_SMEM > 200000 else 2
            kernel = bwd_kernel_branch1_step1.warmup(grad_probs, probs, top_indices, bwd_output, topk=topk, N_COLS=n_cols,
                                        NUM_PROGRAMS=NUM_PROGRAMS, NUM_ROWS_PER_BLOCK=NUM_ROWS_PER_BLOCK,
                                        BLOCK_SIZE=BLOCK_SIZE, BLOCK_SIZE_FINAL=BLOCK_SIZE_FINAL, num_stages=num_stages,
                                        num_warps=num_warps, grid=(1, ))
            kernel._init_handles()
            n_regs = kernel.n_regs
            if n_regs == 0:
                n_regs = 1
            size_smem = kernel.metadata.shared
            if size_smem == 0:
                size_smem = 1
            occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
            occupancy = min(occupancy, SIZE_SMEM // size_smem)
            num_programs = NUM_SM * occupancy
            bwd_kernels_branch1_step1[BLOCK_SIZE] = (kernel, num_programs, NUM_PROGRAMS)

        num_programs = min(num_programs, NUM_PROGRAMS)
        # kernel, num_programs, NUM_PROGRAMS = bwd_kernels_branch1_step1.get(BLOCK_SIZE, (None, 0, 1))
        kernel[(num_programs,1,1)](
            grad_probs,
            probs,
            top_indices,
            bwd_output
        )
        return bwd_output, None

def branch1_forward(logits, topk):
    return FusedRouterDrop.apply(logits, topk)

def torch_fun(logits, N):
    y, idx = torch.topk(logits, k=2, dim=1)
    y_t = torch.softmax(y,axis=1)
    tokens_per_expert = torch.bincount(idx.view(-1),minlength=16)
    return y_t, idx, tokens_per_expert

def mean_relative_error(y_true, y_pred, epsilon=1e-8):
    """
    计算两个张量的平均相对误差 (Mean Relative Error, MRE)

    参数:
        y_true (torch.Tensor): 真实值张量。
        y_pred (torch.Tensor): 预测值张量。
        epsilon (float): 一个小常数，用于避免除以零的情况。

    返回:
        torch.Tensor: 平均相对误差。
    """
    # 确保输入张量的形状一致
    if y_true.shape != y_pred.shape:
        raise ValueError("输入张量的形状必须一致")

    # 计算相对误差
    relative_error = torch.abs(y_true - y_pred) / (torch.abs(y_true) + epsilon)

    # 计算平均相对误差
    mre = torch.mean(relative_error)

    return mre
