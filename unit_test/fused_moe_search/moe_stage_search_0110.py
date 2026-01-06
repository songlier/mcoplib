import copy
import argparse
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple, TypedDict
import os
import ray
import torch
import torch.nn.functional as F
import triton
from ray.experimental.tqdm_ray import tqdm
from transformers import AutoConfig
from datetime import datetime
import vllm_metax.patch
from vllm_metax.model_executor.layers.fused_moe.fused_moe import *
from vllm_metax.model_executor.layers.fused_moe.fused_moe import _get_config_quant_dtype
# from vllm_metax.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size)
import argparse
# from vllm_metax.platforms import current_platform
# from vllm_metax.utils import FlexibleArgumentParser
import csv
import triton.language as tl
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)             # 设置CPU生成随机数的种子
    torch.cuda.manual_seed(seed)        # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)    # 如果有多个GPU，为所有的GPU设置随机种子
    np.random.seed(seed)                # 设置numpy生成随机数的种子
    random.seed(seed)                   # 设置Python生成随机数的种子
    torch.backends.cudnn.deterministic = True  # 保证每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False     # 如果确定输入数据的大小或每次的输入数据变化不大，设置为False

import re
def convert_time_to_us(time_str):
    pattern = r'^(\d+\.?\d*)\s*([smu]?s)$'  # 使用^表示字符串开头
    match = re.match(pattern, time_str)   # 使用match函数代替search
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        if unit == 's':
            return value * 1000000
        elif unit == 'ms':
            return value * 1000
        elif unit == 'us':
            return value
    return None

def extract_cuda_total_time(data):
    """Extract and sum CUDA total time for lines containing 'fused_moe_kernel'."""
    lines = data.strip().split('\n')
    total_time_us = 0.0

    for line in lines:
        if 'fused_moe_kernel' in line or 'fusedMoe' in line:
            # Split the line by spaces and filter out empty strings
            parts = [part.strip() for part in line.split() if part.strip()]
            # The 'CUDA total' column is the 8th column (index 7)
            cuda_total_str = parts[8]
            cuda_total_us = convert_time_to_us(cuda_total_str)
            total_time_us += cuda_total_us

    return total_time_us

class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int8_w8a8: bool,
    use_int4_w4a16: bool,
    num_iters: int = 100,
    tag: str = "stage1",) -> float:

    if dtype == torch.bfloat16:
        init_dtype = torch.bfloat16
    elif use_fp8_w8a8:
        init_dtype = use_fp8_w8a8
    else:
        init_dtype = torch.float16
    if dtype == torch.bfloat16:
        x = torch.randn(num_tokens, hidden_size, dtype= torch.bfloat16)
    else:
        x = torch.randn(num_tokens, hidden_size, dtype= torch.float16)


    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None
    w1_zp = None
    w2_zp = None
    block_shape = None

    if use_int8_w8a16:
        w1 = torch.randint(-127,
                           127, (
                               num_experts,
                               shard_intermediate_size,
                               hidden_size,
                           ),
                           dtype=torch.int8)
        w2 = torch.randint(-127,
                           127, (
                               num_experts,
                               hidden_size,
                               shard_intermediate_size // 2,
                           ),
                           dtype=torch.int8)
        w1_scale = torch.randn((num_experts, 2 * shard_intermediate_size),
                               dtype=torch.float32)
        w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32)

    elif use_fp8_w8a8:
        w1 = torch.randn(num_experts,
                         shard_intermediate_size,
                         hidden_size,
                         dtype=init_dtype)
        w2 = torch.randn(num_experts,
                         hidden_size,
                         shard_intermediate_size // 2,
                         dtype=init_dtype)
        w1_scale = torch.randn(num_experts, dtype=torch.float32)
        w2_scale = torch.randn(num_experts, dtype=torch.float32)
        a1_scale = torch.randn(1, dtype=torch.float32)
        a2_scale = torch.randn(1, dtype=torch.float32)
        w1 = w1.to(torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fn)
        
    elif use_int8_w8a8:
        w1 = torch.randint(-127,
                           127, (
                               num_experts,
                               shard_intermediate_size,
                               hidden_size,
                           ),
                           dtype=torch.int8)
        w2 = torch.randint(-127,
                           127, (
                               num_experts,
                               hidden_size,
                               shard_intermediate_size // 2,
                           ),
                           dtype=torch.int8)
        w1_scale = torch.randn((num_experts, shard_intermediate_size, 1),dtype=torch.float32)
        w2_scale = torch.randn((num_experts, hidden_size,  1), dtype=torch.float32)
    elif use_int4_w4a16:
        w1 = torch.randint(0,
                           256, (
                               num_experts,
                               shard_intermediate_size,
                               hidden_size // 2,
                           ),
                           dtype=torch.uint8)
        w2 = torch.randint(0,
                           256, (
                               num_experts,
                               hidden_size,
                               shard_intermediate_size // 4,
                           ),
                           dtype=torch.uint8)
        w1_scale = torch.randn((num_experts, shard_intermediate_size, 16), dtype= torch.float16)
        w1_zp = torch.randint(0, 256, (num_experts, shard_intermediate_size // 2, 16), dtype=torch.uint8)
            
        w2_scale = torch.randn((num_experts, hidden_size, 6), dtype= torch.float16)
        w2_zp = torch.randint(0, 256, (num_experts, hidden_size // 2, 6), dtype=torch.uint8)
        block_shape = [0, 128]

    else:
        w1 = torch.randn(num_experts,
                         shard_intermediate_size,
                         hidden_size,
                         dtype=init_dtype)
        w2 = torch.randn(num_experts,
                         hidden_size,
                         shard_intermediate_size // 2,
                         dtype=init_dtype)
        # raise ValueError(f"Not supported yet")
    if use_int4_w4a16:
        gating_output = torch.randn(num_iters,
                                    num_tokens,
                                    num_experts,
                                    dtype=torch.float16)

            
        input_gating = torch.empty(num_tokens, num_experts, dtype=torch.float16)
    else:
        gating_output = torch.randn(num_iters,
                                    num_tokens,
                                    num_experts,
                                    dtype=torch.float32)

            
        input_gating = torch.empty(num_tokens, num_experts, dtype=torch.float32)
    # topk_weights = None
    # topk_ids = None
    # sorted_token_ids = None
    # expert_ids = None
    # num_tokens_post_padded = None
    # compute_type = (tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16)
    if x.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif x.dtype == torch.float16:
        compute_type = tl.float16
    elif x.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {x.dtype}")

    # print(f"-----{dtype=}")
    if tag == 'stage1': 
        intermediate_cache1 = torch.randn((num_tokens, topk, shard_intermediate_size), device="cuda", dtype=dtype)
    else:
        intermediate_cache2 = torch.randn((num_tokens * topk, shard_intermediate_size//2), device="cuda", dtype=dtype)
        intermediate_cache3 = torch.randn((num_tokens, topk, w2.shape[1]), device="cuda", dtype=dtype)

    def prepare(i: int):
        
        input_gating.copy_(gating_output[i])
        gating_output_softmax = F.softmax(input_gating, dim=-1)
        # print(f"{gating_output_softmax.shape=}, {gating_output_softmax.dtype=}")
        # global topk_weights,topk_ids,sorted_token_ids,expert_ids,num_tokens_post_padded
        topk_weights, topk_ids, token_expert_indices = fused_topk(x, gating_output_softmax, topk, False)
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size(topk_ids, config['BLOCK_SIZE_M'], num_experts))

        return topk_weights,topk_ids,sorted_token_ids,expert_ids,num_tokens_post_padded

    def run(topk_weights,topk_ids,sorted_token_ids,expert_ids,num_tokens_post_padded,tag):
        # print(f"tag={tag}, invoke_fused_moe_kernel, per_channel_quant:{per_channel_quant}")
        qtype = _get_config_quant_dtype(use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a8=use_int8_w8a8,
                                # use_int8_w8a16=use_int8_w8a16,
                                # use_int4_w4a16=use_int4_w4a16,
                                use_mxfp4_w4a4=False,)
        if tag == 'stage1':
            qcurr_hidden_states, a1q_scale = moe_kernel_quantize_input(
                                A=x,
                                A_scale=a1_scale,
                                quant_dtype=qtype,
                                per_act_token_quant=per_channel_quant
                                ) 
            invoke_fused_moe_kernel(qcurr_hidden_states,
                                w1,
                                intermediate_cache1,
                                a1q_scale,
                                w1_scale,
                                w1_zp, 
                                topk_weights,
                                # topk_ids,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                False,
                                topk_ids.shape[1],
                                config,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a8=use_int8_w8a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                orig_acc_dtype=x.dtype,
                                per_channel_quant=per_channel_quant,
                                block_shape=block_shape
                                )
        else:
            qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
                            A=intermediate_cache2,
                            A_scale=a2_scale,
                            quant_dtype=qtype,
                            per_act_token_quant=per_channel_quant) 
            invoke_fused_moe_kernel(qintermediate_cache2,
                            w2,
                            intermediate_cache3,
                            a2q_scale,
                            w2_scale,
                            w2_zp, 
                            topk_weights,
                            # topk_ids,
                            sorted_token_ids,
                            expert_ids,
                            num_tokens_post_padded,
                            True,
                            1,
                            config,
                            compute_type=compute_type,
                            use_fp8_w8a8=use_fp8_w8a8,
                            use_int8_w8a8=use_int8_w8a8,
                            use_int8_w8a16=use_int8_w8a16,
                            use_int4_w4a16=use_int4_w4a16,
                            orig_acc_dtype=x.dtype,
                            per_channel_quant=per_channel_quant,
                            block_shape=block_shape
                            )


    # JIT compilation & warmup
    topk_weights,topk_ids,sorted_token_ids,expert_ids,num_tokens_post_padded = prepare(0)
    run(topk_weights,topk_ids,sorted_token_ids,expert_ids,num_tokens_post_padded,tag)
    torch.cuda.synchronize()

    # Warmup
    topk_weights,topk_ids,sorted_token_ids,expert_ids,num_tokens_post_padded = prepare(0)
    for _ in range(10):
        # graph.replay()
        run(topk_weights,topk_ids,sorted_token_ids,expert_ids,num_tokens_post_padded,tag)
    torch.cuda.synchronize()

    elapsed_time_torch = 0
    # 修改为torch.profiler
    for i in range(num_iters):
        topk_weights,topk_ids,sorted_token_ids,expert_ids,num_tokens_post_padded = prepare(i)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            # graph.replay()
            run(topk_weights,topk_ids,sorted_token_ids,expert_ids,num_tokens_post_padded,tag)
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        # print(convert_time_to_us(table))
        print(table)
        elapsed_time_torch += extract_cuda_total_time(table)
    torch_avg = elapsed_time_torch / (num_iters )
  
    return torch_avg


def get_configs_compute_bound(tag: str) -> List[Dict[str, int]]:
    # Reduced search space for faster tuning.
    # TODO(woosuk): Increase the search space and use a performance model to
    # prune the search space.
    configs: List[BenchmarkConfig] = []
    if tag == "stage1":
        num_stages_list = [2, 3, 4]
        splitk_list = [1, 2]
    else:
        num_stages_list = [1, 2, 3, 4]
        splitk_list = [1]
    for num_stages in num_stages_list:
       for block_m in [16, 32, 64, 128]:
           for block_k in [32, 64, 128, 256]:
               for block_n in [32, 64, 128, 256]:
                   for num_warps in [4, 8]:
                       for group_size in [1, 16, 32, 64]:
                            for pipline in ["basic", "cpasync"]:
                                for scenario in ["", "unroll", "storeCoalesce"]:
                                    for splitk in splitk_list:
                                        if scenario == "storeCoalesce":
                                            if not (block_m == 128 and block_n == 128 and (block_k == 256 or block_k == 128 or block_k == 32)
                                                or (block_m == 32 and block_n == 256 and block_k == 32)):
                                                continue
                                        if pipline == "basic" and scenario== "unroll":
                                            continue
                                        if block_m == block_n and block_m >= 128 and pipline == "cpasync":
                                            if (block_m+block_n)*block_k*sizeofdata>64*1024:    #sizeofdata  :  bf16=2, w8a8=1
                                                continue
                                        elif pipline == "cpasync":
                                            if (block_m+block_n)*block_k*num_stages*sizeofdata>64*1024:  #sizeofdata  :  bf16=2, w8a8=1
                                                continue
                                        if (block_m ==16 and block_k ==32) and num_warps == 8 and pipline == "cpasync":
                                            continue
                                        if (block_m == 128 and block_n == 128 and block_k == 256) and num_stages != 4:
                                            continue
                                        if (block_m == 128 and block_n == 256 and block_k == 128):
                                            continue
                                        if pipline == "basic":
                                            if (block_m == 128 and block_n == 128 and block_k == 128):
                                                continue
                                        if block_m == 128 and block_n == 128 and (block_k == 128 or block_k == 256):
                                            if pipline == "basic":
                                                continue
                                            if num_warps != 4:
                                                continue
                                            if num_stages != 4:
                                                continue
                                        configs.append({
                                            "BLOCK_SIZE_M": block_m,
                                            "BLOCK_SIZE_N": block_n,
                                            "BLOCK_SIZE_K": block_k,
                                            "GROUP_SIZE_M": group_size,
                                            "num_warps": num_warps,
                                            "num_stages": num_stages,
                                            "pipeline":pipline,
                                            "scenario": scenario,
                                            "ACCF32": False, 
                                            "SPLIT_K": splitk,
                                        })
    # configs.append({
    #     "BLOCK_SIZE_M": 128,
    #     "BLOCK_SIZE_N": 32,
    #     "BLOCK_SIZE_K": 32,
    #     "GROUP_SIZE_M": 1,
    #     "num_warps": 8,
    #     "num_stages": 4,
    #     "pipeline":"cpasync",
    #     "scenario": "unroll",
    #     "ACCF32": False, 
    #     "SPLIT_K": 1,
    # })
    return configs

def tune(
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int8_w8a8: bool,
    use_int4_w4a16: bool,
    search_space: List[Dict[str, int]],
    save_path:str,
    tag:str,
    debug:bool = False,
) -> Dict[str, int]:
    best_config = None
    best_time = float("inf")
    time_data = []

    # for config in tqdm(search_space):
    for config in search_space:
        if (num_tokens > 63 and config["SPLIT_K"] == 2) or (num_tokens < 16 and config["BLOCK_SIZE_M"] > 16) \
            or (num_tokens < 32 and config["BLOCK_SIZE_M"] > 32) or (num_tokens < 64 and config["BLOCK_SIZE_M"] > 64):
            print(f"skip {config=}")
            continue
        if (shard_intermediate_size >= 7168 and hidden_size >= 7168):
            if (config["BLOCK_SIZE_N"] < 64 or config["BLOCK_SIZE_K"] < 64):
                print(f"skip {config=}")
                continue
        print(f"test {config=}") 
        if debug:
            kernel_time = benchmark_config(config,
                                            num_tokens,
                                            num_experts,
                                            shard_intermediate_size,
                                            hidden_size,
                                            topk,
                                            dtype,
                                            use_fp8_w8a8,
                                            use_int8_w8a16,
                                            use_int8_w8a8,
                                            use_int4_w4a16,
                                            num_iters=10,
                                            tag=tag)
        else:
            try:
                kernel_time = benchmark_config(config,
                                                num_tokens,
                                                num_experts,
                                                shard_intermediate_size,
                                                hidden_size,
                                                topk,
                                                dtype,
                                                use_fp8_w8a8,
                                                use_int8_w8a16,
                                                use_int8_w8a8,
                                                use_int4_w4a16,
                                                num_iters=10,
                                                tag=tag)
            except :
                # Some configurations may be invalid and fail to compile.
                print("Some configurations may be invalid and fail to compile.")
                continue
        print(f"{config}, {kernel_time}")
        time_data.append(f"{config}, {kernel_time}")
        if kernel_time < best_time:
            best_time = kernel_time
            best_config = config
    now = datetime.now()
    print(f"{now.ctime()}] Completed tuning for batch_size={num_tokens}")
    save_txt = f"{save_path}/{tag}-{num_tokens}.txt"
    with open(save_txt, "w") as f:
        for item in time_data:
            f.write(item + "\n")

    # assert best_config is not None
    return best_config



def main(args: argparse.Namespace):
    print(args)
    torch.set_default_device("cuda")
    set_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model,trust_remote_code=True)
    # from vllm.transformers_utils.configs.mmgpt import Step3vConfig
    # config = Step3vConfig().text_config
    # config = Step3vConfig(hidden_size=7168, intermediate_size=18432, \
    #     num_attention_heads=64, num_attention_groups=1, \
    #     num_hidden_layers=61, max_seq_len=65536, vocab_size=128815,  \
    #     moe_every_n_layer=1, use_moe=True, moe_intermediate_size=5120,  \
    #     moe_num_experts=48, moe_top_k=3, head_dim=256, max_position_embedding=65536,\
    #     share_expert_dim=5120, share_q_dim=2048, norm_expert_weight=False,  \
    #     moe_layers_enum="4,5,6,7,8,9,10,11,12,13,14,15,16,17,185,46,47,48,49,50,51,52,53,54,55,56,57,58,59", \
    #     use_im_start_end=True,  vision_select_layer=-1, image_token_len=169).text_config


    if config.architectures[0] == "DbrxForCausalLM":
        E = config.ffn_config.moe_num_experts
        topk = config.ffn_config.moe_top_k
        intermediate_size = config.ffn_config.ffn_hidden_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] == "JambaForCausalLM":
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] in ["Qwen2MoeForCausalLM", "Qwen3MoeForCausalLM"]:
        E = config.num_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
        # hid_model = config.hidden_size
    elif config.architectures[0] == "DeepseekV2ForCausalLM":
        E = config.n_routed_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] == "DeepseekV3ForCausalLM":
        E = config.n_routed_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] == "BaiChuanMoEForCausalLM":
        # E = 8
        E = 16
        topk = 2
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] == "Glm4MoeForCausalLM":
        E = config.n_routed_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] == "Step2MiniForCausalLM":
        E = config.moe_num_experts
        topk = config.moe_top_k
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] == "Ernie4_5_MoeForCausalLM":
        E = config.moe_num_experts
        topk = config.moe_k
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    else:
        # Default: Mixtral.
        E = config.num_local_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    hidden_size = config.hidden_size
    dtype = config.torch_dtype


    use_fp8_w8a8 = args.dtype == "fp8_w8a8"
    use_int8_w8a16 = args.dtype == "int8_w8a16"
    use_int8_w8a8 = args.dtype == "int8_w8a8"
    use_int4_w4a16 = args.dtype == "int4_w4a16"

    print(f"E : {E}")
    print(f"topk : {topk}")
    print(f"shard_intermediate_size : {shard_intermediate_size}")
    print(f"hidden_size : {hidden_size}")
    print(f"data_type : {args.dtype}")
    print(f"use_int8_w8a8 : {use_int8_w8a8}")
    # if use_int8_w8a8:
    #     dtype = torch.int8
    # else:
    #     dtype = torch.bfloat16
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    # print(f"config.torch_dtype: {dtype}")
    if args.batch_size is None:
        batch_sizes = [
           1, 2, 4, 8, 16, 24, 32, 48, 64,
             96, 128, 256, 512, 1024, 1536,2048, 3072, 4096
        ]
        # 1 64 256 4096
    else:
        batch_sizes = [args.batch_size]
        # batch_sizes_str = args.batch_size.split("_")
        # batch_sizes = [int(x) for x in batch_sizes_str]
        # print(f"batch_sizes : {batch_sizes}")

    search_space = get_configs_compute_bound(args.tag)
    print(f"Start tuning over {len(search_space)} configurations...")
    # if args.dtype == "int8_w8a8" and args.tune:
    for batch_size in batch_sizes:
        start = time.time()
        tune(batch_size, E, shard_intermediate_size, hidden_size,
                    topk, dtype, use_fp8_w8a8, use_int8_w8a16, use_int8_w8a8, use_int4_w4a16, search_space,save_path=savepath,tag =args.tag, debug=args.debug)
        end = time.time()
        print(f"Tuning took {end - start:.2f} seconds")
    

if __name__ == "__main__":
    # parser = FlexibleArgumentParser()
    parser = argparse.ArgumentParser()
    # /pde_ai/models/llm/DeepSeek/DeepSeek-V2-236B/
    # /pde_ai/models/llm/Mistral/Mixtral-8x7B-Instruct-v0.1/
    # /pde_ai/models/llm/DeepSeek/DeepSeek-V3-BF16/
    # now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # print(f"this time: {now_time}")
    parser.add_argument("--model",
                        type=str,
                        default="/pde_ai/models/llm/DeepSeek/DeepSeek-V3-BF16/")
    # parser.add_argument("--tp-size", "-tp", type=int, default=4)
    parser.add_argument("--tp-size", "-tp", type=int, required=False)
    parser.add_argument("--dtype",
                        type=str,
                        choices=["auto", "fp8_w8a8", "int8_w8a16", "int8_w8a8", "bfloat16", "int4_w4a16"],
                        default="auto")
    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--batch-size", type=str,required=False)
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--tag",
                        type=str,
                        choices=["stage1", "stage2"],
                        default=None)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    name= args.model.split("/")[-2]
    global savepath, sizeofdata, per_channel_quant
    if args.tag is None:
        tags = ["stage1", "stage2"]
    else:
        tags = [args.tag]
    for tag in tags:
        args.tag = tag
        if args.dtype == "bfloat16":
            savepath = f"./tp{args.tp_size}/{args.tag}_bf16/{name}/"
            sizeofdata = 2
            per_channel_quant = False
        elif args.dtype == "int8_w8a8" and args.tune:
            savepath = f"./tp{args.tp_size}/{args.tag}_w8a8/{name}/"
            sizeofdata = 1
            per_channel_quant = True
        elif args.dtype == "int4_w4a16":
            savepath = f"./tp{args.tp_size}/{args.tag}_w4a16/{name}/"
            sizeofdata = 0.5
            per_channel_quant = False
        else:
            print("this model search dtype parameter is error.")
            exit()
    
        os.makedirs(savepath, exist_ok=True)
        main(args)
