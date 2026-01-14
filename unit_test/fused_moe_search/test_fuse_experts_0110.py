import copy
import argparse
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple, TypedDict
import os
# import ray
import torch
import torch.nn.functional as F
import triton
# from ray.experimental.tqdm_ray import tqdm
from transformers import AutoConfig
import vllm_metax.patch
from vllm_metax.model_executor.layers.fused_moe.fused_moe import *
from mcoplib import triton_fused_moe
# from vllm.model_executor.layers.fused_moe.fused_moe import *
# from fused_moe_1014 import *
# from vllm.platforms import current_platform
# from vllm.utils import FlexibleArgumentParser
import argparse
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
    num_iters: int = 100,) -> float:

    if dtype == torch.bfloat16:
        init_dtype = torch.bfloat16
    elif use_fp8_w8a8:
        init_dtype = use_fp8_w8a8
    else:
        init_dtype = torch.float16

    if dtype == torch.bfloat16:
        x = torch.randn(num_tokens, hidden_size, dtype = torch.bfloat16)
    else:
        x = torch.randn(num_tokens, hidden_size, dtype = torch.float16)


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
    

   
    compute_type = (tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16)

    def prepare(i: int):
        input_gating.copy_(gating_output[i])
        gating_output_softmax = F.softmax(input_gating, dim=-1)
        # vllm 0.8.5
        # topk_weights, topk_ids = fused_topk(x, gating_output_softmax, topk, True)
        # vllm 0.9.1
        topk_weights, topk_ids, token_expert_indices = fused_topk(x, gating_output_softmax, topk, True)
        return topk_weights,topk_ids

    def run(topk_weights, topk_ids):
        apply_router_weight_on_input = False
        fused_experts_impl(x, w1, w2, topk_weights, topk_ids, inplace=True, 
                    use_int8_w8a8=use_int8_w8a8,
                    use_int4_w4a16=use_int4_w4a16,
                    per_channel_quant=per_channel_quant,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    w1_zp=w1_zp,
                    w2_zp=w2_zp,
                    a1_scale=a1_scale,
                    a2_scale=a2_scale,
                    block_shape=block_shape)
        
    # JIT compilation & warmup
    topk_weights,topk_ids = prepare(0)
    run(topk_weights,topk_ids)
    torch.cuda.synchronize()

    # Warmup
    topk_weights,topk_ids = prepare(0)
    for _ in range(30):
        # graph.replay()
        run(topk_weights,topk_ids)
    torch.cuda.synchronize()

    # 实测 100次取平均
    elapsed_time_torch = 0

    stage1_time_list = []
    stage2_time_list = []
    topk_time = 0
    align_time = 0
    dy_scale_time = 0
    other_time = 0 
    # 修改为torch.profiler
    for i in range(num_iters):
        count = 0
        topk_weights,topk_ids = prepare(i)
        torch.cuda.synchronize()
        stage1_time = 0.0
        stage2_time = 0.0
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            # graph.replay()
            run(topk_weights,topk_ids)
        for evt in prof.events():
            # print(f"{evt=}")
            if evt.device_time_total == 0:
                continue
            if "fused_moe_kernel" in evt.name or "fusedMoe" in evt.name:  
                if count == 0:
                    stage1_time += evt.device_time_total
                    count = 1
                else:
                    stage2_time += evt.device_time_total
                    count = 0
            elif "TopK" in evt.name or "topk" in evt.name or "triton_" in evt.name:
                topk_time += evt.device_time_total
            elif "moe_align_block_size_kernel" in evt.name:
                align_time += evt.device_time_total
            elif "dynamic_scaled_int8_quant" in evt.name:
                dy_scale_time += evt.device_time_total
            else:
                other_time += evt.device_time_total
        if i == num_iters-1:
            print(f"dtype:{args.dtype},tp_size:{args.tp_size},batch_size:{num_tokens}")
            table = prof.key_averages().table(sort_by="device_time_total", row_limit=10)
            print(table)
        stage1_time_list.append(stage1_time)
        stage2_time_list.append(stage2_time)

    stage1_time = np.mean(stage1_time_list)
    stage2_time = np.mean(stage2_time_list)
    stage1_std = np.std(stage1_time_list, ddof=1)
    stage2_std = np.std(stage2_time_list, ddof=1)
    topk_time = topk_time / (num_iters)
    align_time = align_time / (num_iters)
    dy_scale_time = dy_scale_time / (num_iters)
    other_time = other_time / (num_iters)
    import statistics
    stage1_min = min(stage1_time_list)
    stage1_max = max(stage1_time_list)
    stage1_med = statistics.median(stage1_time_list)
    stage2_min = min(stage2_time_list)
    stage2_max = max(stage2_time_list)
    stage2_med = statistics.median(stage2_time_list)
    # print(f"stage1_time: {stage1_time} us[var:{stage1_var}],stage2_time:{stage2_time} us[var:{stage2_var}],add time:{stage1_time+stage2_time}us")
    print(f"stage1_time: {stage1_time} us[var:{stage1_std}, min:{stage1_min}, max:{stage1_max}, med:{stage1_med}]")
    print(f"stage2_time:{stage2_time} us[var:{stage2_std}, min:{stage2_min}, max:{stage2_max}, med:{stage2_med}]")
    print(f"topk_time: {topk_time} us")
    top_k_num = topk_ids.shape[1]
    M = num_tokens
    N, K1 = w1.shape[1], w1.shape[2]
    K2 = w2.shape[2]
    
    if use_int8_w8a8:
        total_flops = top_k_num * (2 * M * N * K1 + 2 * M * K1 * K2) / 2
    else:
        total_flops = top_k_num * (2 * M * N * K1 + 2 * M * K1 * K2)
    tflops = total_flops / ((stage1_time + stage2_time) * 1e6)
    
    return topk_time, stage1_time, stage2_time, align_time, dy_scale_time, other_time, tflops



def main(args: argparse.Namespace):
    print(args)
    torch.set_default_device("cuda")
    set_seed(args.seed)
    
    if "step" in args.model:
        from vllm.transformers_utils.configs.mmgpt import Step3vConfig
        config = Step3vConfig(hidden_size=7168, intermediate_size=18432, \
            num_attention_heads=64, num_attention_groups=1, \
            num_hidden_layers=61, max_seq_len=65536, vocab_size=128815,  \
            moe_every_n_layer=1, use_moe=True, moe_intermediate_size=5120,  \
            moe_num_experts=48, moe_top_k=3, head_dim=256, max_position_embedding=65536,\
            share_expert_dim=5120, share_q_dim=2048, norm_expert_weight=False,  \
            moe_layers_enum="4,5,6,7,8,9,10,11,12,13,14,15,16,17,185,46,47,48,49,50,51,52,53,54,55,56,57,58,59", \
            use_im_start_end=True,  vision_select_layer=-1, image_token_len=169).text_config
    else:
        config = AutoConfig.from_pretrained(args.model,trust_remote_code=True)

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
    elif config.architectures[0] in ["Qwen2MoeForCausalLM", "Qwen3MoeForCausalLM", "Qwen3NextForCausalLM"]:
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
    elif config.architectures[0] == "Glm4MoeForCausalLM":
        E = config.n_routed_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
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

    # E = 128
    # topk = 8
    print(f"E : {E}")
    print(f"topk : {topk}")
    print(f"shard_intermediate_size : {shard_intermediate_size}")
    print(f"hidden_size : {hidden_size}")
    print(f"data_type : {args.dtype}")
    print(f"use_int8_w8a8 : {use_int8_w8a8}")
    print(f"use_int4_w4a16 : {use_int4_w4a16}")
    if args.batch_size is None:
        batch_sizes = [1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64,
             96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]
        # batch_sizes = [1536, 2048, 3072, 4096]
        # batch_sizes = range(1, 8192)
        # batch_sizes = [i for i in range(1,41)] + [44, 48, 56, 64, 96, 128, 256, 512, 1024, 1536] + \
        #             [i for i in range(2048, 4097, 1024)]
    else:
        batch_sizes = [args.batch_size]
    
    for batch_size in batch_sizes:
        print(f"--------------batch_size : {batch_size}---------------")
        # try:
        start = time.time()
        topk_time, stage1_time, stage2_time, align_time, dy_scale_time, other_time, tflops = benchmark_config(None,
                            batch_size,
                            E,
                            shard_intermediate_size,
                            hidden_size,
                            topk,
                            dtype,
                            use_fp8_w8a8=use_fp8_w8a8,
                            use_int8_w8a16=use_int8_w8a16,
                            use_int8_w8a8=use_int8_w8a8,
                            use_int4_w4a16=use_int4_w4a16,
                            num_iters=100)
        end = time.time()
        print(f"Tuning took {end - start:.2f} seconds")
        # except :
        #     # Some configurations may be invalid and fail to compile.
        #     print(f"batch_size {batch_size} fail to run.")
        #     topk_time, stage1_time, stage2_time, align_time, dy_scale_time = -1, -1, -1, -1, -1
        
        save_csv = f"{savepath}.csv"
        print(f"result save in {save_csv}")
        with open(save_csv, "a+", newline='') as csvfile:
            fieldnames = ['model', 'dtype','batch_size', 'tp_size', 'topk_time', 
                    'stage1_time', 'stage2_time', 'moe_time', 'align_time', 
                    'dy_scale_time', 'other_time', 'tflops', 'total_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # 检查文件是否为空
            if os.path.getsize(save_csv) == 0:
                # Write the header row if the file is empty
                writer.writeheader()

            writer.writerow({
                'model': model_name,
                'dtype': args.dtype,
                'batch_size': batch_size,
                'tp_size': args.tp_size,
                'topk_time': topk_time,
                'stage1_time': stage1_time,
                'stage2_time': stage2_time,
                'moe_time': stage1_time + stage2_time,
                'align_time': align_time,
                'dy_scale_time': dy_scale_time,
                'other_time': other_time,
                'tflops': tflops,
                'total_time': topk_time + stage1_time + stage2_time + align_time + dy_scale_time + other_time
            })

if __name__ == "__main__":
    # parser = FlexibleArgumentParser()
    parser = argparse.ArgumentParser()
    # /pde_ai/models/llm/DeepSeek/DeepSeek-V2-236B/
    # /pde_ai/models/llm/Mistral/Mixtral-8x7B-Instruct-v0.1/
    # /pde_ai/models/llm/DeepSeek/DeepSeek-V3-BF16/
    parser.add_argument("--model",
                        type=str, required=True,
                        default="/pde_ai/models/llm/DeepSeek/DeepSeek-V3-BF16/")
    # parser.add_argument("--tp-size", "-tp", type=int, default=4)
    parser.add_argument("--tp-size", "-tp", type=int, required=True)
    parser.add_argument("--dtype",
                        type=str, required=True,
                        choices=["auto", "fp8_w8a8", "int8_w8a16", "int8_w8a8", "bfloat16", "int4_w4a16"],
                        default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=False)
    args = parser.parse_args()
    
    global savepath, sizeofdata, per_channel_quant, model_name
    model_name= args.model.split("/")[-2]
    if args.dtype == "bfloat16":
        savepath = f"./{model_name}_tp{args.tp_size}_bf16"
        sizeofdata = 2
        per_channel_quant = False
    elif args.dtype == "int4_w4a16":
        savepath = f"./{model_name}_tp{args.tp_size}_w4a16"
        sizeofdata = 0.5
        per_channel_quant = False
    elif args.dtype == "int8_w8a8":
        moe_option = os.environ.get('MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE')
        if moe_option is not None and moe_option == '1':
            os.environ["MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE"] = '0'
        int8_env1 = os.environ.get('TRITON_ENABLE_MACA_COMPILER_INT8_OPT')
        int8_env2 = os.environ.get('TRITON_ENABLE_ELEMENTWISE_PK_FMA_OPT')
        if int8_env1 is None or int8_env2 is None:
            os.environ["TRITON_ENABLE_MACA_COMPILER_INT8_OPT"] = '1'
            os.environ["TRITON_ENABLE_ELEMENTWISE_PK_FMA_OPT"] = '1'
            int8_env1 = os.environ.get('TRITON_ENABLE_MACA_COMPILER_INT8_OPT')
            int8_env2 = os.environ.get('TRITON_ENABLE_ELEMENTWISE_PK_FMA_OPT')
            print(f"TRITON_ENABLE_MACA_COMPILER_INT8_OPT:{int8_env1}")
            print(f"TRITON_ENABLE_ELEMENTWISE_PK_FMA_OPT:{int8_env2}")
            print("[WARNING] Check and clear the Triton Cache directory!!!!!!!!")
        savepath = f"./{model_name}_tp{args.tp_size}_w8a8"
        sizeofdata = 1
        per_channel_quant = True

    else:
        print("this model search dtype parameter is error.")
        exit()
    
    # os.makedirs(savepath, exist_ok=True)
    main(args)
