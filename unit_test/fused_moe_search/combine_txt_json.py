import json
from collections import defaultdict
import ast
import re, csv
from vllm.model_executor.layers.fused_moe.fused_moe import *
from vllm.utils import FlexibleArgumentParser
from transformers import AutoConfig

# 定义读取文件的函数
def read_txt(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(r"(.+),\s*(\d+\.\d+)\n?$", line)
            if match:
                dict_str = match.group(1)
                time_str = match.group(2)
                config_dict = ast.literal_eval(dict_str)
                config_dict['time'] = float(time_str)
                data.append(config_dict)
    return data

# 定义找到每个 BLOCK_SIZE_M 的最优配置的函数
def find_best_config(data):
    best_configs = {}
    for record in data:
        block_size_m = record['BLOCK_SIZE_M']
        time = record['time']
        # 如果当前 BLOCK_SIZE_M 还没有记录，或者当前记录的耗时更短
        if (block_size_m not in best_configs or time < best_configs[block_size_m]['time']) and time > 0:
            best_configs[block_size_m] = record
    return best_configs

def get_config_dtype_str(
        dtype: torch.dtype,
        use_int4_w4a16: Optional[bool] = False,
        use_int8_w8a8: Optional[bool] = False,
        use_int8_w8a16: Optional[bool] = False,
        use_fp8_w8a8: Optional[bool] = False,
        use_mxfp4_w4a4: Optional[bool] = False) -> Optional[str]:
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_int8_w8a8:
        return "int8_w8a8"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif use_int4_w4a16:
        return "int4_w4a16"
    elif use_mxfp4_w4a4:
        return "mxfp4_w4a4"
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None

def get_config_file_name(E: int,
                         N: int,
                         dtype: Optional[str],
                         block_shape: Optional[list[int]] = None) -> str:
    #device_name = current_platform.get_device_name().replace(" ", "_")
    device_name = "Device_4000"
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_shape_selector = ("" if not block_shape or not all(block_shape) else
                            f",block_shape={block_shape}").replace(" ", "")
    return f"E={E},N={N},device_name={device_name}{dtype_selector}{block_shape_selector}.json"  # noqa: E501

def save_configs(w8a8) -> None:
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

    hidden_size = config.hidden_size
    dtype = config.torch_dtype
    # use_fp8_w8a8 = "fp8_w8a8"
    # use_int8_w8a16 = "int8_w8a16"
    
    if config.architectures[0] in ["Qwen2MoeForCausalLM", "Qwen3MoeForCausalLM"]:
        num_experts = config.num_experts
        # topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] in ["Step2MiniForCausalLM"]:
        num_experts = config.moe_num_experts
        # topk = config.moe_top_k
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] == "DeepseekV2ForCausalLM":
        num_experts = config.n_routed_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] == "DeepseekV3ForCausalLM":
        num_experts = config.n_routed_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] == "Ernie4_5_MoeForCausalLM":
        num_experts = config.moe_num_experts
        topk = config.moe_k
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    elif config.architectures[0] == "Glm4MoeForCausalLM":
        num_experts = config.n_routed_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
    else:
        # Default: Mixtral.
        num_experts = config.num_local_experts
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
        shard_intermediate_size = 2 * intermediate_size // args.tp_size
        print("this model name function is fail!")
        # return "None"
    # E = 128
    # topk = 4
    # intermediate_size = 2048
    # shard_intermediate_size = 2 * intermediate_size // args.tp_size
    # hidden_size = 7168
    # dtype = torch.bfloat16

    if w8a8 == 0:
        dtype_str = get_config_dtype_str(dtype)
    else:
        use_int8_w8a8 = "int8_w8a8"
        dtype_str = get_config_dtype_str(dtype,
                                        use_int8_w8a8=use_int8_w8a8)

    filename = f"H={hidden_size},"+get_config_file_name(num_experts, shard_intermediate_size // 2,
                                    dtype_str)

    print(f"Writing best config to {filename}...")
    return filename
    
# 定义主函数
def main(name):
    # 读取两个阶段的文件
    batch_sizes=[1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096] 
    # ,\
    #                 5120, 6144, 7168, 8192, 9216, 10240, 12288, 14336, 16384, 18432, 20480,\
    #                 24576, 28672, 32768, 36864, 40960, 49152, 57344, 65536]
    global_best_config = {}
    global_best_config_csv = {}
    if args.w8a8 == 0:
        savepath1=f"./tp{args.tp_size}/stage1_bf16/{name}"
        savepath2=f"./tp{args.tp_size}/stage2_bf16/{name}"
    else:
        savepath1=f"./tp{args.tp_size}/stage1_w8a8/{name}"
        savepath2=f"./tp{args.tp_size}/stage2_w8a8/{name}"

    # 找到全局最优的 BLOCK_SIZE_M
    for batchsize in batch_sizes:
        data1_path = f'{savepath1}/stage1-{batchsize}.txt'
        data2_path = f'{savepath2}/stage2-{batchsize}.txt'
        
        Numstage = 0
        best_stage = None
        global_best_time = float('inf')
        stage_name = None
        None_stage = None
        
        if os.path.exists(data1_path):
            stage1_data = read_txt(data1_path)
            # 找到每个阶段的最优配置
            best_stage1 = find_best_config(stage1_data)
            Numstage += 1
            if Numstage == 1:
                best_stage = best_stage1
                stage_name = "stage1"
                None_stage = "stage2"
        else:
            best_stage1 = None
        
        if os.path.exists(data2_path):
            stage2_data = read_txt(data2_path)
            # 找到每个阶段的最优配置
            best_stage2 = find_best_config(stage2_data)
            Numstage += 1
            if Numstage == 1:
                best_stage = best_stage2
                stage_name = "stage2"
                None_stage = "stage1"
        else:
            best_stage2 = None
        
        if Numstage == 2:
            for block_size_m in best_stage1:
                if block_size_m in best_stage2:
                    total_time = best_stage1[block_size_m]['time'] + best_stage2[block_size_m]['time']
                    if total_time < global_best_time and total_time > 0.0:
                        global_best_time = total_time
                        global_best_config[batchsize] = {
                            "stage1": {k: v for k, v in best_stage1[block_size_m].items() if k != 'time'},
                            "stage2": {k: v for k, v in best_stage2[block_size_m].items() if k != 'time'},
                        }
                        global_best_config_csv[batchsize] = {
                            "stage1": {k: v for k, v in best_stage1[block_size_m].items() if k != 'time'},
                            "stage2": {k: v for k, v in best_stage2[block_size_m].items() if k != 'time'},
                            "time" : {"stage1_time": best_stage1[block_size_m]['time'], 
                                    "stage2_time": best_stage2[block_size_m]['time'],
                                    "total_time" : total_time}
                        }
        elif Numstage == 1:
            print(f"{batchsize=} Only found {stage_name} data.")
            for block_size_m in best_stage:
                s_time = best_stage[block_size_m]['time']
                if s_time < global_best_time and s_time > 0.0:
                    global_best_time = s_time
                    global_best_config[batchsize] = {
                        stage_name: {k: v for k, v in best_stage[block_size_m].items() if k != 'time'},
                        None_stage: None,
                    }
                    global_best_config_csv[batchsize] = {
                        stage_name: {k: v for k, v in best_stage[block_size_m].items() if k != 'time'},
                        None_stage: None,
                        "time" : {stage_name+"_time": best_stage[block_size_m]['time'], 
                                None_stage+"_time": best_stage2[block_size_m]['time'],
                                "total_time" : s_time}
                    }
        else:
            print(f"{batchsize=} stage1 and stage2 data file is not found.")

    savepath = save_configs(args.w8a8)
    with open(f'./tp{args.tp_size}/' + savepath, 'w') as f:
        json.dump(global_best_config, f, indent=4)
    with open(savepath[:-5]+"_stage1.csv", 'w', newline='') as csvfile1:   
        with open(savepath[:-5]+"_stage2.csv", 'w', newline='') as csvfile2:
            fieldnames = ['model', 'batch_size', 'stage', 'time', "totol_time",'BLOCK_SIZE_M',  'BLOCK_SIZE_N', 'BLOCK_SIZE_K', 'GROUP_SIZE_M',\
                        'num_warps', 'num_stages', 'pipeline', 'pipeline', 'scenario', 'ACCF32', 'SPLIT_K' ]
            csvwriter1 = csv.writer(csvfile1)
            csvwriter1.writerow(fieldnames)
            csvwriter2 = csv.writer(csvfile2)
            csvwriter2.writerow(fieldnames)
            for bs in global_best_config_csv:
                best_config = global_best_config_csv[bs]
                for sn in ['stage1', 'stage2']:
                    content = [name, bs, sn, best_config["time"][sn+"_time"], best_config["time"]["total_time"]]
                    if best_config[sn] is not None:
                        for attr in best_config[sn]:
                            content += [best_config[sn][attr]]
                    else:
                        content += (["None"] * 11)
                    if sn == 'stage1':
                        csvwriter1.writerow(content)
                    else:
                        csvwriter2.writerow(content)
                
    print(f"最优配置已保存到{savepath}")

if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        default="/pde_ai/models/llm/stepfun/step3v/docs2T/ckpt4_bf16")
                        # /pde_ai/models/llm/DeepSeek/DeepSeek-V3-BF16/
    parser.add_argument("--tp-size", "-tp", type=int, default=8)
    parser.add_argument("--w8a8", type=int, default=0)
    args = parser.parse_args()
    name = args.model.split("/")[-2]
    print(f"{name=}")
    main(name)