#pragma once

#include "utils.h"


template <typename scalar_t_in, typename scalar_t_out>
__global__ void fused_rope_fwd_kernel(
    scalar_t_in* qkv, 
    scalar_t_in* cos, 
    scalar_t_in* sin, 
    long* indexes, 
    scalar_t_out* qkv_output,
    const int seq_len, 
    const int qkv_num,
    const int num_head, 
    const int head_dim, 
    const int head_dim_half, 
    const int element_num){


    int seq_id = blockIdx.x;
    
    int thread_per_head = blockDim.x / num_head; // 8

    int num_head_id = threadIdx.x / thread_per_head;
    
    int head_dim_id = threadIdx.x % thread_per_head * 8; // 8 for fetch 8

    int qkv_indx = seq_id * qkv_num * num_head * head_dim;

    int q_index = qkv_indx + num_head_id * head_dim + head_dim_id;
    int k_index = q_index + num_head * head_dim;

    int seq_index = indexes == nullptr ? seq_id : indexes[seq_id];
    int sin_cos_index = seq_index * head_dim_half + head_dim_id;


    scalar_t_in cos_value[8];
    scalar_t_in sin_value[8];

    scalar_t_in q_value[2][8];
    scalar_t_in k_value[2][8];

    float float_cos_value[8];
    float float_sin_value[8];

    float float_q_value[2][8];
    float float_k_value[2][8];

    float float_q_result[2][8];
    float float_k_result[2][8];

    scalar_t_out q_result[2][8];
    scalar_t_out k_result[2][8];


    *(float4*)(&cos_value[0]) = *(float4*)(&cos[sin_cos_index]);
    *(float4*)(&sin_value[0]) = *(float4*)(&sin[sin_cos_index]);

    *(float4*)(&q_value[0]) = *(float4*)(&qkv[q_index]);
    *(float4*)(&q_value[1]) = *(float4*)(&qkv[q_index + head_dim_half]);

    *(float4*)(&k_value[0]) = *(float4*)(&qkv[k_index]);
    *(float4*)(&k_value[1]) = *(float4*)(&qkv[k_index + head_dim_half]);

    #pragma unroll
    for(int i = 0; i < 8; i++){
        float_cos_value[i] = convert_fp16_to_fp32(cos_value[i]);
        float_sin_value[i] = convert_fp16_to_fp32(sin_value[i]);

        float_q_value[0][i] = convert_fp16_to_fp32(q_value[0][i]);
        float_q_value[1][i] = convert_fp16_to_fp32(q_value[1][i]);

        float_k_value[0][i] = convert_fp16_to_fp32(k_value[0][i]);
        float_k_value[1][i] = convert_fp16_to_fp32(k_value[1][i]);

        float_q_result[0][i] = float_q_value[0][i] * float_cos_value[i] - float_q_value[1][i] * float_sin_value[i];
        float_q_result[1][i] = float_q_value[0][i] * float_sin_value[i] + float_q_value[1][i] * float_cos_value[i];

        float_k_result[0][i] = float_k_value[0][i] * float_cos_value[i] - float_k_value[1][i] * float_sin_value[i];
        float_k_result[1][i] = float_k_value[0][i] * float_sin_value[i] + float_k_value[1][i] * float_cos_value[i];

        q_result[0][i] = convert_fp32_to_fp16<scalar_t_out>(float_q_result[0][i]);
        q_result[1][i] = convert_fp32_to_fp16<scalar_t_out>(float_q_result[1][i]);

        k_result[0][i] = convert_fp32_to_fp16<scalar_t_out>(float_k_result[0][i]);
        k_result[1][i] = convert_fp32_to_fp16<scalar_t_out>(float_k_result[1][i]);

    }
    
    *(float4*)(&qkv_output[q_index]) = *(float4*)(&q_result[0]);
    *(float4*)(&qkv_output[q_index + head_dim_half]) = *(float4*)(&q_result[1]);

    *(float4*)(&qkv_output[k_index]) = *(float4*)(&k_result[0]);
    *(float4*)(&qkv_output[k_index + head_dim_half]) = *(float4*)(&k_result[1]);
}




template <typename scalar_t_in, typename scalar_t_out>
__global__ void fused_rope_bwd_kernel(
    scalar_t_out* qkv, 
    scalar_t_in* cos, 
    scalar_t_in* sin, 
    long* indexes, 
    scalar_t_in* qkv_output,
    const int seq_len, 
    const int qkv_num,
    const int num_head, 
    const int head_dim, 
    const int head_dim_half, 
    const int element_num){


    int seq_id = blockIdx.x;
    
    int thread_per_head = blockDim.x / num_head; // 8

    int num_head_id = threadIdx.x / thread_per_head;
    
    int head_dim_id = threadIdx.x % thread_per_head * 8; // 8 for fetch 8

    int qkv_indx = seq_id * qkv_num * num_head * head_dim;

    int q_index = qkv_indx + num_head_id * head_dim + head_dim_id;
    int k_index = q_index + num_head * head_dim;

    int seq_index = indexes == nullptr ? seq_id : indexes[seq_id];
    int sin_cos_index = seq_index * head_dim_half + head_dim_id;


    scalar_t_in cos_value[8];
    scalar_t_in sin_value[8];

    scalar_t_out q_value[2][8];
    scalar_t_out k_value[2][8];

    float float_cos_value[8];
    float float_sin_value[8];

    float float_q_value[2][8];
    float float_k_value[2][8];

    float float_q_result[2][8];
    float float_k_result[2][8];

    scalar_t_in q_result[2][8];
    scalar_t_in k_result[2][8];


    *(float4*)(&cos_value[0]) = *(float4*)(&cos[sin_cos_index]);
    *(float4*)(&sin_value[0]) = *(float4*)(&sin[sin_cos_index]);

    *(float4*)(&q_value[0]) = *(float4*)(&qkv[q_index]);
    *(float4*)(&q_value[1]) = *(float4*)(&qkv[q_index + head_dim_half]);

    *(float4*)(&k_value[0]) = *(float4*)(&qkv[k_index]);
    *(float4*)(&k_value[1]) = *(float4*)(&qkv[k_index + head_dim_half]);

    #pragma unroll
    for(int i = 0; i < 8; i++){
        float_cos_value[i] = convert_fp16_to_fp32(cos_value[i]);
        float_sin_value[i] = convert_fp16_to_fp32(sin_value[i]);

        float_q_value[0][i] = convert_fp16_to_fp32(q_value[0][i]);
        float_q_value[1][i] = convert_fp16_to_fp32(q_value[1][i]);

        float_k_value[0][i] = convert_fp16_to_fp32(k_value[0][i]);
        float_k_value[1][i] = convert_fp16_to_fp32(k_value[1][i]);

        float_q_result[0][i] = float_q_value[0][i] * float_cos_value[i] + float_q_value[1][i] * float_sin_value[i];
        float_q_result[1][i] = - float_q_value[0][i] * float_sin_value[i] + float_q_value[1][i] * float_cos_value[i];

        float_k_result[0][i] = float_k_value[0][i] * float_cos_value[i] + float_k_value[1][i] * float_sin_value[i];
        float_k_result[1][i] = - float_k_value[0][i] * float_sin_value[i] + float_k_value[1][i] * float_cos_value[i];

        q_result[0][i] = convert_fp32_to_fp16<scalar_t_in>(float_q_result[0][i]);
        q_result[1][i] = convert_fp32_to_fp16<scalar_t_in>(float_q_result[1][i]);

        k_result[0][i] = convert_fp32_to_fp16<scalar_t_in>(float_k_result[0][i]);
        k_result[1][i] = convert_fp32_to_fp16<scalar_t_in>(float_k_result[1][i]);

    }
    
    *(float4*)(&qkv_output[q_index]) = *(float4*)(&q_result[0]);
    *(float4*)(&qkv_output[q_index + head_dim_half]) = *(float4*)(&q_result[1]);

    *(float4*)(&qkv_output[k_index]) = *(float4*)(&k_result[0]);
    *(float4*)(&qkv_output[k_index + head_dim_half]) = *(float4*)(&k_result[1]);
}