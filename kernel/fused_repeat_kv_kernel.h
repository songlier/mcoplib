#pragma once

#include "utils.h"

template<typename scalar_t, int num_threads>
__global__ void fused_repeat_kv_fwd_kernel_opt(
    scalar_t* input,
    scalar_t* output,
    const int q_num_head,
    const int kv_num_head,
    const int head_dim,
    const int num_elems_per_batch,
    const int ielems_per_batch
)
{
    constexpr int N = 16 / sizeof(scalar_t);
    int v0 = q_num_head / kv_num_head;
    int batch_offset = blockIdx.y * num_elems_per_batch;
    int input_batch_offset = blockIdx.y * ielems_per_batch;
    int x_offset = (blockIdx.x * num_threads + threadIdx.x) * N;
    if(x_offset >= num_elems_per_batch) return;
    int b0, remain, q_num_head_id , kv_num_head_id, head_id, v1;
    int patch0 = q_num_head * head_dim;
    b0 = x_offset / patch0;
    remain = x_offset % patch0;
    q_num_head_id = remain / head_dim;
    kv_num_head_id = q_num_head_id / v0;
    v1 = q_num_head_id % v0;
    head_id = remain % head_dim;
    int input_offset = kv_num_head_id * (v0 + 2) * head_dim;
    if(b0 == 0) {
        input_offset = input_offset + head_id + v1 * head_dim;
    } else if(b0 == 1) {
        input_offset = input_offset + v0 * head_dim + head_id;
    } else {
        input_offset = input_offset + (v0 + 1) * head_dim + head_id;
    }
    *(float4*)(output + batch_offset + x_offset) = *(float4*)(input + input_batch_offset + input_offset);
}

// input   seq, bs, partition, (q_num_head / partition + kv_num_head * 2 / partition) * head_dim
// output  seq, bs, 3, q_num_head, head_dim
template <typename scalar_t>
__global__ void fused_repeat_kv_fwd_kernel(
    scalar_t* input, 
    scalar_t* output,
    const int seq_len, 
    const int batch_size, 
    const int partition, 
    const int q_num_head,
    const int kv_num_head,
    const int head_dim){
    
    int v0 = q_num_head / kv_num_head;
    int thread_id = threadIdx.x;    
    int seq_id = blockIdx.x;
    int batch_id = blockIdx.y;
    int thread_per_head = head_dim / 8;
    int partition_id =  thread_id / thread_per_head % partition;
    int q_num_head_id = thread_id / thread_per_head / partition * v0;
    int kv_num_head_id = thread_id / thread_per_head / partition;
    int q_num_head_partition_sep = q_num_head / partition;
    int kv_num_head_partition_sep = kv_num_head / partition;
    

    int head_dim_id = thread_id % thread_per_head * 8;

    int src_partition_stride = (q_num_head + 2 * kv_num_head) / partition * head_dim;
    int src_batch_stride = partition * src_partition_stride;
    int src_seq_stride =  batch_size * src_batch_stride;

    int dst_batch_stride = 3 * q_num_head * head_dim;
    int dst_seq_stride = batch_size * dst_batch_stride;

    int q_src_index = seq_id * src_seq_stride + batch_id * src_batch_stride + partition_id * src_partition_stride + q_num_head_id * head_dim + head_dim_id;
    int q_dst_index = seq_id * dst_seq_stride + batch_id * dst_batch_stride + (partition_id * q_num_head_partition_sep + q_num_head_id) * head_dim + head_dim_id;


    int k_src_index = seq_id * src_seq_stride + batch_id * src_batch_stride + partition_id * src_partition_stride + (q_num_head_partition_sep + kv_num_head_id) * head_dim + head_dim_id;
    int k_dst_index = seq_id * dst_seq_stride + batch_id * dst_batch_stride + (v0 * partition_id * kv_num_head_partition_sep + kv_num_head_id) * head_dim + head_dim_id + q_num_head * head_dim;


    int v_src_index = k_src_index + head_dim;
    int v_dst_index = k_dst_index + q_num_head * head_dim;

    scalar_t reg_key[8];
    scalar_t reg_value[8];

    *(float4*)(&reg_key[0])   = *(float4*)(&input[k_src_index]);
    *(float4*)(&reg_value[0]) = *(float4*)(&input[v_src_index]);

    for(int i = 0; i < v0; ++i){
        *(float4*)(&output[q_dst_index + i * head_dim]) = *(float4*)(&input[q_src_index + i * head_dim]);
    }

    for(int i = 0; i < v0; ++i){
        *(float4*)(&output[k_dst_index + i * head_dim]) = *(float4*)(&reg_key[0]);
        *(float4*)(&output[v_dst_index + i * head_dim]) = *(float4*)(&reg_value[0]);
    }    
}

template <typename scalar_t>
__global__ void fused_repeat_kv_bwd_kernel(
    scalar_t* input, 
    scalar_t* output,
    const int seq_len, 
    const int batch_size, 
    const int partition, 
    const int q_num_head,
    const int kv_num_head,
    const int head_dim){
    int v0 = q_num_head / kv_num_head;
    int thread_id = threadIdx.x;    
    int seq_id = blockIdx.x;
    int batch_id = blockIdx.y;
    int thread_per_head = head_dim >> 3;
    int value0 = thread_id / thread_per_head;
    int value1 = value0 / partition;
    int partition_id =  value0 % partition;
    int q_num_head_id = value1 * v0;
    int kv_num_head_id = value1 ;
    int q_num_head_partition_sep = q_num_head / partition;
    int kv_num_head_partition_sep = kv_num_head / partition;
    
    int head_dim_id = thread_id % thread_per_head * 8;

    int dst_partition_stride = (q_num_head + 2 * kv_num_head) / partition * head_dim;
    int dst_batch_stride = partition * dst_partition_stride;
    int dst_seq_stride =  batch_size * dst_batch_stride;

    int src_batch_stride = 3 * q_num_head * head_dim;
    int src_seq_stride = batch_size * src_batch_stride;
    int src_seq_size = seq_id * src_seq_stride + batch_id * src_batch_stride;
    int dst_seq_size = seq_id * dst_seq_stride + batch_id * dst_batch_stride + partition_id * dst_partition_stride;
    int q_dst_index = dst_seq_size + q_num_head_id * head_dim + head_dim_id;
    int q_src_index = src_seq_size + (partition_id * q_num_head_partition_sep + q_num_head_id) * head_dim + head_dim_id;
    int k_dst_index = dst_seq_size + (q_num_head_partition_sep + kv_num_head_id) * head_dim + head_dim_id;
    int k_src_index = src_seq_size + (v0 * partition_id * kv_num_head_partition_sep + kv_num_head_id) * head_dim + head_dim_id + q_num_head * head_dim;
    int v_dst_index = k_dst_index + head_dim;
    int v_src_index = k_src_index + q_num_head * head_dim;
    float reg_float_key[8] = {0.0f};
    float reg_float_value[8] = {0.0f};
    scalar_t reg_key[8];
    scalar_t reg_value[8];
    scalar_t result_reg_key[8];
    scalar_t result_reg_value[8];
    for(int i = 0; i < v0; ++i){
        int stride = i * head_dim;
        *(float4*)(&output[q_dst_index + i * head_dim]) = *(float4*)(&input[q_src_index + stride]);
        *(float4*)(&reg_key[0])   = *(float4*)(&input[k_src_index + stride]);
        *(float4*)(&reg_value[0]) = *(float4*)(&input[v_src_index + stride]);
        #pragma unroll
        for(int j =0; j < 8; ++j){
            reg_float_key[j]   += convert_fp16_to_fp32(reg_key[j]);
            reg_float_value[j] += convert_fp16_to_fp32(reg_value[j]);
        }
    }
    #pragma unroll
    for(int i = 0; i < 8; ++i){
        result_reg_key[i]   = convert_fp32_to_fp16<scalar_t>(reg_float_key[i]);
        result_reg_value[i] = convert_fp32_to_fp16<scalar_t>(reg_float_value[i]);
    }
    *(float4*)(&output[k_dst_index]) = *(float4*)(&result_reg_key[0]);
    *(float4*)(&output[v_dst_index]) = *(float4*)(&result_reg_value[0]);
}