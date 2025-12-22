#pragma once

#include "utils.h"


template <typename T, typename T2>
__global__ void fused_bias_add_kernel(const T* __restrict__ input, 
                                        const T* __restrict__ bias,
                                        const T* __restrict__ residual, 
                                        T* __restrict__ output, 
                                        const float dropout_prob, 
                                        const int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        T2 input_val[4];
        // T2 bias_val[4];
        T2 residual_val[4];
        T2 output_val[4];

        *(float4*)(&input_val[0]) = *(float4*)(&input[idx << 3]);
        // *(float4*)(&bias_val[0]) = *(float4*)(&bias[idx << 3]);
        *(float4*)(&residual_val[0]) = *(float4*)(&residual[idx << 3]);
        
        LOOP_UNROLL
        for (int i = 0; i < 4; i++) {
            // output_val[i] = __hadd2(input_val[i], bias_val[i]);
            // output_val[i] = __hadd2(output_val[i], residual_val[i]);
            output_val[i] = __hadd2(input_val[i], residual_val[i]);
        }
        
        *(float4*)(&output[idx << 3]) = *(float4*)(&output_val[0]);
    }
}

template<typename T, typename T2>
__global__ void fused_bias_add_kernel_na(const T* __restrict__ input, 
                                        const T* __restrict__ bias,
                                        const T* __restrict__ residual, 
                                        T* __restrict__ output, 
                                        const float dropout_prob, 
                                        const int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int N = sizeof(float4) / sizeof(T);
    int64_t thread_offset = idx * N;
    if(thread_offset >= numel) return;
    const T * ptr_input = input + thread_offset;
    const T * ptr_residual = residual + thread_offset;
    T* ptr_output = output + thread_offset;
    int64_t length = numel - thread_offset;
    if(length < N) {
        for(int i = 0; i < length; i++) {
            ptr_output[i] = __hadd(ptr_input[i], ptr_residual[i]);
        }
    } else {
        float4 reg_input, reg_residual;
        float4 reg_dst;
        reg_input = *(float4*)ptr_input;
        reg_residual = *(float4*)ptr_residual;
        T2* ptr_reg_i = (T2*)&reg_input;
        T2* ptr_reg_r = (T2*)&reg_residual;
        T2* ptr_reg_dst = (T2*)&reg_dst;
        #pragma unroll 4
        for(int i = 0; i < 4; i++) {
            ptr_reg_dst[i] = __hadd2(ptr_reg_i[i], ptr_reg_r[i]);
        }
        *(float4*)(ptr_output) = reg_dst;
    }
}

template <typename T>
__global__ void fused_bias_dropout_add_kernel(const T* __restrict__ input, 
                                        const T* __restrict__ bias,
                                        const T* __restrict__ residual, 
                                        T* __restrict__ output, 
                                        const float dropout_prob, 
                                        const int numel) {
    // TODO: Implement dropout
    printf("Dropout not implemented yet\n");
}