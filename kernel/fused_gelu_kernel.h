#pragma once

#include "utils.h"

template<typename T>
__device__ __forceinline__ T gelu_kernel(T x, T bias)
{
    printf("not support\n");
}

template<>
__device__ __forceinline__ float gelu_kernel<float>(float x, float bias) {
    x = x + bias;
    return x * 0.5 * (1.0 + tanh(0.79788456 * x * (1 + 0.044715 * x * x)));
}

template<>
__device__ __forceinline__ half gelu_kernel<half>(half x, half bias) {
    float x_f = convert_fp16_to_fp32(x);
    float b_f = convert_fp16_to_fp32(bias);
    x_f = x_f + b_f;
    float d = x_f * 0.5 * (1.0 + tanh(0.79788456 * x_f * (1 + 0.044715 * x_f * x_f)));
    return convert_fp32_to_fp16<half>(d);
}

template<>
__device__ __forceinline__ bfloat16 gelu_kernel<bfloat16>(bfloat16 x, bfloat16 bias) {
    float x_f = convert_fp16_to_fp32(x);
    float b_f = convert_fp16_to_fp32(bias);
    x_f = x_f + b_f;
    float d = x_f * 0.5 * (1.0 + tanh(0.79788456 * x_f * (1 + 0.044715 * x_f * x_f)));
    return convert_fp32_to_fp16<bfloat16>(d);
}

template<typename T>
__device__ __forceinline__ T gelu_bwd_kernel(T x, T y, T bias)
{
    printf("not support\n");
}

template<>
__device__ __forceinline__ float gelu_bwd_kernel<float>(float x, float y, float bias) {
    x = x + bias;
    float tanh_out = tanh(0.79788456 * x * (1 + 0.044715 * x * x));
    float ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out);
    return ff * y;
}

template<>
__device__ __forceinline__ half gelu_bwd_kernel<half>(half x, half y, half bias) {
    float x_f = convert_fp16_to_fp32(x);
    float b_f = convert_fp16_to_fp32(bias);
    x_f = x_f + b_f;
    float tanh_out = tanh(0.79788456 * x_f * (1 + 0.044715 * x_f * x_f));
    float ff = 0.5 * x_f * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x_f * x_f)) + 0.5 * (1 + tanh_out);
    float y_f = convert_fp16_to_fp32(y);
    return convert_fp32_to_fp16<half>(ff*y_f);
}

template<>
__device__ __forceinline__ bfloat16 gelu_bwd_kernel<bfloat16>(bfloat16 x, bfloat16 y, bfloat16 bias) {
    float x_f = convert_fp16_to_fp32(x);
    float b_f = convert_fp16_to_fp32(bias);
    x_f = x_f + b_f;
    float tanh_out = tanh(0.79788456 * x_f * (1 + 0.044715 * x_f * x_f));
    float ff = 0.5 * x_f * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x_f * x_f)) + 0.5 * (1 + tanh_out);
    float y_f = convert_fp16_to_fp32(y);
    return convert_fp32_to_fp16<bfloat16>(ff*y_f);
}

template<typename T, int num_threads>
__global__ void fused_gelu_fwd_kernel(const T* __restrict__ input, const T* __restrict__ bias, T* __restrict__ output, const int num_elems, const int hidden_size){
    constexpr int N = 16 / sizeof(T);
    int idx = (blockIdx.x * num_threads + threadIdx.x)*N;
    if(idx >= num_elems) return;
    if(num_elems - idx < N) {
        int length = num_elems - idx;
        const T * ptr_input = input + idx;
        T* ptr_output = output + idx;
        for(int i = 0; i < length; i++) {
            ptr_output[i] = gelu_kernel<T>(ptr_input[i], bias[(idx + i) % hidden_size]);
        }
    } else {
        float4 reg_input = *(float4*)(input + idx);
        float4 reg_output;
        T* ptr_reg_input = (T*)&reg_input;
        T* ptr_reg_output = (T*)&reg_output;
        #pragma unroll N
        for(int i = 0; i < N; i++) {
            ptr_reg_output[i] = gelu_kernel<T>(ptr_reg_input[i], bias[((idx + i) % hidden_size)]);
        }
        *(float4*)(output + idx) = reg_output;
    }
}

template<typename T, int num_threads>
__global__ void fused_gelu_fwd_align_kernel(const T* __restrict__ input, const T* __restrict__ bias, T* __restrict__ output, const int num_elems, const int hidden_size){
    constexpr int N = 16 / sizeof(T);
    int idx = (blockIdx.x * num_threads + threadIdx.x)*N;
    if(idx >= num_elems) return;
    float4 reg_input = *(float4*)(input + idx);
    float4 reg_output;
    T* ptr_reg_input = (T*)&reg_input;
    T* ptr_reg_output = (T*)&reg_output;
    int bias_offset = idx % hidden_size;
    float4 reg_bias = *(float4*)(bias + bias_offset);
    T* ptr_reg_bias = (T*)&reg_bias;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
        ptr_reg_output[i] = gelu_kernel<T>(ptr_reg_input[i], ptr_reg_bias[i]);
    }
    *(float4*)(output + idx) = reg_output;
}

template<typename T, int num_threads>
__global__ void fused_gelu_bwd_kernel(const T* __restrict__ input, const T* __restrict__ input1, const T* __restrict__ bias, T* __restrict__ output, const int num_elems, const int hidden_size){
    constexpr int N = 16 / sizeof(T);
    int idx = (blockIdx.x * num_threads + threadIdx.x)*N;
    if(idx >= num_elems) return;
    if(num_elems - idx < N) {
        int length = num_elems - idx;
        const T * ptr_input = input + idx;
        const T* ptr_input1 = input1 + idx;
        T* ptr_output = output + idx;
        for(int i = 0; i < length; i++) {
            ptr_output[i] = gelu_bwd_kernel<T>(ptr_input[i], ptr_input1[i], bias[(idx + i)% hidden_size]);
        }
    } else {
        float4 reg_input = *(float4*)(input + idx);
        float4 reg_input1 = *(float4*)(input1 + idx);
        float4 reg_output;
        T* ptr_reg_input = (T*)&reg_input;
        T* ptr_reg_input1 = (T*)&reg_input1;
        T* ptr_reg_output = (T*)&reg_output;
        #pragma unroll N
        for(int i = 0; i < N; i++) {
            ptr_reg_output[i] = gelu_bwd_kernel<T>(ptr_reg_input[i], ptr_reg_input1[i], bias[(idx + i)%hidden_size]);
        }
        *(float4*)(output + idx) = reg_output;
    }
}

template<typename T, int num_threads>
__global__ void fused_gelu_bwd_align_kernel(const T* __restrict__ input, const T* __restrict__ input1, const T*__restrict__ bias, T* __restrict__ output, const int num_elems, const int hidden_size){
    constexpr int N = 16 / sizeof(T);
    int idx = (blockIdx.x * num_threads + threadIdx.x)*N;
    if(idx >= num_elems) return;
    float4 reg_input = *(float4*)(input + idx);
    float4 reg_input1 = *(float4*)(input1 + idx);
    float4 reg_bias = *(float4*)(bias + (idx % hidden_size));
    float4 reg_output;
    T* ptr_reg_input = (T*)&reg_input;
    T* ptr_reg_input1 = (T*)&reg_input1;
    T* ptr_reg_output = (T*)&reg_output;
    T* ptr_reg_bias = (T*)&reg_bias;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
        ptr_reg_output[i] = gelu_bwd_kernel<T>(ptr_reg_input[i], ptr_reg_input1[i], ptr_reg_bias[i]);
    }
    *(float4*)(output + idx) = reg_output;
}

