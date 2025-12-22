#pragma once

#include "utils.h"


template <typename T, typename T2>
__global__ void fused_bias_swiglu_fwd_kernel(const T* __restrict__ input, 
                                                const T* __restrict__ bias,
                                                T* __restrict__ output, 
                                                const int dim,
                                                const int stride_in,
                                                const int stride_out) {
    int bid = blockIdx.x;
    int tid = blockIdx.y * blockDim.x + threadIdx.x;

    if (((tid + 1) << 3) > dim) {
        return;
    }

    int row = bid;
    int col = tid << 3;

    T2 input_val[2][4];
    // T2 bias_val[2][4];
    T2 output_val[4];

    float2 datax, datay;

    *(float4*)(&input_val[0][0]) = *(float4*)(&input[row * stride_in + col]);
    *(float4*)(&input_val[1][0]) = *(float4*)(&input[row * stride_in + col + dim]);
    // *(float4*)(&bias_val[0][0]) = *(float4*)(&bias[col]);
    // *(float4*)(&bias_val[1][0]) = *(float4*)(&bias[col + dim]);
    
    LOOP_UNROLL
    for (int i = 0; i < 4; i++) {
        // datax = convert_2fp16_to_2fp32(__hadd2(input_val[0][i], bias_val[0][i]));
        // datay = convert_2fp16_to_2fp32(__hadd2(input_val[1][i], bias_val[1][i]));
        datax = convert_2fp16_to_2fp32(input_val[0][i]);
        datay = convert_2fp16_to_2fp32(input_val[1][i]);
        datax.x = __fdividef(datax.x, __expf(-1.0f * datax.x) + 1.0f) * datay.x;
        datax.y = __fdividef(datax.y, __expf(-1.0f * datax.y) + 1.0f) * datay.y;
        output_val[i] = convert_2fp32_to_2fp16_rn<T2>(datax);
    }
    
    *(float4*)(&output[row * stride_out + col]) = *(float4*)(&output_val[0]);
}

template <typename T, typename T2>
__global__ void fused_bias_swiglu_bwd_kernel(const T* __restrict__ input, 
                                                const T* __restrict__ bias,
                                                const T* __restrict__ grad_output, 
                                                T* __restrict__ grad_input, 
                                                T* __restrict__ grad_bias, 
                                                const int dim,
                                                const int stride_in,
                                                const int stride_out) {
    int bid = blockIdx.x;
    int tid = blockIdx.y * blockDim.x + threadIdx.x;

    if (((tid + 1) << 3) > dim) {
        return;
    }

    int row = bid;
    int col = tid << 3;

    T2 input_val[2][4];
    T2 grad_output_val[4];
    T2 grad_input_val[2][4];

    float2 datax, datay, datag;
    float tmp_sigmoid_x, tmp_x, tmp_y;

    *(float4*)(&input_val[0][0]) = *(float4*)(&input[row * stride_in + col]);
    *(float4*)(&input_val[1][0]) = *(float4*)(&input[row * stride_in + col + dim]);
    *(float4*)(&grad_output_val[0]) = *(float4*)(&grad_output[row * stride_out + col]);
    
    LOOP_UNROLL
    for (int i = 0; i < 4; i++) {
        datax = convert_2fp16_to_2fp32(input_val[0][i]);
        datay = convert_2fp16_to_2fp32(input_val[1][i]);
        datag = convert_2fp16_to_2fp32(grad_output_val[i]);

        // sigmoid(x)
        tmp_sigmoid_x = __fdividef(1.0f, __expf(-1.0f * datax.x) + 1.0f);
        tmp_x = datax.x;
        tmp_y = datay.x;
        // d_out * y * sigmoid(x) + dout * y * sigmoid(x) * (1 - sigmoid(x)) * x
        datax.x = datag.x * tmp_y * tmp_sigmoid_x * (1 + tmp_x * (1 - tmp_sigmoid_x));
        // d_out * x * sigmoid(x)
        datay.x = datag.x * tmp_x * tmp_sigmoid_x;

        tmp_sigmoid_x = __fdividef(1.0f, __expf(-1.0f * datax.y) + 1.0f);
        tmp_x = datax.y;
        tmp_y = datay.y;
        datax.y = datag.y * tmp_y * tmp_sigmoid_x * (1 + tmp_x * (1 - tmp_sigmoid_x));
        datay.y = datag.y * tmp_x * tmp_sigmoid_x;

        grad_input_val[0][i] = convert_2fp32_to_2fp16_rn<T2>(datax);
        grad_input_val[1][i] = convert_2fp32_to_2fp16_rn<T2>(datay);
    }
    
    *(float4*)(&grad_input[row * stride_in + col]) = *(float4*)(&grad_input_val[0][0]);
    *(float4*)(&grad_input[row * stride_in + col + dim]) = *(float4*)(&grad_input_val[1][0]);
}
