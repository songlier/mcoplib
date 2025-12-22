#pragma once

#define USE_MACA

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef USE_MACA
    #include <mc_common.h>
    #include <mc_runtime.h>
    #include "common/maca_fp16.h"
    #include "common/maca_bfloat16.h"
#else
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>
#endif

#define DIV_UP(x, y) ((x) + (y) - 1) / (y)
#define LOOP_UNROLL _Pragma("unroll")

#ifdef USE_MACA
    #define bfloat16 maca_bfloat16  
    #define bfloat162 maca_bfloat162
#else
    #define bfloat16 __nv_bfloat16
    #define bfloat162 __nv_bfloat162
#endif

__device__ __forceinline__ float convert_fp16_to_fp32(half data) { return __half2float(data); }
__device__ __forceinline__ float convert_fp16_to_fp32(bfloat16 data) { return __bfloat162float(data); }

__device__ __forceinline__ float2 convert_2fp16_to_2fp32(half2 data) { return __half22float2(data); }
__device__ __forceinline__ float2 convert_2fp16_to_2fp32(bfloat162 data) { return __bfloat1622float2(data); }

template <typename T>
__device__ __forceinline__ T convert_fp32_to_fp16(float data) {
    return;
}
template <>
__device__ __forceinline__ half convert_fp32_to_fp16<half>(float data) {
    return __float2half(data);
}
template <>
__device__ __forceinline__ bfloat16 convert_fp32_to_fp16<bfloat16>(float data) {
    return __float2bfloat16(data);
}

template <typename T>
__device__ __forceinline__ T convert_fp32_to_fp16_rn(float data) {
    return;
}
template <>
__device__ __forceinline__ half convert_fp32_to_fp16_rn<half>(float data) {
    return __float2half_rn(data);
}
template <>
__device__ __forceinline__ bfloat16 convert_fp32_to_fp16_rn<bfloat16>(float data) {
    return __float2bfloat16_rn(data);
}

template <typename T>
__device__ __forceinline__ T convert_2fp32_to_2fp16_rn(float2 data) {
    return;
}
template <>
__device__ __forceinline__ half2 convert_2fp32_to_2fp16_rn<half2>(float2 data) {
    return __float22half2_rn(data);
}
template <>
__device__ __forceinline__ bfloat162 convert_2fp32_to_2fp16_rn<bfloat162>(float2 data) {
    return __float22bfloat162_rn(data);
}

template <typename T>
__device__ __forceinline__ T float_to_target(float value);

template <>
__device__ __forceinline__ bfloat16 float_to_target<bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <>
__device__ __forceinline__ half float_to_target<half>(float value) {
    return __float2half(value);
}

template <>
__device__ __forceinline__ float float_to_target<float>(float value) {
    return (value);
}

template <typename T>
__device__ __forceinline__ float target_to_float(T value);

template <>
__device__ __forceinline__ float target_to_float<bfloat16>(bfloat16 value) {
    return __bfloat162float(value);
}

template <>
__device__ __forceinline__ float target_to_float<half>(half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float target_to_float<float>(float value) {
    return value;
}

__device__ __forceinline__ int32_t ScanWarp(int32_t val) {
  int32_t lane = threadIdx.x & 31;
  int32_t tmp = __shfl_up_sync(0xffffffff, val, 1);
  if (lane >= 1) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffff, val, 2);
  if (lane >= 2) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffff, val, 4);
  if (lane >= 4) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffff, val, 8);
  if (lane >= 8) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffff, val, 16);
  if (lane >= 16) {
    val += tmp;
  }
  return val;
}

__device__ __forceinline__ int32_t ScanWarp64(int32_t val) {
  int32_t lane = threadIdx.x & 63;
  int32_t tmp = __shfl_up_sync(0xffffffffffffffff, val, 1);
  if (lane >= 1) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffffffffffff, val, 2);
  if (lane >= 2) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffffffffffff, val, 4);
  if (lane >= 4) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffffffffffff, val, 8);
  if (lane >= 8) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffffffffffff, val, 16);
  if (lane >= 16) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffffffffffff, val, 32);
  if (lane >= 32) {
    val += tmp;
  }
  return val;
}

// Using this func need blockDimx.x % 32 = 0
__device__ __forceinline__ int32_t ScanBlock(int32_t val, int32_t valid_block_size) {
    __shared__ int smem[32];
    int valid_val = threadIdx.x < valid_block_size ? val : 0;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x / 31;
    int cum_sum_warp_val = ScanWarp(valid_val);
    if (lane == 31) {
        smem[warp_id] = cum_sum_warp_val;
    }
    __syncthreads();
    if (warp_id == 0 && threadIdx.x < (valid_block_size  + 31) / 32) {
        int tmp = ScanWarp(smem[threadIdx.x]);
        smem[threadIdx.x] = tmp;
    }
    __syncthreads();
    if (warp_id > 0) {
        cum_sum_warp_val += smem[warp_id - 1];
    }
    return cum_sum_warp_val;
}

#include <torch/extension.h>
#include <torch/torch.h>

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), "Tensor " #x " must be on CUDA")
#define CHECK_DTYPE(x, true_dtype) \
    TORCH_CHECK(x.dtype() == true_dtype, "Tensor " #x " must have dtype (" #true_dtype ")")
#define CHECK_DTYPE_EQ(x, y) \
    TORCH_CHECK(x.dtype() == y.dtype(), "Tensor " #x " must have dtype equal to " #y)
#define CHECK_DIMS(x, true_dim) \
    TORCH_CHECK(x.dim() == true_dim, "Tensor " #x " must have dimension number (" #true_dim ")")
#define CHECK_NUMEL(x, minimum) \
    TORCH_CHECK(x.numel() >= minimum, "Tensor " #x " must have at last " #minimum " elements")
#define CHECK_SHAPE(x, ...) \
    TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), "Tensor " #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_SHAPE_EQ(x, y) \
    TORCH_CHECK(x.sizes() == y.sizes(), "Tensor " #x " must have shape equal to " #y)
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), "Tensor " #x " must be contiguous")
#define CHECK_LASTDIM_CONTIGUOUS(x) \
    TORCH_CHECK(x.stride(-1) == 1, "Tensor " #x " must be contiguous at the last dimension")

template <
    typename T,
    /// Number of elements in the array
    int N,
    /// Alignment requirement in bytes
    int Alignment = sizeof(T) * N
>
class alignas(Alignment) AlignedArrayI4 {
public:
    T data[N];
};

static __device__ __forceinline__ int8_t float_to_int8_rn(float const x) {
  // CUDA path
  int32_t dst;
  dst = __float2int_rn(x);
  dst = min(dst, 127);
  dst = max(dst, -127);
  return reinterpret_cast<const int8_t&>(dst);
}