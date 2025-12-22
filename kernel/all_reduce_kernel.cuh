#pragma once

#include "utils.h"
#define REDUCE_WARP_SIZE 32

template<typename T>
__device__ __forceinline__ T max_(T v1, T v2){
  return v1 > v2? v1 : v2;
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = 32, unsigned int mask = 0xffffffff)
{
    return __shfl_down_sync(mask, value, delta, width);
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, unsigned int delta, int width = 32, unsigned int mask = 0xffffffff)
{
    return __shfl_xor_sync(mask, value, delta, width);
}

struct Block1D {
    static __forceinline__ __device__ int Tid() {return threadIdx.x;}

    static __forceinline__ __device__ int Warps() {
        return (blockDim.x + REDUCE_WARP_SIZE - 1) / REDUCE_WARP_SIZE;
    }
};

struct Block2D {
    static __forceinline__ __device__ int Tid() {
        return threadIdx.x + threadIdx.y * blockDim.x;
    }

    static __forceinline__ __device__ int Warps() {
        return (blockDim.x * blockDim.y + REDUCE_WARP_SIZE - 1) / REDUCE_WARP_SIZE;
    }
};

template <typename T>
__forceinline__ __device__ T WarpReduceMax(T val, int size = REDUCE_WARP_SIZE) {
#pragma unroll
  for (int offset = (REDUCE_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val = max_(val, WARP_SHFL_DOWN(val, offset));
  }
  return val;
}

template <typename T>
__forceinline__ __device__ T WarpReduceSum(T val, int size = REDUCE_WARP_SIZE) {
#pragma unroll
  for (int offset = (size >> 1); offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
}

template <typename T>
__forceinline__ __device__ T WarpAllReduceSum(T val, int size = REDUCE_WARP_SIZE) {
#pragma unroll
  for (int offset = (size >> 1); offset > 0; offset >>= 1) {
    val += WARP_SHFL_XOR(val, offset);
  }
  return val;
}

template <typename T>
__forceinline__ __device__ T WarpAllReduceMax(T val, int size = REDUCE_WARP_SIZE) {
#pragma unroll
  for (int offset = (size >> 1); offset > 0; offset >>= 1) {
    val = max_(val, WARP_SHFL_XOR(val, offset));
  }
  return val;
}

template <typename T, typename B = Block1D>
__inline__ __device__ T BlockReduceSum(T val) {
  __shared__ T smem[32];
  const int tid = B::Tid();
  const int lid = tid % REDUCE_WARP_SIZE;
  const int wid = tid / REDUCE_WARP_SIZE;

  val = WarpReduceSum(val);
  __syncthreads(); // prevent races when BlockReduces are called in a row.
  if (lid == 0) {
    smem[wid] = val;
  }
  __syncthreads();
  if (wid == 0) {
    val = (tid < B::Warps()) ? smem[lid] : (0.0f);
    val = WarpReduceSum(val);
  }
  return val;
}

template <typename T, typename B = Block1D>
__inline__ __device__ T BlockAllReduceSum(T val) {
  __shared__ T smem[32];
  const int tid = B::Tid();
  const int lid = tid % REDUCE_WARP_SIZE;
  const int wid = tid / REDUCE_WARP_SIZE;

  val = WarpReduceSum(val);
  __syncthreads(); // prevent races when BlockReduces are called in a row.
  if (lid == 0) {
    smem[wid] = val;
  }
  __syncthreads();
  val = (tid < B::Warps()) ? smem[lid] : (0.0f);
  if (wid == 0) {
  val = WarpReduceSum(val); 
    if (lid == 0) {
      smem[0] = val;
    }
  }
  __syncthreads();
  return smem[0];
}

template <typename T, typename B = Block1D>
__inline__ __device__ T BlockReduceMax(T val) {
  __shared__ T smem[32];
  const int tid = B::Tid();
  const int lid = tid % REDUCE_WARP_SIZE;
  const int wid = tid / REDUCE_WARP_SIZE;

  val = WarpReduceMax(val);
  __syncthreads();
  if (lid == 0) {
    smem[wid] = val;
  }
  __syncthreads();
  val = (tid < B::Warps()) ? smem[lid] : float_to_target<T>(-9999.f);
  if (wid == 0) {
    val = WarpReduceMax(val);
  }
  return val;
}

template <typename T, typename B = Block1D>
__inline__ __device__ T BlockAllReduceMax(T val, T* shared) {
  const int tid = B::Tid();
  const int lid = tid % REDUCE_WARP_SIZE;
  const int wid = tid / REDUCE_WARP_SIZE;
  val = WarpReduceMax(val);
  __syncthreads(); // prevent races when BlockReduces are called in a row.
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (tid < B::Warps()) ? shared[lid] : T(-99999);
  if (wid == 0) {
  val = WarpReduceMax(val); 
    if (lid == 0) {
      shared[0] = val;
    }
  }
  __syncthreads();
  return shared[0];
}

#undef REDUCE_WARP_SIZE