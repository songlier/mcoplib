#pragma once
#include "utils.cuh"
// #include <cuda_runtime.h>
#include "sglang/include/scalar_type.hpp"

#define WAVES_PER_BLOCK (THREADS/WAVE)
#define TILE_M (BLOCKS_M*SLICE_M)
#define TILE_N (BLOCKS_N*SLICE_N)
#define TILE_K (BLOCKS_K*SLICE_K)
#define N_ITERS (TILE_N / (WAVES_PER_BLOCK*SLOT))
#define LOADING_A_LOOP SLICE_K * TILE_M / (sizeof(PackType) / sizeof(scalar_t)) / THREADS
#define AS_PTR_B128(x) ((PackTypeInt4*)x)
#define AS_PTR_B64(x) ((PackTypeInt2*)x)
#define AS_PTR_B32(x) ((float*)x)
#define AS_PTR_B16(x) ((half*)x)
#define AS_PTR_B8(x) ((uint8_t*)x)

#define div_ceil(x, y) (x + y - 1) / (y)

int inline get_power2(uint32_t v) {
    uint32_t power = 0;
    uint32_t mask = 0x00000001;
    while (power < 32) {
        if ((v & mask) > 0) break;
        power++;
        mask <<= 1;
    }
    if ((1 << power) != v) return -1;
    return static_cast<int>(power);
}


namespace hgemm_marlin_gptq {

//#define DEBUG
using PackTypeInt4 = b128VecType;
using PackTypeInt2 = b64VecType;
using PackType = uint32_t;

typedef __NATIVE_VECTOR__(2, float) v2f;
using PackTypeFloat2 = v2f;
constexpr static int Q4BITS = 4;
constexpr static int Q8BITS = 8;
constexpr static int PACK_RATIO_4BITS = sizeof(PackType) * 8 / Q4BITS;
constexpr static int PACK_RATIO_8BITS = sizeof(PackType) * 8 / Q8BITS;
constexpr static int SLICE_M = 16;
constexpr static int SLICE_N = 16;
constexpr static int SLICE_K = 32;
constexpr static int PAD_SLICE_K = 40;
constexpr static int SLOT    = 16;
constexpr static int WAVE    = 64;
constexpr static int WAVE_SLOTS = 4;
constexpr static int PEUS = 13*8*4; //For C500, There are 8 DPC and each DPC have 13 APs, each AP have 4 PEUs
constexpr static int MAX_BLOCKS_M = 4;

template<typename scalar_t>
__device__ __forceinline__ scalar_t float2scalar(float x) {
  scalar_t out = 0;
  return out;
}

template<>
__device__ __forceinline__ half float2scalar(float x) {
  return __float2half(x);
}

template<>
__device__ __forceinline__ nv_bfloat16 float2scalar(float x) {
  return __float2bfloat16(x);
}

template<typename scalar_t>
__device__ __forceinline__ float scalar2float(scalar_t x) {
    return (float)x;
}

template<>
__device__ __forceinline__ float scalar2float<half>(half x) {
    return __half2float(x);
}

template<>
__device__ __forceinline__ float scalar2float<nv_bfloat16>(nv_bfloat16 x) {
    return __bfloat162float(x);
}

// #define CVT_B0TOF32(q, out) asm volatile("cvt_b0tof32 %0,%1;\n":"=r"(out):"r"(q));
// #define CVT_B1TOF32(q, out) asm volatile("cvt_b1tof32 %0,%1;\n":"=r"(out):"r"(q));
// #define CVT_B2TOF32(q, out) asm volatile("cvt_b2tof32 %0,%1;\n":"=r"(out):"r"(q));
// #define CVT_B3TOF32(q, out) asm volatile("cvt_b3tof32 %0,%1;\n":"=r"(out):"r"(q));

//FIXME: We'd rather a quant group will not divided into serveral blocks
template<int BLOCKS_M, int BLOCKS_N, int BLOCKS_K>
struct TileManager {
    int tile_start_row;
    int tile_start_col;
    int tiles_k;
    int my_iters = 0;
    bool global_pred = true;
    int atomic_idx = 0;
    bool last_atomic_cache = false;

    __device__ __forceinline__ void init(int m, int n, int k, int bidx, int iters) {
        //Calculate tile start row and cols so we can calculate the offset address of a b and c
        int tile_idx = iters * bidx;
        int tiles_n = div_ceil(n, TILE_N);
        tiles_k = div_ceil(k, TILE_K);
        //if (tile_idx >= tiles_n*tiles_k) return false;
        global_pred = tile_idx < tiles_n * tiles_k;
        int tile_col = tile_idx / tiles_k;
        int tile_row = tile_idx - tile_col * tiles_k;
        tile_start_col = tile_col;
        tile_start_row = tile_row;
        my_iters = tile_idx + iters >= tiles_n*tiles_k ? tiles_n*tiles_k - tile_idx : iters;
        my_iters = global_pred ? my_iters : 0;
        atomic_idx = div_ceil(tile_start_row, iters);
        last_atomic_cache = tile_start_row + my_iters >= tiles_k;
    }

    __device__ __forceinline__ void next_tile() {
        tile_start_col = tile_start_row + 1 == tiles_k ? tile_start_col + 1 : tile_start_col;
        tile_start_row = tile_start_row + 1 == tiles_k ? 0 : tile_start_row + 1;
        --my_iters;
        global_pred = my_iters > 0;
    }

    __device__ __host__ __forceinline__ bool need_save_data() {
        if (global_pred && my_iters == 1) return true;
        if (global_pred && tile_start_row + 1 == tiles_k) return true;
        return false;
    }

    //support for preloading next tile in current tile calculation
    //The point is when all quanted values are dequanted and all a are stored to bsm already
    //Then the registers for scales, zeros, and a_caches are free to use, bsm for scales are already
    //free to use.
    //The main problem we will face is that no more registers can be used to cache tile information
    //when we want to preload next tile data
    bool flag_save_data = false;
    int tile_start_col_cache; //Kept for saving result n calculation

    __device__ __forceinline__ void next_tile_pre() {
        flag_save_data = need_save_data();
        tile_start_col_cache = tile_start_col;
        next_tile();
    }

    __device__ __forceinline__ bool need_save_data_pre() {
        return flag_save_data;
    }

    __device__ __forceinline__ void reset_atomic_idx() {
        atomic_idx = 0;
        last_atomic_cache = tile_start_row + my_iters >= tiles_k;
    }
};

struct ThreadView {
    int tid;
    int wave_idx;
    int wave_tid;
    int slot_idx;
    int slot_tid;

    __device__ __forceinline__ void init() {
        tid = threadIdx.x;
        wave_idx = tid / WAVE;
        wave_tid = tid % WAVE;
        slot_idx = wave_tid / SLOT;
        slot_tid = wave_tid % SLOT;
    }
};

//deqaunt a uint32_t
template<class scalar_t>
__device__ __forceinline__ void dequant_gptq_4bits(const PackType& p, scalar_t (&out)[8], const v2f& scale, const v2f& scale_zero) {
    v2f a0;
    //v2f scale_2f = {scale, scale};
    //v2f scale_zero_2f = {scale_zero, scale_zero};
    int p0 = p & 0x0f0f0f0f;
    CVT_B0TOF32(p0, a0.x);
    CVT_B2TOF32(p0, a0.y);
    a0 = _pk_fma_f32(a0, scale, scale_zero);
    out[0] = float2scalar<scalar_t>(a0.x);
    out[1] = float2scalar<scalar_t>(a0.y);

    CVT_B1TOF32(p0, a0.x);
    CVT_B3TOF32(p0, a0.y);
    a0 = _pk_fma_f32(a0, scale, scale_zero);
    out[4] = float2scalar<scalar_t>(a0.x);
    out[5] = float2scalar<scalar_t>(a0.y);

    p0 = (p >> 4) & 0x0f0f0f0f;
    CVT_B0TOF32(p0, a0.x);
    CVT_B2TOF32(p0, a0.y);
    a0 = _pk_fma_f32(a0, scale, scale_zero);
    out[2] = float2scalar<scalar_t>(a0.x);
    out[3] = float2scalar<scalar_t>(a0.y);

    CVT_B1TOF32(p0, a0.x);
    CVT_B3TOF32(p0, a0.y);
    a0 = _pk_fma_f32(a0, scale, scale_zero);
    out[6] = float2scalar<scalar_t>(a0.x);
    out[7] = float2scalar<scalar_t>(a0.y);
}

template<class scalar_t>
__device__ __forceinline__ void dequant_gptq_8bits(const PackType& p, scalar_t (&out)[4], const v2f& scale, const v2f& scale_zero) {
    v2f a0;
    CVT_B0TOF32(p, a0.x);
    CVT_B1TOF32(p, a0.y);
    a0 = _pk_fma_f32(a0, scale, scale_zero);
    out[0] = float2scalar<scalar_t>(a0.x);
    out[1] = float2scalar<scalar_t>(a0.y);

    CVT_B2TOF32(p, a0.x);
    CVT_B3TOF32(p, a0.y);
    a0 = _pk_fma_f32(a0, scale, scale_zero);
    out[2] = float2scalar<scalar_t>(a0.x);
    out[3] = float2scalar<scalar_t>(a0.y);
}

// decompress zero
__device__ __forceinline__ void decompress_zero_4bits(const PackType& zp, float (&out)[8]) {
    v2f a0;
    int p0 = zp & 0x0f0f0f0f;
    CVT_B0TOF32(p0, a0.x);
    CVT_B2TOF32(p0, a0.y);
    out[0] = -a0.x;
    out[1] = -a0.y;

    CVT_B1TOF32(p0, a0.x);
    CVT_B3TOF32(p0, a0.y);
    out[4] = -a0.x;
    out[5] = -a0.y;

    p0 = (zp >> 4) & 0x0f0f0f0f;
    CVT_B0TOF32(p0, a0.x);
    CVT_B2TOF32(p0, a0.y);
    out[2] = -a0.x;
    out[3] = -a0.y;

    CVT_B1TOF32(p0, a0.x);
    CVT_B3TOF32(p0, a0.y);
    out[6] = -a0.x;
    out[7] = -a0.y;
}

}