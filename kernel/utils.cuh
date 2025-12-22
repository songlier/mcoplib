#pragma once

/**
 * Quantization utilities including:
 *   Adjusted maximum values for qtypes.
 *   Minimum scaling factors for qtypes.
 */

#include <cmath>
#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <iostream>
#include <cstdint>

#include <torch/all.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>


#define MAYBE_HOST_DEVICE C10_HOST_DEVICE

using b32VecType = uint32_t;
using b64VecType = __NATIVE_VECTOR__(2, uint32_t);
using b128VecType = __NATIVE_VECTOR__(4, uint32_t);
using b128VecType_i = __NATIVE_VECTOR__(4, int32_t);
using Float4VecType = __NATIVE_VECTOR__(4, float);


using PackTypeInt4 = b128VecType;
using PackTypeInt2 = b64VecType;
using PackType = uint32_t;

typedef enum
{
    OP_N = 0,
    OP_T = 1
} Operation_t;

#define cast_half(ptr) reinterpret_cast<__half *>(ptr)
#define cast_b16(ptr) reinterpret_cast<uint16_t *>(ptr)
#define cast_b32(ptr) reinterpret_cast<uint32_t *>(ptr)
#define cast_b64(ptr) reinterpret_cast<b64VecType *>(ptr)
#define cast_u64(ptr) reinterpret_cast<uint64_t *>(ptr)
#define cast_b128(ptr) reinterpret_cast<b128VecType *>(ptr)
#define cast_b128_i(ptr) reinterpret_cast<b128VecType_i *>(ptr)

#ifdef __MACA_ARCH__
    #define __builtin_rcpf(x) __builtin_mxc_rcpf(x)

    #define __builtin_mbcnt_lo(mask, initial_value) __builtin_mxc_mbcnt_lo(mask, initial_value)

    #define CVT_B0TOF32(q, out) out = __builtin_mxc_b0_cast_to_f32(q);
    #define CVT_B1TOF32(q, out) out = __builtin_mxc_b1_cast_to_f32(q);
    #define CVT_B2TOF32(q, out) out = __builtin_mxc_b2_cast_to_f32(q);
    #define CVT_B3TOF32(q, out) out = __builtin_mxc_b3_cast_to_f32(q);
    #define _pk_fma_f32 __builtin_mxc_pk_fma_f32
    #define FENCE__ asm volatile(";")
    #define arrive_gvmcnt(num) __builtin_mxc_arrive(64 + num)
    #define arrive_bsmcnt(num) __builtin_mxc_arrive(4096 + 128 * num)
    #define arrive_gvm_bsmcnt(gvm, bsm) __builtin_mxc_arrive(4096 | (128 * bsm) | 64 | gvm)
    #define barrier __builtin_mxc_barrier_inst
    #define barrier_all __builtin_mxc_barrier_ex(0)
    #define barrier_bsm __builtin_mxc_barrier_ex(1)
    #define barrier_inst __builtin_mxc_barrier_ex(2)

    #define ldg_b32_reg_noasync(dst, base, pred, ret0_en)                                                \
        dst = __builtin_mxc_ldg_b32_predicator(cast_b32(base), 0, ret0_en, true, false, false, pred, 1, \
                                            MACA_ICMP_EQ)[0];
    #define ldg_b64_reg_noasync(dst, base, pred, ret0_en)                                                \
        dst = __builtin_mxc_ldg_b64_predicator(cast_b64(base), 0, ret0_en, true, false, false, pred, 1, \
                                            MACA_ICMP_EQ);
    #define ldg_b128_reg_noasync(dst, base, pred, ret0_en)                                                \
        dst = __builtin_mxc_ldg_b128_predicator(cast_b128(base), 0, ret0_en, true, false, false, pred, 1, \
                                            MACA_ICMP_EQ);

    #define ldg_b32_reg_async(dst, base, pred, ret0_en)                                                \
        dst = __builtin_mxc_ldg_b32_predicator(cast_b32(base), 0, ret0_en, true, false, true, pred, 1, \
                                            MACA_ICMP_EQ)[0];
    #define ldg_b64_reg_async(dst, base, pred, ret0_en)                                                \
        dst = __builtin_mxc_ldg_b64_predicator(cast_b64(base), 0, ret0_en, true, false, true, pred, 1, \
                                            MACA_ICMP_EQ);
    #define ldg_b128_reg_async(dst, base, pred, ret0_en)                                                 \
        dst = __builtin_mxc_ldg_b128_predicator(cast_b128(base), 0, ret0_en, true, false, true, pred, 1, \
                                                MACA_ICMP_EQ);
    #define ldg_b64_v4h_reg_async(dst, base, pred, ret0_en)                                                \
        dst = __builtin_mxc_ldg_b64_predicator(cast_b64(base), 0, ret0_en, true, false, true, pred, 1, \
                                            MACA_ICMP_EQ);

    #define ldg_b32_bsm_noasync(saddr, base, pred, ret0_en)                                                    \
        __builtin_mxc_ldg_b32_bsm_predicator(cast_b32(saddr), cast_b32(base), 0, ret0_en, true, false, false, \
                                            pred, 1, MACA_ICMP_EQ);
    #define ldg_b64_bsm_noasync(saddr, base, pred, ret0_en)                                                    \
        __builtin_mxc_ldg_b64_bsm_predicator(cast_b64(saddr), cast_b64(base), 0, ret0_en, true, false, false, \
                                            pred, 1, MACA_ICMP_EQ);
    #define ldg_b128_bsm_noasync(saddr, base, pred, ret0_en)                                                \
        __builtin_mxc_ldg_b128_bsm_predicator(cast_b128(saddr), cast_b128(base), 0, ret0_en, true, false, \
                                            false, pred, 1, MACA_ICMP_EQ);

    #define ldg_b32_bsm_async(saddr, base, pred, ret0_en)                                                    \
        __builtin_mxc_ldg_b32_bsm_predicator(cast_b32(saddr), cast_b32(base), 0, ret0_en, true, false, true, \
                                            pred, 1, MACA_ICMP_EQ);
    #define ldg_b64_bsm_async(saddr, base, pred, ret0_en)                                                    \
        __builtin_mxc_ldg_b64_bsm_predicator(cast_b64(saddr), cast_b64(base), 0, ret0_en, true, false, true, \
                                            pred, 1, MACA_ICMP_EQ);
    #define ldg_b128_bsm_async(saddr, base, pred, ret0_en)                                                \
        __builtin_mxc_ldg_b128_bsm_predicator(cast_b128(saddr), cast_b128(base), 0, ret0_en, true, false, \
                                            true, pred, 1, MACA_ICMP_EQ);

    #define stg_b32_async(data, base, pred)                                                   \
        __builtin_mxc_stg_b32_predicator(cast_b32(base), 0, data, true, false, true, pred, 1, \
                                        MACA_ICMP_EQ);
    #define stg_b64_async(data, base, pred)                                                   \
        __builtin_mxc_stg_b64_predicator(cast_b64(base), 0, data, true, false, true, pred, 1, \
                                        MACA_ICMP_EQ);
    #define stg_b128_async(data, base, pred)                                                    \
        __builtin_mxc_stg_b128_predicator(cast_b128(base), 0, data, true, false, true, pred, 1, \
                                        MACA_ICMP_EQ);

    #define perm_b32(dst, reg1, reg2, selector) dst = __builtin_mxc_byte_perm(reg1, reg2, selector)

    #define mma_16x16x16f16(a_reg, b_reg, c_reg) \
        c_reg = __builtin_mxc_mma_16x16x16f16(a_reg, b_reg, c_reg)

    #define mma_16x16x16bf16(a_reg, b_reg, c_reg) \
        c_reg = __builtin_mxc_mma_16x16x16bf16(a_reg, b_reg, c_reg)
#endif


template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, int8_t>>>
struct quant_type_max {
  static constexpr T val() { return std::numeric_limits<T>::max(); }
};

// Using the default max value from pytorch (240.0 0x7F) will cause accuracy
// issues when running dynamic quantization. Here use 224.0 0x7E for rocm.

template <typename T>
MAYBE_HOST_DEVICE static constexpr T quant_type_max_v =
    quant_type_max<T>::val();

template <typename T,
          typename = std::enable_if_t<std::is_same_v<T, int8_t>>>
struct min_scaling_factor {
  C10_DEVICE C10_ALWAYS_INLINE static float val() {
    return 1.0f / (quant_type_max_v<T> * 512.0f);
  }
};

template <>
struct min_scaling_factor<int8_t> {
  C10_DEVICE C10_ALWAYS_INLINE static float val() {
    return std::numeric_limits<float>::epsilon();
  }
};

template<class scalar_t>
__device__ __forceinline__ void mma_16x16x16(PackTypeInt2& a, PackTypeInt2& b, PackTypeInt4& c) {
    if constexpr (std::is_same_v<scalar_t, half> || std::is_same_v<scalar_t, at::Half>) {
        mma_16x16x16f16(a, b, c);
    }
    else if constexpr (std::is_same_v<scalar_t, nv_bfloat16> || std::is_same_v<scalar_t, at::BFloat16>) {
        mma_16x16x16bf16(a, b, c);
    }
    else {
        if((threadIdx.x == 0) && (blockIdx.x == 0)){
            printf("unsupported scalar type . %s %s %d\n", __FILE__, __func__, __LINE__);
        } 
    }
}

template <typename scalar_t>
__device__ __forceinline__ float scalar2float(scalar_t val) {
    if constexpr (std::is_same_v<scalar_t, half> || std::is_same_v<scalar_t, at::Half>) {
        return __half2float(val);
    }
    else if constexpr (std::is_same_v<scalar_t, nv_bfloat16> || std::is_same_v<scalar_t, at::BFloat16>) {
        return static_cast<float>(val);
    } else { //for double and float
        return static_cast<float>(val);
    }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t float2scalar(float val) {
    if constexpr (std::is_same_v<scalar_t, half> || std::is_same_v<scalar_t, at::Half>) {
        return __float2half(val);
    } else if constexpr (std::is_same_v<scalar_t, nv_bfloat16> || std::is_same_v<scalar_t, at::BFloat16>) {
        return __float2bfloat16_rn(val);
    } else { //for double and float
        return static_cast<scalar_t>(val);
    }
}