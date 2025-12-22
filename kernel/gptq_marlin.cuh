/*
 * Modified by Neural Magic
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */
/*
 * Adapted from  https://github.com/vllm-project/vllm/tree/main/csrc/quantization/gptq_marlin
 */
#include "hgemm_gptq.h"
#include "common.cuh"
#include "utils.cuh"

#define STATIC_ASSERT_SCALAR_TYPE_VALID(scalar_t)               \
  static_assert(std::is_same<scalar_t, half>::value ||          \
                    std::is_same<scalar_t, nv_bfloat16>::value, \
                "only float16 and bfloat16 is supported");

template <typename T>
inline std::string str(T x) {
  return std::to_string(x);
}

namespace mcOpLib {

template <int tileK, int tileM, typename dtype>
__global__ void perm_a(dtype *output, const dtype *input, const uint32_t *idx, int k, int m, int lda) {
    int tid = threadIdx.x;
    int row = blockIdx.x * tileK + tid;
    int col_st = blockIdx.y * tileM;
    if (row < k) {
        int index = idx[row];
        #pragma unroll tileM
        for (int i = 0; i < tileM; ++i) {
            int col = col_st + i;
            if (col < m) {
                output[row + lda * col] = input[index + lda * col];
            }
        }
    }
}

template<typename scalar_t, int BLOCK_STRIDE = 64, int BLOCK_THREADS=256>
__global__ void pack_atomic_cache(float* atomic_cache, scalar_t* output, int m, int n, int k, int size_atomic_cache) {
  int tid = threadIdx.x;
  int bdx = blockIdx.x;
  int bdy = blockIdx.y;
  __shared__ float cache[BLOCK_THREADS*4]; //16k memory
  //Firstly, load all/part value into shared memory
  constexpr int concurrent_lines = BLOCK_THREADS / BLOCK_STRIDE;
  int loading_block_idx = tid / BLOCK_STRIDE;
  int loading_block_lane = tid % BLOCK_STRIDE;
  float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;

  for (int l = 0; l < size_atomic_cache; l += concurrent_lines) {
      int w_offset = (bdx * BLOCK_STRIDE + loading_block_lane) << 2;
      bool pred = loading_block_idx + l < size_atomic_cache && w_offset < n;
      float4 v;
      //Address: stride_batch : bdy * size_atomic_cache * n
      //       : stride_in_line: bdx * BLOCK_STRIDE
      //       : stride_block_lane: loading_block_lane
      ldg_b128_reg_noasync(*((b128VecType*)&v), atomic_cache + bdy * size_atomic_cache * n + (l + loading_block_idx) * n + w_offset, pred, true);
      //save to shared memory, shared memory will only store max_lines * 128
      sum0 += v.x;
      sum1 += v.y;
      sum2 += v.z;
      sum3 += v.w;
  }
  cache[tid*4] = sum0;
  cache[tid*4+1] = sum1;
  cache[tid*4+2] = sum2;
  cache[tid*4+3] = sum3;
  __syncthreads();
  if (tid < BLOCK_STRIDE) {
    for (int i = 1; i < concurrent_lines; i++) {
      sum0 += cache[i * BLOCK_STRIDE * 4 + tid*4];
      sum1 += cache[i * BLOCK_STRIDE * 4 + tid*4 + 1];
      sum2 += cache[i * BLOCK_STRIDE * 4 + tid*4 + 2];
      sum3 += cache[i * BLOCK_STRIDE * 4 + tid*4 + 3];
    }
    int offset = (tid + bdx * BLOCK_STRIDE) << 2;
    bool pred =  offset < n;
    scalar_t sd[4];
    sd[0] = hgemm_marlin_gptq::float2scalar<scalar_t>(sum0);
    sd[1] = hgemm_marlin_gptq::float2scalar<scalar_t>(sum1);
    sd[2] = hgemm_marlin_gptq::float2scalar<scalar_t>(sum2);
    sd[3] = hgemm_marlin_gptq::float2scalar<scalar_t>(sum3);
    stg_b64_async(*(long *)(&sd), output + bdy * n + offset, pred);
  }
}

template<typename scalar_t, int SIZE_ATOMIC_CACHE, int BLOCK_STRIDE = 64, int BLOCK_THREADS=256>
__global__ void pack_atomic_cache_const(float* atomic_cache, scalar_t* output, int m, int n, int k) {
  int tid = threadIdx.x;
  int bdx = blockIdx.x;
  int bdy = blockIdx.y;
  __shared__ float cache[BLOCK_THREADS*4]; //16k memory
  //Firstly, load all/part value into shared memory
  constexpr int concurrent_lines = BLOCK_THREADS / BLOCK_STRIDE;
  int loading_block_idx = tid / BLOCK_STRIDE;
  int loading_block_lane = tid % BLOCK_STRIDE;
  float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
  #pragma unroll
  for (int l = 0; l < SIZE_ATOMIC_CACHE; l += concurrent_lines) {
      int w_offset = bdx * BLOCK_STRIDE * 4 + loading_block_lane * 4 ;
      bool pred = loading_block_idx + l < SIZE_ATOMIC_CACHE && w_offset < n;
      float4 v;
      //Address: stride_batch : bdy * size_atomic_cache * n
      //       : stride_in_line: bdx * BLOCK_STRIDE
      //       : stride_block_lane: loading_block_lane
      ldg_b128_reg_noasync(*((b128VecType*)&v), atomic_cache + bdy * SIZE_ATOMIC_CACHE * n + (l + loading_block_idx) * n + w_offset, pred, true);
      //save to shared memory, shared memory will only store max_lines * 128
      sum0 += v.x;
      sum1 += v.y;
      sum2 += v.z;
      sum3 += v.w;
      // if (bdy == 0 && bdx * BLOCK_STRIDE * 4 + loading_block_lane * 4 == 512) {
      //   printf("loading atomic cache position m0 n1968, v.x = %f, l = %d, sum0 = %f\n", v.x, l + loading_block_idx, sum0);
      // }
  }
  cache[tid*4] = sum0;
  cache[tid*4+1] = sum1;
  cache[tid*4+2] = sum2;
  cache[tid*4+3] = sum3;
  __syncthreads();
  if (tid < BLOCK_STRIDE) {
    for (int i = 1; i < concurrent_lines; i++) {
      sum0 += cache[i * BLOCK_STRIDE * 4 + tid*4];
      sum1 += cache[i * BLOCK_STRIDE * 4 + tid*4 + 1];
      sum2 += cache[i * BLOCK_STRIDE * 4 + tid*4 + 2];
      sum3 += cache[i * BLOCK_STRIDE * 4 + tid*4 + 3];
    }
    // if (bdy == 0 && bdx * BLOCK_STRIDE * 4 + tid * 4 == 512) {
    //   printf("loading atomic cache position m0 n1968, sum0 = %f\n", sum0);
    // }
    int offset = (tid + bdx * BLOCK_STRIDE) << 2;
    bool pred =  offset < n;
    scalar_t sd[4];
    sd[0] = hgemm_marlin_gptq::float2scalar<scalar_t>(sum0);
    sd[1] = hgemm_marlin_gptq::float2scalar<scalar_t>(sum1);
    sd[2] = hgemm_marlin_gptq::float2scalar<scalar_t>(sum2);
    sd[3] = hgemm_marlin_gptq::float2scalar<scalar_t>(sum3);
    stg_b64_async(*(long *)(&sd), output + bdy * n + offset, pred);
  }
}

template <typename input_type, const sglang::ScalarTypeId w_type_id, typename output_type, typename quant_packed_type, bool use_atomic_cache=true>
bool launch_gemm_gptq(int m,
                      int n,
                      int k,
                      int quant_group,
                      const input_type *dA,
                      int lda,
                      const quant_packed_type *dB,
                      int ldb,
                      output_type *dC,
                      float *dC_temp,
                      const int *size_m_ptr,
                      int ldc,
                      quant_packed_type *d_zeros,
                      input_type *d_scales,
                      int chunks,
                      int size_atomic_cache,
                      cudaStream_t stream = nullptr) {
    using namespace hgemm_marlin_gptq;
    if(n % 16 != 0) {
        printf("n %% 16 != 0, n = %d\n", n);
        return false;
    }
    if(k % 32 != 0) {
        printf("k %% 32 != 0, k = %d\n", k);
        return false;
    }
    if (dC_temp == nullptr) {
      printf("We need C_temp for fp32 atomic\n");
      return false;
    }
    //const sglang::ScalarTypeId w_type_id = sglang::kU4B8.id();
    const int THREADS = 256;
    // const int BLOCKS_M = (m + 15) / 16;
    // const int BLOCKS_N = (n + 15) / 16;
    // const int BLOCKS_K = (k + 31) / 32;
    int BLOCKS_M = div_ceil(m, SLICE_M);
    if(BLOCKS_M >= MAX_BLOCKS_M && BLOCKS_M % MAX_BLOCKS_M != 0) {
        printf("Error: input m is error, m = %d, blocks_m = %d\n", m, BLOCKS_M);
        return false;
    }
    if (BLOCKS_M > MAX_BLOCKS_M) BLOCKS_M = MAX_BLOCKS_M;
    int BLOCKS_N = 8;
    //It is better let TILE_K = quant_group
    //But if quant_group is too large, a quant_group can be divided into two parts
    int BLOCKS_K = quant_group / SLICE_K;
    if (quant_group > 128) BLOCKS_K = 128 / SLICE_K;
    if (BLOCKS_M == 1 || BLOCKS_M == 2) {
        BLOCKS_N = 16;
    }
    const bool HAS_ACT_ORDER = false;
    const bool HAS_ZP = (w_type_id == sglang::kU4.id()) || (w_type_id == sglang::kU8.id());
    int *g_idx = nullptr;
    bool HAS_NK_PRED = true;
    // Real size_m needed is stored in size_m_ptr, we have no information about it, so
    // HAS_M_PRED should always be true
    bool HAS_M_PRED = true;
    constexpr bool fp32_atomic = true;
    //int TILE_N = BLOCKS_N * SLICE_N;
    //int TILE_K = BLOCKS_K * SLICE_K;
    //int TILE_M = BLOCKS_M * SLICE_M;
    if (n % TILE_N == 0 && k % TILE_K == 0) {
        HAS_NK_PRED = false;
    }
    if (m % TILE_M == 0 && size_m_ptr == nullptr) {
        HAS_M_PRED = false;
    }

#define LAUNCH_GPTQ(threads, bm, bn, bk, has_act_order, has_zp, has_nk_pred, has_m_pred) \
    else if (THREADS == threads && BLOCKS_M == bm && BLOCKS_N == bn \
        && BLOCKS_K == bk  && HAS_ACT_ORDER == has_act_order \
        && HAS_ZP == has_zp \
        && HAS_M_PRED == has_m_pred && HAS_NK_PRED == has_nk_pred) { \
            launch_gemm_gptq_kernel<input_type, w_type_id, \
                    threads, bm, bn, bk, has_act_order, has_zp, has_m_pred, has_nk_pred, fp32_atomic, use_atomic_cache>( \
                    (const PackTypeInt4*)dA, \
                    (const PackTypeInt4*)dB, \
                    (PackTypeInt4*)dC, \
                    (PackTypeInt4*)dC_temp, \
                    (const int*)size_m_ptr, \
                    (const PackTypeInt4*)d_scales, \
                    (const PackTypeInt4*)d_zeros, \
                    nullptr, m, n, k, quant_group, chunks, size_atomic_cache, stream); \
    }

#define LAUNCH_GPTQ_K(bk, has_act_order, has_zp, has_nk_pred, has_m_pred) \
    LAUNCH_GPTQ(256, 1, 16, bk, has_act_order, has_zp, has_nk_pred, has_m_pred) \
    LAUNCH_GPTQ(256, 2, 16, bk, has_act_order, has_zp, has_nk_pred, has_m_pred) \
    LAUNCH_GPTQ(256, 3, 8, bk, has_act_order, has_zp, has_nk_pred, has_m_pred) \
    LAUNCH_GPTQ(256, 4, 8, bk, has_act_order, has_zp, has_nk_pred, has_m_pred)

#define LAUNCH_GPTQ_ZP(has_zp, has_nk_pred, has_m_pred) \
    LAUNCH_GPTQ_K(1, false, has_zp, has_nk_pred, has_m_pred) \
    LAUNCH_GPTQ_K(2, false, has_zp, has_nk_pred, has_m_pred) \
    LAUNCH_GPTQ_K(4, false, has_zp, has_nk_pred, has_m_pred) \
    LAUNCH_GPTQ_K(8, false, has_zp, has_nk_pred, has_m_pred)

#define LAUNCH_GPTQ_PRED(has_nk_pred, has_m_pred) \
    LAUNCH_GPTQ_ZP(false, has_nk_pred, has_m_pred)
    // LAUNCH_GPTQ_ZP(true, has_nk_pred, has_m_pred)

    if (false) {

    }
    LAUNCH_GPTQ_PRED(true, true)
    LAUNCH_GPTQ_PRED(true, false)
    LAUNCH_GPTQ_PRED(false, true)
    LAUNCH_GPTQ_PRED(false, false)
    else {
        printf("BLOCKS_M=%d, BLOCKS_N=%d, BLOCKS_k=%d, THREADS=%d, HAS_ACT_ORDER=%d, HAS_ZP=%d, quant_group=%d, HAS_M_PRED=%d, HAS_NK_PRED=%d is not supported\n",
        BLOCKS_M, BLOCKS_N, BLOCKS_K, THREADS, HAS_ACT_ORDER, HAS_ZP, quant_group, HAS_M_PRED, HAS_NK_PRED);
        return false;
    }

    return true;
}


template <typename input_type, const sglang::ScalarTypeId w_type_id, typename output_type, typename quant_packed_type, bool use_atomic_cache = true>
bool launch_gemm(int m,
                int n,
                int k,
                int quant_group,
                int size_atomic_cache,
                const input_type *dA,
                int lda,
                const quant_packed_type *dB,
                int ldb,
                output_type *dC,
                float* dC_temp,
                const int* size_m_ptr,
                int ldc,
                quant_packed_type *d_zeros,
                input_type *d_scales,
                uint32_t* g_idx,
                bool is_gptq = true,
                cudaStream_t stream = nullptr) {
    using namespace hgemm_marlin_gptq;
    //constexpr int max_blocks_m = 4;
    int total_m_blocks = div_ceil(m, SLICE_M);
    int chunks = total_m_blocks / MAX_BLOCKS_M;
    int rest_blocks_m = total_m_blocks % MAX_BLOCKS_M;
    const input_type *dA_actual = dA;
    bool ret = true;
    if (chunks > 0) {
        int real_m = m > chunks * MAX_BLOCKS_M * SLICE_M ? chunks * MAX_BLOCKS_M * SLICE_M : m;
        if (is_gptq) {
            ret = launch_gemm_gptq<input_type, w_type_id, output_type, quant_packed_type, use_atomic_cache>(
              real_m, n, k, quant_group,
              dA_actual, lda, dB, ldb, dC, dC_temp, size_m_ptr, ldc,
              d_zeros, d_scales, chunks, size_atomic_cache, stream);
        }
    }
    // In order to fit rules of cuda graph, only one kernel should be executed here
    // If code runs here, something must be wrong!
    if (rest_blocks_m > 0) {
        int m_offset = chunks * MAX_BLOCKS_M * SLICE_M;
        if (is_gptq) {
            ret = ret && launch_gemm_gptq<input_type, w_type_id, output_type, quant_packed_type, use_atomic_cache>(
              m - m_offset, n, k, quant_group,
              dA_actual + lda * m_offset, lda, dB, ldb, dC + ldc * m_offset, dC_temp + ldc * m_offset * size_atomic_cache, size_m_ptr, ldc,
              d_zeros, d_scales, 1, size_atomic_cache, stream);
        }
    }
    //Do repack
    int block = 256;
    dim3 grid = dim3(div_ceil(n, 256), m, 1);
#define LAUNCH_GPTQ_REPACK(SIZE_ATOMIC_CACHE) \
    else if (size_atomic_cache == SIZE_ATOMIC_CACHE) { \
        pack_atomic_cache_const<output_type, SIZE_ATOMIC_CACHE><<<grid, block, 0, stream>>>(dC_temp, dC, m, n, k); \
    }

    if (size_atomic_cache > 16) {
        pack_atomic_cache<output_type><<<grid, block, 0, stream>>>(dC_temp, dC, m, n, k, size_atomic_cache);
    }
    LAUNCH_GPTQ_REPACK(1)
    LAUNCH_GPTQ_REPACK(2)
    LAUNCH_GPTQ_REPACK(3)
    LAUNCH_GPTQ_REPACK(4)
    LAUNCH_GPTQ_REPACK(5)
    LAUNCH_GPTQ_REPACK(6)
    LAUNCH_GPTQ_REPACK(7)
    LAUNCH_GPTQ_REPACK(8)
    LAUNCH_GPTQ_REPACK(9)
    LAUNCH_GPTQ_REPACK(10)
    LAUNCH_GPTQ_REPACK(11)
    LAUNCH_GPTQ_REPACK(12)
    LAUNCH_GPTQ_REPACK(13)
    LAUNCH_GPTQ_REPACK(14)
    LAUNCH_GPTQ_REPACK(15)
    LAUNCH_GPTQ_REPACK(16)
    return ret;
}

} //namespace mcoplib
