#pragma once

#include "utils.h"
#include "all_reduce_kernel.cuh"
#include <cub/cub.cuh>

#define WARP_SIZE 32

#define SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
    __shfl_xor_sync(uint32_t(-1), var, lane_mask, width)

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

/// Aligned array type
template <
    typename T,
    /// Number of elements in the array
    int N,
    /// Alignment requirement in bytes
    int Alignment = sizeof(T) * N
>
class alignas(Alignment) AlignedArray {
public:
    T data[N];
};

template<typename scalar_t>
__device__ __forceinline__ scalar_t get_weight(const int32_t& v) {
    const scalar_t* idx_and_weight = (const scalar_t*)&v;
    return idx_and_weight[0];
}

template<typename scalar_t>
__device__ __forceinline__ scalar_t get_weight(const int64_t& v) {
    const scalar_t* idx_and_weight = (const scalar_t*)&v;
    return idx_and_weight[0];
}

template<typename scalar_t, typename mask_t>
struct WarpSortProcessor {
    static __device__ __forceinline__ void warpSortDescendingOpt(scalar_t (&idx_and_weight)[2], int tid) ;
};

template<typename scalar_t>
struct WarpSortProcessor<scalar_t, uint16_t> {
    static __device__ __forceinline__ void warpSortDescendingOpt(scalar_t (&idx_and_weight)[2], int tid) {
        const uint32_t MASK = 0xffffffff;
        int64_t val = *(int64_t*)idx_and_weight;
        for (int width = 2; width < 16; width <<=1 ) {
            for (int step = width >> 1; step > 0; step >>=1) {
                const bool direction = ((tid & width) == 0);
                int64_t other_temp_val = __shfl_xor_sync(MASK, val, step);
                int other_tid = tid ^ step;

                float current_weight_bits = get_weight<scalar_t>(val);
                float other_weight_bits = get_weight<scalar_t>(other_temp_val);
                int current_index = val >> 32;
                int other_index = other_temp_val >> 32;

                bool weight_gt = other_weight_bits > current_weight_bits;
                bool weight_eq = other_weight_bits == current_weight_bits;
                bool index_lt = other_index < current_index;

                bool other_is_big = weight_gt | (weight_eq & index_lt);
                bool swap = (tid < other_tid) ^ (other_is_big) ^ (direction);

                val = swap ? other_temp_val : val;
            }
        }
        for (int step = 8; step > 0; step >>= 1) {
            int64_t other_temp_val = __shfl_xor_sync(MASK, val, step);
            int other_tid = tid ^ step;

            float current_weight_bits = get_weight<scalar_t>(val);
            float other_weight_bits = get_weight<scalar_t>(other_temp_val);
            int current_index = val >> 32;
            int other_index = other_temp_val >> 32;

            bool weight_gt = other_weight_bits > current_weight_bits;
            bool weight_eq = other_weight_bits == current_weight_bits;
            bool index_lt = other_index < current_index;

            bool other_is_big = weight_gt | (weight_eq & index_lt);
            bool swap = (tid < other_tid) ^ (!other_is_big);
            val = swap ? other_temp_val : val;
        }
        *(int64_t*)idx_and_weight = val;
    }
};

template<typename scalar_t>
struct WarpSortProcessor<scalar_t, uint32_t> {
    static __device__ __forceinline__ void warpSortDescendingOpt(scalar_t (&idx_and_weight)[2], int tid) {
        uint32_t MASK = 0xffffffff;
        int64_t val = *(int64_t*)idx_and_weight;
        for (int width = 2; width <= WARP_SIZE; width <<=1) {
            for (int step = width >> 1; step > 0; step >>=1) {
                const bool is_not_final_phase = (width != WARP_SIZE);
                const uint32_t bitmask = (tid & width);
                const bool direction = is_not_final_phase & (bitmask == 0);
                int64_t other_temp_val = __shfl_xor_sync(MASK, val, step);
                int other_tid = tid ^ step;

                scalar_t current_weight_bits = get_weight<scalar_t>(val);
                scalar_t other_weight_bits = get_weight<scalar_t>(other_temp_val);

                int current_index = val >> 32;
                int other_index = other_temp_val >> 32;

                bool weight_gt = other_weight_bits > current_weight_bits;
                bool weight_eq = other_weight_bits == current_weight_bits;
                bool index_lt = other_index < current_index;
                bool cond = (tid < other_tid) ^ direction;

                bool swap = (cond & (weight_gt | (weight_eq & index_lt))) |
                            (!cond & ((other_weight_bits < current_weight_bits) | (weight_eq & (other_index > current_index))));

                val = swap ? other_temp_val : val;
            }
        }
        *(int64_t*)idx_and_weight = val;
    }
};

template<typename scalar_t>
struct WarpSortProcessor<scalar_t, uint64_t> {
    static __device__ __forceinline__ void warpSortDescendingOpt(scalar_t (&idx_and_weight)[2], int tid) {
        uint64_t MASK=0xffffffffffffffff;
        int64_t val = *(int64_t*)idx_and_weight;
        for (int width = 2; width < 64; width <<=1 ) {
            for (int step = width >> 1; step > 0; step >>=1) {
                const bool direction = ((tid & width) == 0);
                int64_t other_temp_val = __shfl_xor_sync(MASK, val, step);
                int other_tid = tid ^ step;

                float current_weight_bits = get_weight<scalar_t>(val);
                float other_weight_bits = get_weight<scalar_t>(other_temp_val);
                int current_index = val >> 32;
                int other_index = other_temp_val >> 32;

                bool weight_gt = other_weight_bits > current_weight_bits;
                bool weight_eq = other_weight_bits == current_weight_bits;
                bool index_lt = other_index < current_index;

                bool other_is_big = weight_gt | (weight_eq & index_lt);
                bool swap = (tid < other_tid) ^ (other_is_big) ^ (direction);

                val = swap ? other_temp_val : val;
            }
        }
        for (int step = 32; step > 0; step >>= 1) {
            int64_t other_temp_val = __shfl_xor_sync(MASK, val, step);
            int other_tid = tid ^ step;

            float current_weight_bits = get_weight<scalar_t>(val);
            float other_weight_bits = get_weight<scalar_t>(other_temp_val);
            int current_index = val >> 32;
            int other_index = other_temp_val >> 32;

            bool weight_gt = other_weight_bits > current_weight_bits;
            bool weight_eq = other_weight_bits == current_weight_bits;
            bool index_lt = other_index < current_index;

            bool other_is_big = weight_gt | (weight_eq & index_lt);
            bool swap = (tid < other_tid) ^ (!other_is_big);
            val = swap ? other_temp_val : val;
        }
        *(int64_t*)idx_and_weight = val;
    }
};

template<typename scalar_t, typename mask_t>
__device__ __forceinline__ void SortElement(scalar_t (&idx_and_weight)[2], int tid) {
    WarpSortProcessor<scalar_t, mask_t>::warpSortDescendingOpt(idx_and_weight, tid);
}

template <typename T, typename RT, typename AT>
struct Math {
    static inline __device__ bool lt(T lhs, T rhs)
    {
        return lhs < rhs;
    }
    static inline __device__ bool gt(T lhs, T rhs)
    {
        return lhs > rhs;
    }
    static inline __device__ bool eq(T lhs, T rhs)
    {
        return lhs == rhs;
    }
};

template <>
struct Math<half, half, half> {
    static inline __device__ bool lt(half lhs, half rhs)
    {
        return __hlt(lhs, rhs);
    }

    static inline __device__ bool gt(half lhs, half rhs)
    {
        return __hgt(lhs, rhs);
    }
    
    static inline __device__ bool eq(half lhs, half rhs)
    {
        return __heq(lhs, rhs);
    }
};

template <typename T>
__device__ unsigned int convert2u(T value);

template <>
__device__ unsigned int convert2u(__half value)
{
    // must use short, for reverse convert
    unsigned short int x    = __half_as_ushort(value);
    unsigned short int mask = (x & 0x8000) ? 0xffff : 0x8000;
    unsigned int res        = x ^ mask;
    return res;
}
template <>
__device__ unsigned int convert2u(float value)
{
    unsigned int x    = __float_as_uint(value);
    unsigned int mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
    unsigned int res  = x ^ mask;
    return res;
}

template <typename T>
__device__ T convertu2(unsigned int value);

template <>
__device__ __half convertu2(unsigned int value)
{
    unsigned short int sht  = (unsigned short int)value;
    unsigned short int mask = (sht & 0x8000) ? 0x8000 : 0xffff;
    unsigned short int x    = sht ^ mask;
    return __ushort_as_half(x);
}
template <>
__device__ float convertu2(unsigned int value)
{
    unsigned int mask = (value & 0x80000000) ? 0x80000000 : 0xffffffff;
    unsigned int x    = value ^ mask;
    return __uint_as_float(x);
}

__device__ __inline__ void bfi(unsigned int &ret, unsigned int y, unsigned int bit_start, unsigned int num_bits)
{
    y = y << bit_start;
    unsigned int MASK_X = ((1 << num_bits) - 1) << bit_start;
    unsigned int MASK_Y = ~MASK_X;
    ret = (ret & MASK_Y) | (y & MASK_X);
}

__device__ __inline__  unsigned int bfe(unsigned int source, unsigned int bit_start, unsigned int num_bits)
{
    return (static_cast<unsigned int>(source) << (32 - bit_start - num_bits)) >> (32 - num_bits);
}

template <typename T>
__device__ unsigned int find_desired(
    unsigned int *smem,
    int lane,
    const unsigned int mask,
    const unsigned int desired,
    const int inputSliceStride,
    const T *inputSlice,
    const int sliceSize)
{
    if (threadIdx.x == 0) {
        smem[0] = 0;
    }
    __syncthreads();

    for (int off = threadIdx.x; off - lane < sliceSize; off += blockDim.x) {
        bool inRange          = off < sliceSize;
        T value               = inRange ? inputSlice[off * inputSliceStride] : (T)0;
        unsigned int intValue = convert2u<T>(value);
        bool flag             = inRange && ((intValue & mask) == desired);
        if (flag) {
            smem[0] = 1;
            smem[1] = intValue;
        }
        __syncthreads();

        unsigned int isFound = smem[0];
        intValue             = smem[1];

        if (isFound) {
        return intValue;
        }
    }
    return 0;
}

template <typename T>
__device__ T scanInWarp(T value, int lane)
{
    T lanePrefix = value;
    for (int i = 1; i < 64; i <<= 1) {
        value = __shfl_up_sync(0xffffffffffffffff, lanePrefix, i, 64);
        if (lane >= i) {
        lanePrefix += value;
        }
    }
    return lanePrefix;
}

__device__ void prefix_scan(
    int *smem,
    const uint64_t active,
    const int activeWarps,
    const bool flag,
    int &index,
    int &blkTotal)
{
    if (threadIdx.x < blockDim.x / 32 + 2) {
        smem[threadIdx.x] = 0;
    }
    __syncthreads();

    uint64_t ballot   = __ballot_sync(active, flag);
    int lane              = threadIdx.x & 63;
    uint64_t laneMask = ~(0xffffffffffffffff << lane);
    laneMask              = active & laneMask;
    int warpId            = threadIdx.x >> 6;
    unsigned int leader   = __ffsll(active) - 1;
    int total             = __popcll(ballot);
    int prefix            = __popcll(laneMask & ballot);


    if (lane == leader) {
        smem[warpId] = total;
    }
    __syncthreads();

    int warpOff = 0;
    if (threadIdx.x < blockDim.x / 32 + 2) {
        int value         = smem[threadIdx.x];
        int warpPrefix    = scanInWarp<int>(value, lane);
        smem[threadIdx.x] = warpPrefix;
    }
    __syncthreads();

    if (warpId >= 1)
        warpOff = smem[warpId - 1];
    blkTotal = smem[activeWarps - 1];

    if (flag) {
        index = warpOff + prefix;
    }
    // write-after-read dependency
    __syncthreads();
}

template <typename T, bool dir>
__device__ T find_kth_value(
    int* smem,
    int K,
    const int sliceSize,
    const T* __restrict__ inputSlice,
    const int inputSliceStride)
{
    static constexpr int RADIX_SIZE = 16;
    static constexpr int RADIX_BITS = 4;
    static constexpr int RADIX_MASK = RADIX_SIZE - 1;
    int count[RADIX_SIZE];
    // use fixed higher bits to filter data
    unsigned int mask    = 0; // fixed high bit
    unsigned int desired = 0; // current radix bits to fix
    int *radix_hist      = smem;
    unsigned int kthValue;
    for (int pos = 8 * sizeof(int) - RADIX_BITS; pos >= 0; pos -= RADIX_BITS) {
        // reinit radix_hist to 0 every loop
        for (int i = 0; i < RADIX_SIZE; i++) {
            count[i] = 0;
        }
        if (threadIdx.x < RADIX_SIZE) {
            radix_hist[threadIdx.x] = 0;
        }
        __syncthreads();

        const int lane = threadIdx.x & 63;
        for (int off = threadIdx.x; off - lane < sliceSize; off += blockDim.x) {
            bool inRange          = off < sliceSize;
            T value               = inRange ? inputSlice[off * inputSliceStride] : (T)0;
            uint64_t active   = __ballot_sync(0xffffffffffffffff, inRange);
            unsigned int intValue = convert2u<T>(value);

            // filter with desired
            bool inRadix = inRange && ((intValue & mask) == desired);
            int valueRadix = 0;
            valueRadix = bfe(intValue, pos, RADIX_BITS);
    #pragma unroll
            for (int i = 0; i < RADIX_SIZE; i++) {
            bool flag           = inRadix && (valueRadix == i);
            uint64_t ballot = __ballot_sync(active, flag);
            count[i] += __popcll(ballot);
            }
        }
        if ((threadIdx.x & 63) == 0) {
            for (int i = 0; i < RADIX_SIZE; i++) {
            atomicAdd(radix_hist + i, count[i]);
            }
        }
        __syncthreads();

        // all threads in blk are the same
        for (int i = 0; i < RADIX_SIZE; i++) {
        count[i] = radix_hist[i];
        }
        __syncthreads();

        // search K count
        if (dir == 1) { // topK largest
        for (int i = RADIX_SIZE - 1; i >= 0; --i) {
            if (K == count[i] && K == 1) {
            bfi(desired, i, pos, RADIX_BITS);
            bfi(mask, RADIX_MASK, pos, RADIX_BITS);
            kthValue      = find_desired<T>((unsigned int *)smem, threadIdx.x, mask, desired, inputSliceStride, inputSlice, sliceSize);
            T fp_kthValue = convertu2<T>(kthValue);
            return fp_kthValue;
            } else if (K <= count[i]) { // narrow radix unitl K == count[i] == 1
            bfi(desired, i, pos, RADIX_BITS);
            bfi(mask, RADIX_MASK, pos, RADIX_BITS);
            break;
            }
            K -= count[i];
        }
        } else {
        for (int i = 0; i < RADIX_SIZE; ++i) {
            if (K == count[i] && K == 1) {
            bfi(desired, i, pos, RADIX_BITS);
            bfi(mask, RADIX_MASK, pos, RADIX_BITS);
            kthValue      = find_desired<T>((unsigned int *)smem, threadIdx.x, mask, desired, inputSliceStride, inputSlice, sliceSize);
            T fp_kthValue = convertu2<T>(kthValue);
            return fp_kthValue;
            } else if (K <= count[i]) { // narrow radix unitl K == count[i] == 1
            bfi(desired, i, pos, RADIX_BITS);
            bfi(mask, RADIX_MASK, pos, RADIX_BITS);
            break;
            }
            K -= count[i];
        }
        }
    }
    kthValue      = desired;
    T fp_kthValue = convertu2<T>(kthValue);
    return fp_kthValue;
}

__device__ __inline__ int64_t Align(int64_t x, int64_t y) {
    return (x + y - 1) / y * y;
}

int getSortSize(int size) {
    if (size == 1) {
        return 1;
    } else if (size <= 4) {
        return 4;
    } else if (size <= 64) {
        return 64;
    } else if (size <= 128) {
        return 128;
    } else if (size <= 256) {
        return 256;
    } else if (size <= 512) {
        return 512;
    } else if (size <= 1024) {
        return 1024;
    } else {
        return -1;
    }
}

template <typename KEY, typename VALUE, bool largest>
__device__ inline void swap(
    const bool isOdd,
    bool &valid1,
    KEY &value1,
    VALUE &index1,
    bool &valid2,
    KEY &value2,
    VALUE &index2)
{
    bool isLarge = (largest ^ Math<KEY, KEY, KEY>::lt(value1, value2) && valid1) || !valid2;
    bool isEqual = Math<KEY, KEY, KEY>::eq(value1, value2);

    bool indexLarge = (Math<KEY, KEY, KEY>::gt(index1, index2) && valid2) || !valid1;
    bool if_exchange = ((isLarge == isOdd) && !isEqual) || (isEqual && indexLarge);

    if (if_exchange) {
        KEY tmpValue   = value1;
        VALUE tmpIndex = index1;
        bool tmpValid  = valid1;
        value1         = value2;
        index1         = index2;
        valid1         = valid2;
        value2         = tmpValue;
        index2         = tmpIndex;
        valid2         = tmpValid;
    }
}

template <typename KEY, typename VALUE, bool dir, int power2SortSize>
__device__ void bitonicSort(
    KEY *Key,
    VALUE *Value,
    const int sliceSize)
{
    __shared__ KEY smemTopk[power2SortSize];
    __shared__ VALUE smemIndices[power2SortSize];
    __shared__ bool smemValid[power2SortSize];

    KEY *topKSlice      = Key;
    VALUE *indicesSlice = Value;

    int tid       = threadIdx.x;
    int off1      = threadIdx.x;
    int off2      = threadIdx.x + power2SortSize / 2;
    bool inRange1 = off1 < sliceSize;
    bool inRange2 = off2 < sliceSize;
    KEY value1    = inRange1 ? topKSlice[off1] : (KEY)0;
    VALUE index1  = inRange1 ? indicesSlice[off1] : (VALUE)0;
    KEY value2    = inRange2 ? topKSlice[off2] : (KEY)0;
    VALUE index2  = inRange2 ? indicesSlice[off2] : (VALUE)0;

    smemTopk[off1]    = value1;
    smemIndices[off1] = index1;
    smemValid[off1]   = inRange1;
    smemTopk[off2]    = value2;
    smemIndices[off2] = index2;
    smemValid[off2]   = inRange2;
    __syncthreads();

    #pragma unroll
    for (int size = 2; size < power2SortSize; size *= 2) {
        int oddSeg = (tid & (size / 2)) != 0;
    #pragma unroll
        // sort each size
        for (int sub_size = size; sub_size > 1; sub_size /= 2) {
            int stride = sub_size / 2;
            int off    = (tid / stride) * sub_size + (tid & (stride - 1));

            bool inRange1 = smemValid[off];
            KEY value1    = smemTopk[off];
            VALUE index1  = smemIndices[off];
            bool inRange2 = smemValid[off + stride];
            KEY value2    = smemTopk[off + stride];
            VALUE index2  = smemIndices[off + stride];

            swap<KEY, VALUE, dir>(oddSeg,
                                    inRange1,
                                    value1,
                                    index1,
                                    inRange2,
                                    value2,
                                    index2);

            smemTopk[off]             = value1;
            smemIndices[off]          = index1;
            smemValid[off]            = inRange1;
            smemTopk[off + stride]    = value2;
            smemIndices[off + stride] = index2;
            smemValid[off + stride]   = inRange2;

            __syncthreads();
        }
    }

    // sort the whole power2SortSize
    for (int sub_size = power2SortSize; sub_size > 1; sub_size /= 2) {
        int stride = sub_size / 2;
        int off    = (tid / stride) * sub_size + (tid & (stride - 1));

        bool inRange1 = smemValid[off];
        KEY value1    = smemTopk[off];
        VALUE index1  = smemIndices[off];
        bool inRange2 = smemValid[off + stride];
        KEY value2    = smemTopk[off + stride];
        VALUE index2  = smemIndices[off + stride];

        swap<KEY, VALUE, dir>(false,
                                inRange1,
                                value1,
                                index1,
                                inRange2,
                                value2,
                                index2);

        smemTopk[off]             = value1;
        smemIndices[off]          = index1;
        smemValid[off]            = inRange1;
        smemTopk[off + stride]    = value2;
        smemIndices[off + stride] = index2;
        smemValid[off + stride]   = inRange2;

        __syncthreads();
    }

    inRange1 = smemValid[off1];
    value1   = smemTopk[off1];
    index1   = smemIndices[off1];
    inRange2 = smemValid[off2];
    value2   = smemTopk[off2];
    index2   = smemIndices[off2];
    if (inRange1) {
        topKSlice[off1]    = value1;
        indicesSlice[off1] = index1;
    }
    if (inRange2) {
        topKSlice[off2]    = value2;
        indicesSlice[off2] = index2;
    }
    __syncthreads();

    if(tid * 2 < sliceSize) {
        if (tid > 0 && (topKSlice[tid * 2] == topKSlice[tid * 2 - 1]) && (indicesSlice[tid * 2] < indicesSlice[tid * 2 - 1])){
            VALUE tmp;
            tmp = indicesSlice[tid * 2];
            indicesSlice[tid * 2] = indicesSlice[tid * 2 - 1];
            indicesSlice[tid * 2 - 1] = tmp;
        }
    __syncthreads();
        if ((topKSlice[tid * 2] == topKSlice[tid * 2 + 1]) && (indicesSlice[tid * 2] > indicesSlice[tid * 2 + 1])){
            VALUE tmp;
            tmp = indicesSlice[tid * 2];
            indicesSlice[tid * 2] = indicesSlice[tid * 2 + 1];
            indicesSlice[tid * 2 + 1] = tmp;
        }
    }
}

namespace mc_moe_softmax_topk
{

template <typename scalar_t, int TPB>
__launch_bounds__(TPB) __global__
    void moeSoftmax(const scalar_t* input, scalar_t* output, const int num_cols)
{
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    __shared__ float normalizing_factor;
    __shared__ float float_max;

    const int thread_row_offset = blockIdx.x * num_cols;

    cub::Sum sum;
    float threadData(-FLT_MAX);

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        threadData = max(input[idx], threadData);
    }

    const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
    if (threadIdx.x == 0)
    {
        float_max = maxElem;
    }
    __syncthreads();

    threadData = 0;

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        threadData += exp(((input[idx]) - float_max));
    }

    const auto Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

    if (threadIdx.x == 0)
    {
        normalizing_factor = 1.f / Z;
    }
    __syncthreads();

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        const int idx = thread_row_offset + ii;
        const float val = exp(((input[idx]) - float_max)) * normalizing_factor;
        output[idx] = (val);
    }
}

template <typename scalar_t, int TPB>
__launch_bounds__(TPB) __global__ void moeTopK(const scalar_t* inputs_after_softmax, scalar_t* output,
    int* indices, const int num_experts, const int k, const int start_expert, const int end_expert)
{

    using cub_kvp = cub::KeyValuePair<int, scalar_t>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub_kvp thread_kvp;
    cub::ArgMax arg_max;

    const int num_rows = gridDim.x;
    const int block_row = blockIdx.x;
    extern __shared__ float topk_value[];
    float byte_sum = 0;

    const int thread_read_offset = blockIdx.x * num_experts;
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        thread_kvp.key = 0;
        thread_kvp.value = (-1.f); // This is OK because inputs are probabilities

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB)
        {
            const int idx = thread_read_offset + expert;
            inp_kvp.key = expert;
            inp_kvp.value = inputs_after_softmax[idx];

            for (int prior_k = 0; prior_k < k_idx; ++prior_k)
            {
                const int prior_winning_expert = indices[k * block_row + prior_k];

                if (prior_winning_expert == expert)
                {
                    inp_kvp = thread_kvp;
                }
            }

            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        const cub_kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0)
        {
            // Ignore experts the node isn't responsible for with expert parallelism
            const int expert = result_kvp.key;
            const int idx = k * block_row + k_idx;
            // output[idx] = result_kvp.value;
            topk_value[k_idx] = result_kvp.value;
            byte_sum += result_kvp.value;
            indices[idx] = expert;
            assert(indices[idx] >= 0);
            // source_rows[idx] = k_idx * num_rows + block_row;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            const int idx = k * block_row + k_idx;
            output[idx] = topk_value[k_idx] / byte_sum;
        }
    }
}

template <typename scalar_t, int TPB>
__launch_bounds__(TPB) __global__ void moeTopKSoftmax(const scalar_t* inputs_after_softmax, scalar_t* output,
    int* indices, const int num_experts, const int k, const int start_expert, const int end_expert)
{

    using cub_kvp = cub::KeyValuePair<int, scalar_t>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub_kvp thread_kvp;
    cub::ArgMax arg_max;

    const int num_rows = gridDim.x;
    const int block_row = blockIdx.x;

    extern __shared__ float topk_value[];
    float byte_sum = 0;
    float byte_max = -99999.f;

    const int thread_read_offset = blockIdx.x * num_experts;
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        thread_kvp.key = 0;
        thread_kvp.value = (-1.f); // This is OK because inputs are probabilities

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB)
        {
            const int idx = thread_read_offset + expert;
            inp_kvp.key = expert;
            inp_kvp.value = inputs_after_softmax[idx];

            for (int prior_k = 0; prior_k < k_idx; ++prior_k)
            {
                const int prior_winning_expert = indices[k * block_row + prior_k];

                if (prior_winning_expert == expert)
                {
                    inp_kvp = thread_kvp;
                }
            }

            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        const cub_kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0)
        {
            // Ignore experts the node isn't responsible for with expert parallelism
            const int expert = result_kvp.key;

            const int idx = k * block_row + k_idx;
            // output[idx] = result_kvp.value;
            topk_value[k_idx] = result_kvp.value;
            indices[idx] = expert;
            assert(indices[idx] >= 0);
            // source_rows[idx] = k_idx * num_rows + block_row;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            byte_max = max(topk_value[k_idx], byte_max);
        }
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            topk_value[k_idx] = __builtin_expf(topk_value[k_idx] - byte_max);
            byte_sum += topk_value[k_idx];
        }
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            const int idx = k * block_row + k_idx;
            output[idx] = topk_value[k_idx] / byte_sum;
        }
    }
}

template <typename scalar_t, int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
    void topkGatingSoftmax(const scalar_t* input, const bool* finished, scalar_t* output, const int num_rows, int* indices,
        const int k, const int start_expert, const int end_expert)
{
    // We begin by enforcing compile time assertions and setting up compile time constants.
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    // Number of bytes each thread pulls in per load
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(scalar_t);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    // Restrictions based on previous section.
    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

    // We have NUM_EXPERTS elements per row. We specialize for small #experts
    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    // Restrictions for previous section.
    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    // ===================== From this point, we finally start computing run-time variables. ========================

    // Compute CTA and warp rows. We pack multiple rows into a single warp, and a block contains WARPS_PER_CTA warps.
    // This, each block processes a chunk of rows. We start by computing the start row for each block.
    const int cta_base_row = blockIdx.x * ROWS_PER_CTA;

    // Now, using the base row per thread block, we compute the base row per warp.
    const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;

    // The threads in a warp are split into sub-groups that will work on a row.
    // We compute row offset for each thread sub-group
    const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    const int thread_row = warp_base_row + thread_row_in_warp;

    extern __shared__ float topk_value[];
    float byte_sum = 0;

    // Threads with indices out of bounds should early exit here.
    if (thread_row >= num_rows)
    {
        return;
    }
    // We finally start setting up the read pointers for each thread. First, each thread jumps to the start of the
    // row it will read.
    const scalar_t* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

    // Now, we compute the group each thread belong to in order to determine the first column to start loads.
    const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    const scalar_t* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    // Determine the pointer type to use to read in the data depending on the BYTES_PER_LDG template param. In theory,
    // this can support all powers of 2 up to 16.
    // NOTE(woosuk): The original implementation uses CUTLASS aligned array here.
    // We defined our own aligned array and use it here to avoid the dependency on CUTLASS.
    using AccessType = AlignedArray<scalar_t, ELTS_PER_LDG>;

    // Finally, we pull in the data from global mem
    scalar_t row_chunk[VPT];
    AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk);
    const AccessType* vec_thread_read_ptr = reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii)
    {
        row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    // First, we perform a max reduce within the thread. We can do the max in fp16 safely (I think) and just
    // convert to float afterwards for the exp + sum reduction.
    float thread_max = (row_chunk[0]);
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii)
    {
        thread_max = max(thread_max, (row_chunk[ii]));
    }

// Now, we find the max within the thread group and distribute among the threads. We use a butterfly reduce.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        thread_max = max(thread_max, SHFL_XOR_SYNC_WIDTH(thread_max, mask, THREADS_PER_ROW));
    }

    // From this point, thread max in all the threads have the max within the row.
    // Now, we subtract the max from each element in the thread and take the exp. We also compute the thread local sum.
    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        float tmp = __builtin_expf((row_chunk[ii]) - thread_max);
        row_chunk[ii] = (tmp);
        row_sum += tmp;
    }

// Now, we perform the sum reduce within each thread group. Similar to the max reduce, we use a bufferfly pattern.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        row_sum += SHFL_XOR_SYNC_WIDTH(row_sum, mask, THREADS_PER_ROW);
    }

    // From this point, all threads have the max and the sum for their rows in the thread_max and thread_sum variables
    // respectively. Finally, we can scale the rows for the softmax. Technically, for top-k gating we don't need to
    // compute the entire softmax row. We can likely look at the maxes and only compute for the top-k values in the row.
    // However, this kernel will likely not be a bottle neck and it seems better to closer match torch and find the
    // argmax after computing the softmax.
    const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = ((row_chunk[ii]) * reciprocal_row_sum);
    }

    // Now, softmax_res contains the softmax of the row chunk. Now, I want to find the topk elements in each row, along
    // with the max index.
    int start_col = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        // First, each thread does the local argmax
        scalar_t max_val = row_chunk[0];
        int expert = start_col;
#pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG)
        {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii)
            {
                scalar_t val = row_chunk[ldg * ELTS_PER_LDG + ii];

                // No check on the experts here since columns with the smallest index are processed first and only
                // updated if > (not >=)
                if (val > max_val)
                {
                    max_val = val;
                    expert = col + ii;
                }
            }
        }

// Now, we perform the argmax reduce. We use the butterfly pattern so threads reach consensus about the max.
// This will be useful for K > 1 so that the threads can agree on "who" had the max value. That thread can
// then blank out their max with -inf and the warp can run more iterations...
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
        {
            scalar_t other_max = SHFL_XOR_SYNC_WIDTH(max_val, mask, THREADS_PER_ROW);
            int other_expert = SHFL_XOR_SYNC_WIDTH(expert, mask, THREADS_PER_ROW);

            // We want lower indices to "win" in every thread so we break ties this way
            if (other_max > max_val || (other_max == max_val && other_expert < expert))
            {
                max_val = other_max;
                expert = other_expert;
            }
        }

        // Write the max for this k iteration to global memory.
        if (thread_group_idx == 0)
        {
            // The lead thread from each sub-group will write out the final results to global memory. (This will be a
            // single) thread per row of the input/output matrices.
            const int idx = k * thread_row + k_idx;
            // output[idx] = max_val;
            topk_value[threadIdx.y * k + k_idx] = max_val;
            byte_sum += max_val;
            indices[idx] = expert;
        }

        // Finally, we clear the value in the thread with the current max if there is another iteration to run.
        if (k_idx + 1 < k)
        {
            const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
            const int thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

            // Only the thread in the group which produced the max will reset the "winning" value to -inf.
            if (thread_group_idx == thread_to_clear_in_group)
            {
                const int offset_for_expert = expert % ELTS_PER_LDG;
                // Safe to set to any negative value since row_chunk values must be between 0 and 1.
                row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = (-999);
            }
        }
    }

    __syncthreads();
    if (thread_group_idx == 0) {
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            const int idx = k * thread_row + k_idx;
            output[idx] = topk_value[threadIdx.y * k + k_idx] / byte_sum;
        }
    }
}

template <typename scalar_t, typename bitwise_t, int NUM_EXPERTS = 128, int WARPS_PER_CTA = 8, int TOPK = 8, int WAVE_SIZE = 64, int WAVES_PER_ROW = 2>
__launch_bounds__(WARPS_PER_CTA * WAVE_SIZE) __global__
    void topkGatingSoftmaxDecodeOpttt(const scalar_t* input, scalar_t* output, const int num_rows, int* indices)
{
    const int thread_row = blockIdx.x * WARPS_PER_CTA / WAVES_PER_ROW + threadIdx.y;
    const int wave_id_in_row = threadIdx.x / WAVE_SIZE;
    constexpr int bit_offset = sizeof(bitwise_t) * 4;
    if (thread_row >= num_rows) {
        return;
    }
    __shared__ float s_max_v[WARPS_PER_CTA];
    __shared__ float s_sum_v[WARPS_PER_CTA];

    scalar_t row_chunk = input[thread_row * NUM_EXPERTS + threadIdx.x];
    float max_val = (row_chunk);
    float sum_val = __builtin_expf(max_val);

    max_val = fmaxf(max_val, __shfl_down_sync_16(0xffffffffffffffff, max_val, 1));
    max_val = fmaxf(max_val, __shfl_down_sync_16(0xffffffffffffffff, max_val, 2));
    max_val = fmaxf(max_val, __shfl_down_sync_16(0xffffffffffffffff, max_val, 4));
    max_val = fmaxf(max_val, __shfl_down_sync_16(0xffffffffffffffff, max_val, 8));
    max_val = fmaxf(max_val, __shfl_down_sync(0xffffffffffffffff, max_val, 16, WAVE_SIZE));
    max_val = fmaxf(max_val, __shfl_down_sync(0xffffffffffffffff, max_val, 32, WAVE_SIZE));

    sum_val += __shfl_down_sync_16(0xffffffffffffffff, sum_val, 1);
    sum_val += __shfl_down_sync_16(0xffffffffffffffff, sum_val, 2);
    sum_val += __shfl_down_sync_16(0xffffffffffffffff, sum_val, 4);
    sum_val += __shfl_down_sync_16(0xffffffffffffffff, sum_val, 8);
    sum_val += __shfl_down_sync(0xffffffffffffffff, sum_val, 16, WAVE_SIZE);
    sum_val += __shfl_down_sync(0xffffffffffffffff, sum_val, 32, WAVE_SIZE);

    bitwise_t tid = (bitwise_t)threadIdx.x;

    if(threadIdx.x % WAVE_SIZE == 0) {
        s_max_v[threadIdx.y * WAVES_PER_ROW + wave_id_in_row] = max_val;
        s_sum_v[threadIdx.y * WAVES_PER_ROW + wave_id_in_row] = sum_val;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        s_max_v[threadIdx.y * WAVES_PER_ROW] = max(s_max_v[threadIdx.y * WAVES_PER_ROW], s_max_v[threadIdx.y * WAVES_PER_ROW + 1]);
        s_sum_v[threadIdx.y * WAVES_PER_ROW] = s_sum_v[threadIdx.y * WAVES_PER_ROW] + s_sum_v[threadIdx.y * WAVES_PER_ROW + 1];
    }
    __syncthreads();
    
    float rep = 1.0f / (s_sum_v[threadIdx.y * WAVES_PER_ROW] * __builtin_expf(-s_max_v[threadIdx.y * WAVES_PER_ROW]));
    row_chunk = (__builtin_expf((row_chunk) - s_max_v[threadIdx.y * WAVES_PER_ROW]) * rep);

    __shared__ scalar_t shared_experts[WARPS_PER_CTA * TOPK * NUM_EXPERTS / (WAVE_SIZE * WAVES_PER_ROW)][2];
    scalar_t idx_and_weight[2];
    idx_and_weight[0] = row_chunk;
    idx_and_weight[1] = (0.0f);
    *((bitwise_t*)idx_and_weight) |= (tid << bit_offset);
    SortElement<scalar_t, uint64_t>(idx_and_weight, threadIdx.x);
    if (threadIdx.x % WAVE_SIZE < TOPK) {
        *(((bitwise_t*)(shared_experts)) + threadIdx.y * (TOPK * NUM_EXPERTS / WAVE_SIZE) + (threadIdx.x % WAVE_SIZE) + wave_id_in_row * TOPK) = *(bitwise_t*)idx_and_weight;
    }
    __syncthreads();

    *(bitwise_t*)idx_and_weight = threadIdx.x < TOPK * WAVES_PER_ROW ? *((bitwise_t*)shared_experts + threadIdx.y * (TOPK * NUM_EXPERTS / WAVE_SIZE) + threadIdx.x) : 0;
    SortElement<scalar_t, uint16_t>(idx_and_weight, threadIdx.x);

    if (threadIdx.x < TOPK) {
        bitwise_t res = *(bitwise_t*)idx_and_weight;
        int expert_id_ordered = (res >> bit_offset);
        scalar_t max_val_ordered = get_weight<scalar_t>(res);
        scalar_t sum_ = WarpAllReduceSum<scalar_t>(max_val_ordered, TOPK);
        output[thread_row * TOPK + threadIdx.x] = max_val_ordered / sum_;
        indices[thread_row * TOPK + threadIdx.x] = expert_id_ordered;
    }
}

// template <typename scalar_t, int TPB, int BLOCK_SIZE>
// __launch_bounds__(TPB) __global__ void moeTopKSoftmaxVectorlized(const scalar_t* input, scalar_t* output,
//     int* indices, const int num_experts, const int k, const int start_expert, const int end_expert)
// {
//     typedef cub::BlockRadixSort<scalar_t, BLOCK_SIZE, TPB> BlockRadixSort;
//     typedef cub::BlockLoad<scalar_t, BLOCK_SIZE, TPB, cub::BLOCK_LOAD_TRANSPOSE> BlockLoad;
//     typedef cub::BlockStore<scalar_t, BLOCK_SIZE, TPB, cub::BLOCK_STORE_TRANSPOSE> BlockStore;

//     __shared__ union {
//         typename BlockRadixSort::TempStorage sort;
//         typename BlockLoad::TempStorage load;
//         typename BlockStore::TempStorage store;
//     } temp_storage;

//     float row_chunk[TPB];
//     int block_offset = blockIdx.x * TPB * BLOCK_SIZE;
//     BlockLoad(temp_storage.load).Load(input + block_offset, row_chunk);
//     __syncthreads();

//     BlockRadixSort(temp_storage.sort).Sort(row_chunk);
//     __syncthreads();

//     BlockStore(temp_storage.store).Store(output + block_offset, row_chunk);
// }

template <typename scalar_t, typename index_t, int64_t BLOCK_SIZE, int sortBlockSize = 512>
__global__ void selectTopKSoftmax(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ topK,
    index_t* __restrict__ indices,
    const int K,
    const int64_t sliceSize,
    const int inputStride,
    const int topKStride, 
    const int indicesStride
) {
    extern __shared__ scalar_t topk_value[];
    scalar_t* key_smem = &topk_value[0];
    index_t* val_smem = reinterpret_cast<index_t*>(key_smem + K);

    const int inputSliceStride = 1;
    const int topKSliceStride = 1;
    const int indicesSliceStride = 1;

    const scalar_t* inputSlice = input + blockIdx.x * inputStride;
    scalar_t* topKSliceOut = topK + blockIdx.x * K;
    index_t* indicesSliceOut = indices + blockIdx.x * K;
    scalar_t* topKSlice = key_smem;
    index_t* indicesSlice = val_smem;

    __shared__ int radix_hist[2 + BLOCK_SIZE / 32];
    int *smem = radix_hist;

    scalar_t fp_kthValue = find_kth_value<scalar_t, true>(smem, K, sliceSize, inputSlice, inputSliceStride);

    int writeStart  = 0;
    int activeWarps = 0;
    int64_t tmpSize     = sliceSize;
    for (int64_t off = threadIdx.x; off < Align(sliceSize, BLOCK_SIZE); off += BLOCK_SIZE) {
        int curSize         = tmpSize >= BLOCK_SIZE ? BLOCK_SIZE : tmpSize;
        activeWarps         = (curSize + 63) >> 6;
        bool inRange        = off < sliceSize;
        scalar_t value             = inRange ? inputSlice[off * inputSliceStride] : (scalar_t)0;
        uint64_t active = __ballot_sync(0xffffffffffffffff, inRange);

        bool flag;
        flag = inRange && Math<scalar_t, scalar_t, scalar_t>::gt(value, fp_kthValue);
        int index, blkTotal;
        prefix_scan(smem, active, activeWarps, flag, index, blkTotal);

        if (flag) {
            int topKOffset            = writeStart + index;
            int indexOffset           = writeStart + index;
            topKSlice[topKOffset]     = value;
            indicesSlice[indexOffset] = off;
        }
        writeStart += blkTotal;
        tmpSize -= BLOCK_SIZE;
    }
    __syncthreads();

    int topKRemaining = K - writeStart;
    tmpSize           = sliceSize;
    for (int64_t off = threadIdx.x; off < Align(sliceSize, BLOCK_SIZE); off += BLOCK_SIZE) {
        int curSize         = tmpSize >= BLOCK_SIZE ? BLOCK_SIZE : tmpSize;
        activeWarps         = (curSize + 63) >> 6;
        bool inRange        = off < sliceSize;
        scalar_t value             = inRange ? inputSlice[off * inputSliceStride] : (scalar_t)0;
        uint64_t active = __ballot_sync(0xffffffffffffffff, inRange);

        bool flag;
        flag = inRange && Math<scalar_t, scalar_t, scalar_t>::eq(value, fp_kthValue);
        int index, blkTotal;
        prefix_scan(smem, active, activeWarps, flag, index, blkTotal);

        if (flag) {
            int outputIndex = writeStart + index;
            if (outputIndex < K) {
                int topKOffset            = outputIndex;
                int indexOffset           = outputIndex;
                topKSlice[topKOffset]     = value;
                indicesSlice[indexOffset] = off;
            }
        }
        if (topKRemaining < blkTotal) {
            break;
        }
        topKRemaining -= blkTotal;
        writeStart += blkTotal;
        tmpSize -= BLOCK_SIZE;
    }
    __syncthreads();

    if (threadIdx.x < sortBlockSize / 2) {
        bitonicSort<scalar_t, index_t, true, sortBlockSize>(topKSlice, indicesSlice, K);
    }
    __syncthreads();
    scalar_t max_val = -9999;
    scalar_t sum_val = 0.0f;

    if (threadIdx.x >= 64) return;

    for (int idx = threadIdx.x; idx < K; idx += 64) {
        scalar_t weights_ = topKSlice[idx];
        index_t indices_ = indicesSlice[idx];
        indicesSliceOut[idx] = indices_;
        max_val = max(max_val, weights_);
        sum_val += __builtin_expf(weights_);
    }
    for (int stride = 32; stride > 0; stride >>= 1) {
        max_val = max(SHFL_XOR_SYNC_WIDTH(max_val, stride, 64), max_val);
        sum_val += __shfl_xor_sync(0xffffffffffffffff, sum_val, stride);
    }
    
    for (int idx = threadIdx.x; idx < K; idx += 64) {
        scalar_t res_weight = topKSlice[idx];
        res_weight = __builtin_expf(res_weight - max_val) / (sum_val * __builtin_expf(-max_val));
        topKSliceOut[idx] = res_weight;
    }
}

}   // namespace moe_softmax_topk

#undef SHFL_XOR_SYNC_WIDTH