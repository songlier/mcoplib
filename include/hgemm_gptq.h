
#include "common.cuh"
#include <iostream>
namespace hgemm_marlin_gptq {
namespace __hgemm_singular_blocks_k {
template<typename scalar_t, const sglang::ScalarTypeId w_type_id, int THREADS, int BLOCKS_M, int BLOCKS_N, int BLOCKS_K, bool HAS_ACT, bool HAS_ZP, bool HAS_M_PRED, bool HAS_NK_PRED, bool FP32_ATOMIC, bool USE_ATOMIC_CACHE>
struct LoadingManager {
    constexpr static int FragACount = 2;
    using FragA = PackType;
    constexpr static int FragBCount = 1;
    using FragB = PackType;
    constexpr static int FragCCount = 4;
    using FragC = scalar_t;
    const FragA* A;
    const FragA* A_loading;
    const FragB* B;
    const FragB* B_loading;
    FragC* C;
    float* C_temp;
    using FragScaleLoading = half2;
    using FragZeroLoading = uint32_t;
    const FragScaleLoading* scales;
    const FragScaleLoading* scales_loading;
    const FragZeroLoading* zeros;
    const FragZeroLoading* zeros_loading;
    
    int m;
    int n;
    int k;
    int quant_group_power2;
    uint8_t* smem_base;
    int bidx;
    int size_atomic_cache;

    PackTypeInt4* bsm_a_ptr;
    scalar_t* bsm_scales_ptr;
    float* bsm_zeros_ptr;
    float* remaining_bsm_ptr;

    PackTypeInt2 local_a[BLOCKS_M][2];
    PackType local_b[N_ITERS];
    PackType local_b_cache[N_ITERS];
    scalar_t local_dequanted_b_8bits[N_ITERS][2][PACK_RATIO_8BITS];
    scalar_t local_dequanted_b[N_ITERS][PACK_RATIO_4BITS];
    v2f local_scales[N_ITERS];
    v2f local_zeros[N_ITERS];
    FragScaleLoading temp_scales;
    PackType temp_zeros;
    float output[BLOCKS_M][N_ITERS][4];
    FragA temp_a[LOADING_A_LOOP];

    TileManager<BLOCKS_M, BLOCKS_N, BLOCKS_K> tile_manager;
    ThreadView tv;

    __device__ __forceinline__ void set_address(const PackTypeInt4* a,
        const PackTypeInt4* b,
        PackTypeInt4* c,
        PackTypeInt4* c_temp,
        const PackTypeInt4* scale_ptr,
        const PackTypeInt4* zp_ptr = nullptr) {
            A = (const FragA*)a;
            B = (const FragB*)b;
            C = (FragC*)c;
            C_temp = (float*)c_temp;
            scales = (const FragScaleLoading*)scale_ptr;
            if constexpr(w_type_id == sglang::kU4.id()) {
                zeros = (const FragZeroLoading*)zp_ptr;
            }
    }

    __device__ __forceinline__ bool debug() {
        #ifdef DEBUG
        bool do_print = tv.wave_idx == 1 && tv.slot_idx == 0 && tv.slot_tid == 0;
        return do_print;
        #else
        return false;
        #endif
    }

    __device__ __forceinline__ void next_k() {
        //Update only bsm_a_ptr
        bsm_a_ptr = (PackTypeInt4*)smem_base;
        bsm_a_ptr += tv.slot_tid * (PAD_SLICE_K / (sizeof(PackTypeInt4) / sizeof(scalar_t))) + tv.slot_idx;
    }

    __device__ __forceinline__ void next_k_pre() {
        A_loading += SLICE_K / FragACount;
        //B_loading += SLICE_K / PACK_RATIO_4BITS * n;
        if constexpr(w_type_id == sglang::kU4.id() || w_type_id == sglang::kU4B8.id()) {
            B_loading += SLICE_K / PACK_RATIO_4BITS * n;
        } else if constexpr(w_type_id == sglang::kU8.id() || w_type_id == sglang::kU8B128.id()) {
            B_loading += SLICE_K / PACK_RATIO_8BITS * n;
        }
    }

    __device__ __forceinline__ void ldg_a(int k_idx) {
        //32x64/2/256 = 16 / 4 = 4
        int t = tv.tid;
        int k_broad = tile_manager.tile_start_row * TILE_K + k_idx * SLICE_K;
        #pragma unroll LOADING_A_LOOP
        for (int i = 0; i < LOADING_A_LOOP; i++)  {
            int reading_m = t / (SLICE_K / FragACount);
            int reading_k = t % (SLICE_K / FragACount);
            int gvm_offset = reading_m * k / FragACount + reading_k;
            FragA* gvm_addr = (FragA*)A_loading + gvm_offset;
            //FIXME: we cannot do slice k pad as ldg_b32_bsm_async seems does not support padding
            if constexpr(HAS_M_PRED && HAS_NK_PRED) {
                bool pred = reading_m < m;
                bool pred_k = k_broad + reading_k * FragACount < k;
                pred = pred && pred_k && tile_manager.global_pred;
                ldg_b32_reg_noasync(temp_a[i], gvm_addr, pred, true);
            } else if constexpr(HAS_M_PRED) {
                bool pred = reading_m < m && tile_manager.global_pred;
                ldg_b32_reg_noasync(temp_a[i], gvm_addr, pred, true);
            } else if constexpr(HAS_NK_PRED) {
                bool pred_k = k_broad + reading_k * FragACount < k && tile_manager.global_pred;
                ldg_b32_reg_noasync(temp_a[i], gvm_addr, pred_k, true);
            } else {
                ldg_b32_reg_noasync(temp_a[i], gvm_addr, tile_manager.global_pred, true);
            }
            t += THREADS;
        }
    }

    __device__ __forceinline__ void sts_a() {
        FragA* to_bsm_a_ptr = (FragA*)smem_base;
        int t = tv.tid;
        #pragma unroll LOADING_A_LOOP
        for (int i = 0; i < LOADING_A_LOOP; i++)  {
            int reading_m = t / (SLICE_K / FragACount);
            int reading_k = t % (SLICE_K / FragACount);
            int bsm_offset = reading_m * (PAD_SLICE_K / FragACount) + reading_k;
            *(to_bsm_a_ptr + bsm_offset) = temp_a[i];
            t += THREADS;
        }
    }

    __device__ __forceinline__ void lds_a(int midx) {
        *((PackTypeInt4*)local_a[midx]) = *bsm_a_ptr;
        bsm_a_ptr += SLOT * (PAD_SLICE_K / (sizeof(PackTypeInt4) / sizeof(scalar_t)));
    }

    //TODO: implement when N_ITERS==1 or N_ITERS==3
    __device__ __forceinline__ void ldg_b(int k_idx, int korder = 0) {
        if constexpr(HAS_NK_PRED) {
            bool pred_k = tile_manager.tile_start_row * TILE_K + k_idx * SLICE_K + tv.slot_idx * PACK_RATIO_4BITS + korder < k;
            bool pred_n = tile_manager.tile_start_col * TILE_N + (tv.wave_idx * SLOT + tv.slot_tid)*N_ITERS < n;
            bool pred = pred_n && pred_k && tile_manager.global_pred;
            FragB* addr =  (FragB*)B_loading + korder * n;
            if constexpr(N_ITERS == 2) {
                ldg_b64_reg_noasync(*((PackTypeInt2*)local_b_cache), ((PackTypeInt2*)addr), pred, true);
            } else if constexpr(N_ITERS == 4) {
                ldg_b128_reg_noasync(*((PackTypeInt4*)local_b_cache), ((PackTypeInt4*)addr), pred, true);
            }
        } else {
            FragB* addr =  (FragB*)B_loading + korder * n;
            if constexpr(N_ITERS == 2) {
                ldg_b64_reg_noasync(*((PackTypeInt2*)local_b_cache), ((PackTypeInt2*)addr), tile_manager.global_pred, true);
            } else if constexpr(N_ITERS == 4) {
                ldg_b128_reg_noasync(*((PackTypeInt4*)local_b_cache), ((PackTypeInt4*)addr), tile_manager.global_pred, true);
            }
        }
    }

    __device__ __forceinline__ void swap_b_cache(int i) {
        local_b[i] = local_b_cache[i];
    }

    __device__ __forceinline__ void ldg_scales() {
        bool pred = tv.tid < TILE_N / (sizeof(FragScaleLoading) / sizeof(scalar_t)) && tile_manager.global_pred;
        if constexpr(HAS_NK_PRED) {
            pred = pred && tv.tid < (n - tile_manager.tile_start_col * TILE_N) / (sizeof(FragScaleLoading) / sizeof(scalar_t));
        }
        //FragScaleLoading *scale_bsm = (FragScaleLoading*)(smem_base + 0x2000) + tv.tid;
        FragScaleLoading *gvm_addr = (FragScaleLoading*)scales_loading + tv.tid;
        //ldg_b32_bsm_async(scale_bsm, gvm_addr, pred, false);
        ldg_b32_reg_noasync(*((PackType*)&temp_scales), gvm_addr, pred, true);
    }

    __device__ __forceinline__ void ldg_zp() {
        if constexpr(w_type_id == sglang::kU4.id()) {
            bool pred = (tv.tid < TILE_N / PACK_RATIO_4BITS) && tile_manager.global_pred;
            if constexpr(HAS_NK_PRED) {
                pred = pred && tv.tid < ((n - tile_manager.tile_start_col * TILE_N) / PACK_RATIO_4BITS);
            }
            FragZeroLoading *gvm_addr = (FragZeroLoading*)zeros_loading + tv.tid;
            ldg_b32_reg_noasync(*((PackType*)&temp_zeros), gvm_addr, pred, true);
        }
    }

    __device__ __forceinline__ void sts_scales() {
        FragScaleLoading *scale_bsm = (FragScaleLoading*)(smem_base + 0x2000) + tv.tid;
        *scale_bsm = temp_scales;
    }

    __device__ __forceinline__ void sts_zeros() {
        if constexpr(w_type_id == sglang::kU4.id()) {
            bool pred = (tv.tid < TILE_N / PACK_RATIO_4BITS) && tile_manager.global_pred;
            if (pred) {
                float temp[PACK_RATIO_4BITS];
                decompress_zero_4bits(temp_zeros, temp);
                float *scale_bsm = (float*)(smem_base + 0x3000) + tv.tid * PACK_RATIO_4BITS;
                for (int i = 0; i < PACK_RATIO_4BITS; i++) {
                    *(scale_bsm + i) = temp[i];
                }
            }
        }
    }

    //TODO: implement when N_ITERS==1 or N_ITERS==3
    __device__ __forceinline__ void lds_scales() {
        if constexpr(N_ITERS==2) {
            *((half2*)local_dequanted_b[0]) = *((half2*)bsm_scales_ptr);
        } else if constexpr(N_ITERS==4) {
            *((PackTypeInt2*)local_dequanted_b[0]) = *((PackTypeInt2*)bsm_scales_ptr);
        }
    }

    __device__ __forceinline__ void pack_scales() {
        if constexpr(w_type_id == sglang::kU4B8.id()) {
            #pragma unroll
            for (int i = 0; i < N_ITERS; i++) {
                float s = local_dequanted_b[0][i];
                float z = -8 * s;
                local_scales[i] = {s, s};
                local_zeros[i] = {z, z};
            }
        } else if constexpr(w_type_id == sglang::kU4.id()) {
            #pragma unroll
            for (int i = 0; i < N_ITERS; i++) {
                float s = local_dequanted_b[0][i];
                float z = *(bsm_zeros_ptr + i);
                z = z * s;
                local_scales[i] = {s, s};
                local_zeros[i] = {z, z};
            }
        } else if constexpr(w_type_id == sglang::kU8B128.id()) {
            #pragma unroll
            for (int i = 0; i < N_ITERS; i++) {
                float s = local_dequanted_b[0][i];
                float z = -128 * s;
                local_scales[i] = {s, s};
                local_zeros[i] = {z, z};
            }
        } else if constexpr(w_type_id == sglang::kU8.id()) {
            // should apply zeros
        }
    }

    __device__ __forceinline__ void dequant(int kdx, int korder = 0) {
        if constexpr(w_type_id == sglang::kU4.id() || w_type_id == sglang::kU4B8.id()) {
            dequant_gptq_4bits<scalar_t>(local_b[kdx], local_dequanted_b[kdx], local_scales[kdx], local_zeros[kdx]);
        } else if constexpr(w_type_id == sglang::kU8.id() || w_type_id == sglang::kU8B128.id()) {
            dequant_gptq_8bits<scalar_t>(local_b[kdx], local_dequanted_b_8bits[kdx][korder], local_scales[kdx], local_zeros[kdx]);
        }
    }

    __device__ __forceinline__ void matmul(int mdx) {
        if constexpr(w_type_id == sglang::kU4.id() || w_type_id == sglang::kU4B8.id()) {
            #pragma unroll
            for (int i = 0; i < N_ITERS; i++) {
                mma_16x16x16<scalar_t>(local_a[mdx][0], *((PackTypeInt2*)local_dequanted_b[i]), *((PackTypeInt4*)output[mdx][i]));
                mma_16x16x16<scalar_t>(local_a[mdx][1], *((PackTypeInt2*)local_dequanted_b[i] + 1), *((PackTypeInt4*)output[mdx][i]));
	    }
        } else if constexpr(w_type_id == sglang::kU8.id() || w_type_id == sglang::kU8B128.id()) {
            #pragma unroll
            for (int i = 0; i < N_ITERS; i++) {
                mma_16x16x16<scalar_t>(local_a[mdx][0], *((PackTypeInt2*)local_dequanted_b_8bits[i][0]), *((PackTypeInt4*)output[mdx][i]));
                mma_16x16x16<scalar_t>(local_a[mdx][1], *((PackTypeInt2*)local_dequanted_b_8bits[i][1]), *((PackTypeInt4*)output[mdx][i]));
            }
        }
    }

    __device__ __forceinline__ void clear_c() {
        #pragma unroll
        for (int miter = 0; miter < BLOCKS_M; miter++) {
            #pragma unroll
            for (int niter = 0; niter < N_ITERS; niter++) {
                #pragma unroll
                for (int miter2 = 0; miter2 < 4; miter2++) {
                    output[miter][niter][miter2] = 0;
                }
            }
        }
    }

    //functions for preloading next tile data
    __device__ __forceinline__ void init_address_pre(int _m, int _n, int _k, int _quant_group_power2, int _bidx, int _iters, int _size_atomic_cache, uint8_t *_smem_base) {
        tv.init();
        m = _m;
        n = _n;
        k = _k;
        quant_group_power2 = _quant_group_power2;
        bidx = _bidx;
        size_atomic_cache = _size_atomic_cache;
        smem_base = _smem_base;
        tile_manager.init(m, n, k, bidx, _iters);
        init_tile_pre(tile_manager.tile_start_col, tile_manager.tile_start_row);
    }

    __device__ __forceinline__ void init_tile_pre(int col, int row) {
        //Initialize start slice address and set them to A_loading and B_loading
        int offset_n = col * TILE_N;
        int offset_k = row * TILE_K;
        //A_loading address will always be valid
        A_loading = A + offset_k / (FragACount);
        //B_loading = B + (offset_k / PACK_RATIO_4BITS * n + offset_n) / (FragBCount) + tv.slot_idx * n + (tv.wave_idx * SLOT + tv.slot_tid)*N_ITERS;
        if constexpr(w_type_id == sglang::kU4.id() || w_type_id == sglang::kU4B8.id()) {
            B_loading = B + (offset_k / PACK_RATIO_4BITS * n + offset_n) / (FragBCount) + tv.slot_idx * n + (tv.wave_idx * SLOT + tv.slot_tid)*N_ITERS;
        } else if constexpr(w_type_id == sglang::kU8.id() || w_type_id == sglang::kU8B128.id()) {
            B_loading = B + (offset_k / PACK_RATIO_8BITS * n + offset_n) / (FragBCount) + tv.slot_idx * n * 2 + (tv.wave_idx * SLOT + tv.slot_tid)*N_ITERS;
        }
        scales_loading = scales + ((offset_k >> quant_group_power2) * n + offset_n) / (sizeof(FragScaleLoading)/sizeof(scalar_t));
        if constexpr(w_type_id == sglang::kU4.id()) {
            zeros_loading = zeros + ((offset_k >> quant_group_power2) * n + offset_n) / PACK_RATIO_4BITS;
        }
    }

    __device__ __forceinline__ void next_tile_pre() {
        tile_manager.next_tile_pre();
        init_tile_pre(tile_manager.tile_start_col, tile_manager.tile_start_row);
    }

    __device__ __forceinline__ void init_bsm_addr() {
        bsm_a_ptr = (PackTypeInt4*)smem_base;           //use 8k bytes, will load at most 32x128*sizeof(half), either m32k128 or m128k32
        remaining_bsm_ptr = (float*)(smem_base + 0x2000 + 0x1000); //3k bytes
        bsm_a_ptr += tv.slot_tid * (PAD_SLICE_K / (sizeof(PackTypeInt4) / sizeof(scalar_t))) + tv.slot_idx;
        bsm_scales_ptr = (scalar_t*)(smem_base + 0x2000);      //use 128xsizeof(float)*2 = 1k
        bsm_scales_ptr += (tv.wave_idx * SLOT + tv.slot_tid) * N_ITERS;
        if constexpr(w_type_id == sglang::kU4.id()) {
            bsm_zeros_ptr = (float*)(smem_base + 0x3000);
            bsm_zeros_ptr += (tv.wave_idx * SLOT + tv.slot_tid) * N_ITERS;
        }
    }

    __device__ __forceinline__ void write_c(int offset, const float& v) {
        if constexpr(FP32_ATOMIC) {
            atomicAdd(C_temp + offset, v);
        } else {
            atomicAdd(C + offset, (scalar_t)v);
        }
    }

    __device__ __forceinline__ void write_c(int store_m, int n, int store_n, const float& v) {
        if constexpr(USE_ATOMIC_CACHE) {
            //When USE_ATOMIC_CACHE is true, FP32_ATOMIC must be true
            C_temp[store_m * n * size_atomic_cache + tile_manager.atomic_idx * n + store_n] = v;
        } else if constexpr(FP32_ATOMIC) {
            atomicAdd(C_temp + store_m * n + store_n, v);
        } else {
            atomicAdd(C + store_m * n + store_n, (scalar_t)v);
        }
    }

    __device__ __forceinline__ void reset_atomic_cache() {
        if constexpr(USE_ATOMIC_CACHE) {
            if (!tile_manager.last_atomic_cache) return;
            int k_broad = tv.slot_idx * 4;
            int n_broad = (tv.wave_idx * SLOT + tv.slot_tid) * N_ITERS + tile_manager.tile_start_col * TILE_N;
            #pragma unroll
            for (int miter = 0; miter < BLOCKS_M; miter++) {
                #pragma unroll
                for (int niter = 0; niter < N_ITERS; niter++) {
                    int store_n = n_broad + niter;
                    #pragma unroll
                    for (int miter2 = 0; miter2 < 4; miter2++) {
                        int store_m = k_broad + miter * SLICE_M + miter2;
                        if constexpr(HAS_M_PRED && HAS_NK_PRED) {
                            if (store_m < m && store_n < n) {
                                // write_c(store_m * n + store_n, output[miter][niter][miter2]);
                                //Reset unused atomic cache
                                for (int i = tile_manager.atomic_idx + 1; i < size_atomic_cache; i++) {
                                    C_temp[store_m * n * size_atomic_cache + i * n + store_n] = 0;
                                }
                            }
                        } else if constexpr(HAS_M_PRED) {
                            if (store_m < m) {
                                // write_c(store_m * n + store_n, output[miter][niter][miter2]);
                                //Reset unused atomic cache
                                for (int i = tile_manager.atomic_idx + 1; i < size_atomic_cache; i++) {
                                    C_temp[store_m * n * size_atomic_cache + i * n + store_n] = 0;
                                }
                            }
                        } else if constexpr(HAS_NK_PRED) {
                            if (store_n < n) {
                                // write_c(store_m * n + store_n, output[miter][niter][miter2]);
                                //Reset unused atomic cache
                                for (int i = tile_manager.atomic_idx + 1; i < size_atomic_cache; i++) {
                                    C_temp[store_m * n * size_atomic_cache + i * n + store_n] = 0;
                                }
                            }
                        } else {
                            // write_c(store_m * n + store_n, output[miter][niter][miter2]);
                            //Reset unused atomic cache
                            for (int i = tile_manager.atomic_idx + 1; i < size_atomic_cache; i++) {
                                C_temp[store_m * n * size_atomic_cache + i * n + store_n] = 0;
                            }
                        }
                    }
                }
            }
        }
    }

    //atomic write to c
    __device__ __forceinline__ void write_c_pre() {
        int k_broad = tv.slot_idx * 4;
        int n_broad = (tv.wave_idx * SLOT + tv.slot_tid) * N_ITERS + tile_manager.tile_start_col_cache * TILE_N;
        #pragma unroll
        for (int miter = 0; miter < BLOCKS_M; miter++) {
            #pragma unroll
            for (int niter = 0; niter < N_ITERS; niter++) {
                int store_n = n_broad + niter;
                #pragma unroll
                for (int miter2 = 0; miter2 < 4; miter2++) {
                    int store_m = k_broad + miter * SLICE_M + miter2;
                    if constexpr(HAS_M_PRED && HAS_NK_PRED) {
                        if (store_m < m && store_n < n) {
                            // write_c(store_m * n + store_n, output[miter][niter][miter2]);
                            write_c(store_m, n, store_n, output[miter][niter][miter2]);
			            }
                    } else if constexpr(HAS_M_PRED) {
                        if (store_m < m) {
                            // write_c(store_m * n + store_n, output[miter][niter][miter2]);
                            write_c(store_m, n, store_n, output[miter][niter][miter2]);
			            }
                    } else if constexpr(HAS_NK_PRED) {
                        if (store_n < n) {
                            // write_c(store_m * n + store_n, output[miter][niter][miter2]);
                            write_c(store_m, n, store_n, output[miter][niter][miter2]);
			            }
                    } else {
                        // write_c(store_m * n + store_n, output[miter][niter][miter2]);
                        write_c(store_m, n, store_n, output[miter][niter][miter2]);
		            }
                }
            }
        }
        tile_manager.reset_atomic_idx();
    }

    __device__ __forceinline__ void write_c_pre(const float& s) {
        int k_broad = tv.slot_idx * 4;
        int n_broad = (tv.wave_idx * SLOT + tv.slot_tid) * N_ITERS + tile_manager.tile_start_col_cache * TILE_N;
        #pragma unroll
        for (int miter = 0; miter < BLOCKS_M; miter++) {
            #pragma unroll
            for (int niter = 0; niter < N_ITERS; niter++) {
                int store_n = n_broad + niter;
                #pragma unroll
                for (int miter2 = 0; miter2 < 4; miter2++) {
                    int store_m = k_broad + miter * SLICE_M + miter2;
                    if constexpr(HAS_M_PRED && HAS_NK_PRED) {
                        if (store_m < m && store_n < n) {
                            // write_c(store_m * n + store_n, output[miter][niter][miter2] * s);
                            write_c(store_m, n, store_n, output[miter][niter][miter2] * s);
			            }
                    } else if constexpr(HAS_M_PRED) {
                        if (store_m < m) {
                            // write_c(store_m * n + store_n, output[miter][niter][miter2] * s);
                            write_c(store_m, n, store_n, output[miter][niter][miter2] * s);
			            }
                    } else if constexpr(HAS_NK_PRED) {
                        if (store_n < n) {
                            // write_c(store_m * n + store_n, output[miter][niter][miter2] * s);
                            write_c(store_m, n, store_n, output[miter][niter][miter2] * s);
			            }
                    } else {
                        // write_c(store_m * n + store_n, output[miter][niter][miter2] * s);
                        write_c(store_m, n, store_n, output[miter][niter][miter2] * s);
		            }
                }
            }
        }
        tile_manager.reset_atomic_idx();
    }

    template<bool KTAIL>
    __device__ __forceinline__ void on_dequant_niters1(int k_idx) {
        if constexpr(w_type_id == sglang::kU4.id() || w_type_id == sglang::kU4B8.id()) {
            swap_b_cache(0);
            sts_a(); // reorder data_a from reg(gvm resouce) to bsm
            if constexpr(!KTAIL) {
                next_k_pre(); // preload gvm a/b
                ldg_b(k_idx + 1); //preload b for next k
                ldg_a(k_idx + 1); //preload a for next k
            }
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(0); //dequant b0
        } else if constexpr(w_type_id == sglang::kU8.id() || w_type_id == sglang::kU8B128.id()) {
            swap_b_cache(0);
            ldg_b(k_idx, 1);
            sts_a(); // reorder data_a from reg(gvm resouce) to bsm
            dequant(0);
            swap_b_cache(0);
            if constexpr(!KTAIL) {
                next_k_pre(); // preload gvm a/b
                ldg_b(k_idx + 1); //preload b for next k
                ldg_a(k_idx + 1); //preload a for next k
            }
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(0); //dequant b0
        }
    }

    template<bool KTAIL>
    __device__ __forceinline__ void on_dequant_niters2(int k_idx) {
        if constexpr(w_type_id == sglang::kU4.id() || w_type_id == sglang::kU4B8.id()) {
            swap_b_cache(0);
            dequant(0); //dequant b0
            sts_a(); // reorder data_a from reg(gvm resouce) to bsm
            swap_b_cache(1);
            if constexpr(!KTAIL) {
                next_k_pre(); // preload gvm a/b
                ldg_b(k_idx + 1); //preload b for next k
                ldg_a(k_idx + 1); //preload a for next k
            }
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(1);   //dequant b64
        } else if constexpr(w_type_id == sglang::kU8.id() || w_type_id == sglang::kU8B128.id()) {
            swap_b_cache(0);
            swap_b_cache(1);
            sts_a(); // reorder data_a from reg(gvm resouce) to bsm
            ldg_b(k_idx, 1);
            dequant(0); //dequant b0
            dequant(1);
            swap_b_cache(0);
            swap_b_cache(1);
            if constexpr(!KTAIL) {
                next_k_pre(); // preload gvm a/b
                ldg_b(k_idx + 1); //preload b for next k
                ldg_a(k_idx + 1); //preload a for next k
            }
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(0, 1);
            dequant(1, 1);   //dequant b64
        }
    }

    template<bool KTAIL>
    __device__ __forceinline__ void on_dequant_niters3(int k_idx) {
        if constexpr(w_type_id == sglang::kU4.id() || w_type_id == sglang::kU4B8.id()) {
            swap_b_cache(0);
            dequant(0); //dequant b0
            sts_a(); // reorder data_a from reg(gvm resouce) to bsm
            swap_b_cache(1);
            dequant(1);
            swap_b_cache(2);
            if constexpr(!KTAIL) {
                next_k_pre(); // preload gvm a/b
                ldg_b(k_idx + 1); //preload b for next k
                ldg_a(k_idx + 1); //preload a for next k
            }
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(2); //dequant b0
        } else if constexpr(w_type_id == sglang::kU8.id() || w_type_id == sglang::kU8B128.id()) {
            swap_b_cache(0);
            swap_b_cache(1);
            swap_b_cache(2);
            sts_a(); // reorder data_a from reg(gvm resouce) to bsm
            ldg_b(k_idx, 1);
            dequant(0); //dequant b0
            dequant(1); //dequant b0
            dequant(2); //dequant b0
            swap_b_cache(0);
            swap_b_cache(1);
            swap_b_cache(2);
            if constexpr(!KTAIL) {
                next_k_pre(); // preload gvm a/b
                ldg_b(k_idx + 1); //preload b for next k
                ldg_a(k_idx + 1); //preload a for next k
            }
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(0, 1); //dequant b0
            dequant(1, 1); //dequant b0
            dequant(2, 1); //dequant b0
        }
    }

    template<bool KTAIL>
    __device__ __forceinline__ void on_dequant_niters4(int k_idx) {
        if constexpr(w_type_id == sglang::kU4.id() || w_type_id == sglang::kU4B8.id()) {
            swap_b_cache(0);
            dequant(0); //dequant b0
            sts_a(); // reorder data_a from reg(gvm resouce) to bsm
            swap_b_cache(1);
            dequant(1); //dequant b1
            swap_b_cache(2);
            swap_b_cache(3);
            dequant(2); //dequant b2
            if constexpr(!KTAIL) {
                next_k_pre(); // preload gvm a/b
                ldg_b(k_idx + 1); //preload b for next k
                ldg_a(k_idx + 1); //preload a for next k
            }
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(3); //dequant b3
        } else {
            swap_b_cache(0);
            swap_b_cache(1);
            swap_b_cache(2);
            swap_b_cache(3);
            dequant(0); //dequant b0
            sts_a(); // reorder data_a from reg(gvm resouce) to bsm
            ldg_b(k_idx, 1);
            dequant(1); //dequant b0
            dequant(2); //dequant b0
            dequant(3); //dequant b0
            swap_b_cache(0);
            swap_b_cache(1);
            swap_b_cache(2);
            swap_b_cache(3);
            if constexpr(!KTAIL) {
                next_k_pre(); // preload gvm a/b
                ldg_b(k_idx + 1); //preload b for next k
                ldg_a(k_idx + 1); //preload a for next k
            }
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(0, 1); //dequant b3
            dequant(1, 1); //dequant b3
            dequant(2, 1); //dequant b3
            dequant(3, 1); //dequant b3
        }
    }

    template<bool KTAIL>
    __device__ __forceinline__ void on_dequant(int kdx) {
        if constexpr(N_ITERS == 1) on_dequant_niters1<KTAIL>(kdx);
        else if constexpr(N_ITERS == 2) on_dequant_niters2<KTAIL>(kdx);
        else if constexpr(N_ITERS == 3) on_dequant_niters3<KTAIL>(kdx);
        else if constexpr(N_ITERS == 4) on_dequant_niters4<KTAIL>(kdx);
    }
};

template<typename scalar_t,
    const sglang::ScalarTypeId w_type_id,
    const int THREADS,          // number of threads in a threadblock
    const int BLOCKS_M,         // number of 16x16 blocks in the m
                                // dimension (batchsize) of the
                                // threadblock
    const int BLOCKS_N,         // same for n dimension (output)
    const int BLOCKS_K,         // same for k dimension (reduction)
    const bool HAS_ACT_ORDER,   // whether act_order is enabled
    const bool HAS_ZP,          // whether zero-points are enabled
    const bool HAS_M_PRED = true,  //If we should use predictors to load m from gvm
    const bool HAS_NK_PRED = true,  //If we should use predictors to load nk from gvm,
    const bool FP32_ATOMIC = true,
    const bool USE_ATOMIC_CACHE = true,
    const bool HAS_SIZEM_PTR = true //VLLM Version of marlin linear
    >
__global__ void hgemm_gptq(
    const PackTypeInt4* __restrict__ A,  // fp16 input matrix of shape mxk
    const PackTypeInt4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    PackTypeInt4* __restrict__ C,        // fp16 output buffer of shape mxn
    PackTypeInt4* __restrict__ C_tmp,    // fp32 tmp output buffer (for reduce)
    const int* size_m_ptr,
    const PackTypeInt4* __restrict__ scales_ptr,  // fp16 quantization scales of shape
                                          // (k/groupsize)xn
    const PackTypeInt4* __restrict__ zp_ptr,      // 4bit packed zero-points of shape
                                          // (k/groupsize)x(n/pack_factor)
    const int* __restrict__ g_idx,        // int32 group indices of shape k
    int prob_m,           // batch dimension m
    int prob_n,           // output dimension n
    int prob_k,           // reduction dimension k
    int quant_group_power2, // quant group means how many quanted values share the same scale and zero, this value restricts to 2^x where x >= 5
    int max_iters,        // max tile iterations for one block
    int size_atomic_cache, // When USE_ATOMIC_CACHE, one result is divided at most size_atomic_cache parts, all C are write directly, according to offset atomic_idx
    int* locks,           // extra global storage for barrier synchronization
    bool use_fp32_reduce  // whether to use fp32 global reduce
) {
    int bidx = blockIdx.x;
    int bidy = blockIdx.y; //Indicates index of FULL_M_BLOCKS
    if constexpr (HAS_SIZEM_PTR) {
        int real_prob_m = *size_m_ptr;
        //discard data if current m larger than real_prob_m
        if (real_prob_m <= blockIdx.y * (MAX_BLOCKS_M * SLICE_M)) return;
        prob_m = real_prob_m;
    }
    __shared__ uint8_t smem_base[0x4000]; //4x16x256 = 16Kbytes
    using LoadingManagerType = LoadingManager<scalar_t, w_type_id, THREADS, BLOCKS_M, BLOCKS_N, BLOCKS_K, HAS_ACT_ORDER, HAS_ZP, HAS_M_PRED, HAS_NK_PRED, FP32_ATOMIC, USE_ATOMIC_CACHE>;
    LoadingManagerType loading_manager;
    A += blockIdx.y * (MAX_BLOCKS_M * SLICE_M) * prob_k / (sizeof(PackTypeInt4) / sizeof(scalar_t));
    C += blockIdx.y * (MAX_BLOCKS_M * SLICE_M) * prob_n / (sizeof(PackTypeInt4) / sizeof(scalar_t));
    if constexpr(USE_ATOMIC_CACHE) {
        C_tmp += blockIdx.y * (MAX_BLOCKS_M * SLICE_M) * prob_n * size_atomic_cache / (sizeof(PackTypeInt4) / sizeof(float));
    } else if constexpr(FP32_ATOMIC) {
        C_tmp += blockIdx.y * (MAX_BLOCKS_M * SLICE_M) * prob_n / (sizeof(PackTypeInt4) / sizeof(float));
    }
    loading_manager.set_address(A, B, C, C_tmp, scales_ptr, zp_ptr);
    //loading_manager.init_address(prob_m, prob_n, prob_k, bidx, max_iters, smem_base);
    loading_manager.init_address_pre(std::min<int>(MAX_BLOCKS_M*SLICE_M, prob_m - blockIdx.y * (MAX_BLOCKS_M * SLICE_M)), prob_n, prob_k, quant_group_power2, bidx, max_iters, size_atomic_cache, smem_base);
    loading_manager.clear_c();

    loading_manager.reset_atomic_cache();

    while (max_iters > 0) {
        loading_manager.init_bsm_addr(); //reset all bsm address for current tile
        loading_manager.ldg_scales(); //Load all scales to bsm
        loading_manager.ldg_zp();
        loading_manager.ldg_b(0);    //load b0 and b64, two gvm
        loading_manager.ldg_a(0);    //Load first k0~31 and all m, one ldg_b128, heavy load
        loading_manager.sts_scales();
        loading_manager.sts_zeros();
        barrier_bsm;
        loading_manager.lds_scales(); //load scale0 and scale64
        loading_manager.pack_scales(); //pack scales into two v2f structure

        int k_idx = 0;
        if constexpr(BLOCKS_K > 1) {
            #pragma unroll BLOCKS_K - 1
            for (; k_idx < BLOCKS_K - 1; k_idx++) {
                int m_idx = 0;
                loading_manager.template on_dequant<false>(k_idx);
                //Loop for 3 times so that we can add some loading instructions before matmul
                if constexpr(BLOCKS_M > 1) {
                    #pragma unroll BLOCKS_M - 1
                    for (; m_idx < BLOCKS_M - 1; m_idx++) {
                        loading_manager.lds_a(m_idx+1); //pre load next 16 batches
                        loading_manager.matmul(m_idx); //do matmul
                    }
                }
                barrier_bsm; //wait for the last 16 batches ready, becasue of k_idx will be change
                loading_manager.next_k(); //modify gvm/bsm address of a and b
                loading_manager.matmul(m_idx); //do matmul
            }
        }
        int m_idx = 0;
        loading_manager.template on_dequant<true>(k_idx);
        if constexpr(BLOCKS_M > 1) {
            #pragma unroll BLOCKS_M - 1
            for (; m_idx < BLOCKS_M - 1; m_idx++) {
                loading_manager.lds_a(m_idx+1); //pre load next 16 batches
                loading_manager.matmul(m_idx); //do matmul
            }
        }
        loading_manager.next_tile_pre();

        loading_manager.matmul(m_idx); //do matmul
        max_iters--;

        if (loading_manager.tile_manager.need_save_data_pre()) {
            loading_manager.write_c_pre(); // reduce and write back
            loading_manager.clear_c();
        }

        barrier_bsm;
    }
}

} //end of namespace __hgemm_singular_blocks_k

namespace __hgemm_even_blocks_k {
template<typename scalar_t, const sglang::ScalarTypeId w_type_id, int THREADS, int BLOCKS_M, int BLOCKS_N, int BLOCKS_K, bool HAS_ACT, bool HAS_ZP, bool HAS_M_PRED, bool HAS_NK_PRED, bool FP32_ATOMIC, bool USE_ATOMIC_CACHE>
struct LoadingManager {
    constexpr static int FragACount = 4;
    using FragA = PackTypeInt2;
    constexpr static int FragBCount = 1;
    using FragB = PackType;
    constexpr static int FragCCount = 4;
    using FragC = scalar_t;
    const FragA* A;
    const FragA* A_loading;
    const FragB* B;
    const FragB* B_loading;
    FragC* C;
    float* C_temp;
    using FragScaleLoading = half2;
    using FragZeroLoading = uint32_t;
    const FragScaleLoading* scales;
    const FragScaleLoading* scales_loading;
    const FragZeroLoading* zeros;
    const FragZeroLoading* zeros_loading;

    constexpr static int DOUBLE_SLICE_K = SLICE_K * 2;
    constexpr static int DOUBLE_PAD_SLICE_K = SLICE_K * 2 + sizeof(PackTypeInt4) / sizeof(scalar_t);

    int m;
    int n;
    int k;
    int quant_group_power2;
    uint8_t* smem_base;
    int bidx;
    int size_atomic_cache;

    PackTypeInt4* bsm_a_ptr;
    scalar_t* bsm_scales_ptr;
    float* bsm_zeros_ptr;
    //float* remaining_bsm_ptr;

    PackTypeInt2 local_a[BLOCKS_M][2];
    PackType local_b[N_ITERS];
    PackType local_b_cache[N_ITERS];
    scalar_t local_dequanted_b[N_ITERS][PACK_RATIO_4BITS];
    scalar_t local_dequanted_b_8bits[N_ITERS][2][PACK_RATIO_8BITS];
    v2f local_scales[N_ITERS];
    v2f local_zeros[N_ITERS];
    FragScaleLoading temp_scales;
    PackType temp_zeros;
    float output[BLOCKS_M][N_ITERS][4];
    FragA temp_a[LOADING_A_LOOP];

    TileManager<BLOCKS_M, BLOCKS_N, BLOCKS_K> tile_manager;
    ThreadView tv;

    __device__ __forceinline__ void set_address(const PackTypeInt4* a,
        const PackTypeInt4* b,
        PackTypeInt4* c,
        PackTypeInt4* c_temp,
        const PackTypeInt4* scale_ptr,
        const PackTypeInt4* zp_ptr = nullptr) {
            A = (const FragA*)a;
            B = (const FragB*)b;
            C = (FragC*)c;
            C_temp = (float*)c_temp;
            scales = (const FragScaleLoading*)scale_ptr;
            if constexpr(w_type_id == sglang::kU4.id()) {
                zeros = (const FragZeroLoading*)zp_ptr;
            }
    }

    __device__ __forceinline__ bool debug() {
        #ifdef DEBUG
        bool do_print = tv.wave_idx == 1 && tv.slot_idx == 0 && tv.slot_tid == 0;
        return do_print;
        #else
        return false;
        #endif
    }

    __device__ __forceinline__ void next_k0() {
        //reset bsm a to base
        bsm_a_ptr = (PackTypeInt4*)smem_base;
        bsm_a_ptr += tv.slot_tid * (DOUBLE_PAD_SLICE_K / (sizeof(PackTypeInt4) / sizeof(scalar_t))) + tv.slot_idx;
    }

    __device__ __forceinline__ void next_k1() {
        //Update only bsm_a_ptr
        bsm_a_ptr = (PackTypeInt4*)smem_base;
        bsm_a_ptr += tv.slot_tid * (DOUBLE_PAD_SLICE_K / (sizeof(PackTypeInt4) / sizeof(scalar_t))) + tv.slot_idx + WAVE_SLOTS;
        //load k32~k63
        //bsm_a_ptr += 4;
    }

    __device__ __forceinline__ void next_k0_pre() {
        //A_loading += SLICE_K / FragACount;
        //B_loading += SLICE_K / PACK_RATIO_4BITS * n;
        if constexpr(w_type_id == sglang::kU4.id() || w_type_id == sglang::kU4B8.id()) {
            B_loading += SLICE_K / PACK_RATIO_4BITS * n;
        } else if constexpr(w_type_id == sglang::kU8.id() || w_type_id == sglang::kU8B128.id()) {
            B_loading += SLICE_K / PACK_RATIO_8BITS * n;
        }
    }

    __device__ __forceinline__ void next_k1_pre() {
        A_loading += DOUBLE_SLICE_K / FragACount;
        if constexpr(w_type_id == sglang::kU4.id() || w_type_id == sglang::kU4B8.id()) {
            B_loading += SLICE_K / PACK_RATIO_4BITS * n;
        } else if constexpr(w_type_id == sglang::kU8.id() || w_type_id == sglang::kU8B128.id()) {
            B_loading += SLICE_K / PACK_RATIO_8BITS * n;
        }
    }

    __device__ __forceinline__ void ldg_a(int k_idx) {
        //32x64/2/256 = 16 / 4 = 4
        int t = tv.tid;
        int k_broad = tile_manager.tile_start_row * TILE_K + k_idx * SLICE_K;
        #pragma unroll LOADING_A_LOOP
        for (int i = 0; i < LOADING_A_LOOP; i++)  {
            int reading_m = t / (DOUBLE_SLICE_K / FragACount);
            int reading_k = t % (DOUBLE_SLICE_K / FragACount);
            int gvm_offset = reading_m * k / FragACount + reading_k;
            FragA* gvm_addr = (FragA*)A_loading + gvm_offset;
            //FIXME: we cannot do slice k pad as ldg_b32_bsm_async seems does not support padding
            if constexpr(HAS_M_PRED && HAS_NK_PRED) {
                bool pred = reading_m < m;
                bool pred_k = k_broad + reading_k * FragACount < k;
                pred = pred && pred_k && tile_manager.global_pred;
                ldg_b64_reg_noasync(temp_a[i], gvm_addr, pred, true);
            } else if constexpr(HAS_M_PRED) {
                bool pred = reading_m < m && tile_manager.global_pred;
                ldg_b64_reg_noasync(temp_a[i], gvm_addr, pred, true);
            } else if constexpr(HAS_NK_PRED) {
                bool pred_k = k_broad + reading_k * FragACount < k && tile_manager.global_pred;
                ldg_b64_reg_noasync(temp_a[i], gvm_addr, pred_k, true);
            } else {
                ldg_b64_reg_noasync(temp_a[i], gvm_addr, tile_manager.global_pred, true);
            }
            t += THREADS;
        }
    }

    __device__ __forceinline__ void sts_a() {
        FragA* to_bsm_a_ptr = (FragA*)smem_base;
        int t = tv.tid;
        #pragma unroll LOADING_A_LOOP
        for (int i = 0; i < LOADING_A_LOOP; i++)  {
            int reading_m = t / (DOUBLE_SLICE_K / FragACount);
            int reading_k = t % (DOUBLE_SLICE_K / FragACount);
            int bsm_offset = reading_m * (DOUBLE_PAD_SLICE_K / FragACount) + reading_k;
            *(to_bsm_a_ptr + bsm_offset) = temp_a[i];
            t += THREADS;
        }
    }

    __device__ __forceinline__ void lds_a(int midx) {
        *((PackTypeInt4*)local_a[midx]) = *bsm_a_ptr;
        bsm_a_ptr += SLOT * (DOUBLE_PAD_SLICE_K / (sizeof(PackTypeInt4) / sizeof(scalar_t)));
    }

    //TODO: implement when N_ITERS==1 or N_ITERS==3
    //korder used gptq_8bits, ldg_b will load two times in one SLICE_K
    //For example, t0 loads packed_k0, packed_k1, and packed_k0 represents a packed 4 ks in first line of B,
    //and packed_k1 represents a packed 4 ks in second line of B
    __device__ __forceinline__ void ldg_b(int k_idx, int korder = 0) {
        if constexpr(HAS_NK_PRED) {
            bool pred_k = tile_manager.tile_start_row * TILE_K + k_idx * SLICE_K + tv.slot_idx * PACK_RATIO_4BITS + korder < k;
            bool pred_n = tile_manager.tile_start_col * TILE_N + (tv.wave_idx * SLOT + tv.slot_tid)*N_ITERS < n;
            bool pred = pred_n && pred_k && tile_manager.global_pred;
            FragB* addr =  (FragB*)B_loading + korder * n;
            if constexpr(N_ITERS == 2) {
                ldg_b64_reg_noasync(*((PackTypeInt2*)local_b_cache), ((PackTypeInt2*)addr), pred, true);
            } else if constexpr(N_ITERS == 4) {
                ldg_b128_reg_noasync(*((PackTypeInt4*)local_b_cache), ((PackTypeInt4*)addr), pred, true);
            }
        } else {
            FragB* addr =  (FragB*)B_loading + korder * n;
            if constexpr(N_ITERS == 2) {
                ldg_b64_reg_noasync(*((PackTypeInt2*)local_b_cache), ((PackTypeInt2*)addr), tile_manager.global_pred, true);
            } else if constexpr(N_ITERS == 4) {
                ldg_b128_reg_noasync(*((PackTypeInt4*)local_b_cache), ((PackTypeInt4*)addr), tile_manager.global_pred, true);
            }
        }
    }

    __device__ __forceinline__ void swap_b_cache(int i) {
        local_b[i] = local_b_cache[i];
    }

    __device__ __forceinline__ void ldg_scales() {
        bool pred = tv.tid < TILE_N / (sizeof(FragScaleLoading) / sizeof(scalar_t)) && tile_manager.global_pred;
        if constexpr(HAS_NK_PRED) {
            pred = pred && tv.tid < (n - tile_manager.tile_start_col * TILE_N) / (sizeof(FragScaleLoading) / sizeof(scalar_t));
        }
        //FragScaleLoading *scale_bsm = (FragScaleLoading*)(smem_base + 0x2000) + tv.tid;
        FragScaleLoading *gvm_addr = (FragScaleLoading*)scales_loading + tv.tid;
        //ldg_b32_bsm_async(scale_bsm, gvm_addr, pred, false);
        ldg_b32_reg_noasync(*((PackType*)&temp_scales), gvm_addr, pred, true);
    }

    __device__ __forceinline__ void ldg_zp() {
        if constexpr(w_type_id == sglang::kU4.id()) {
            bool pred = (tv.tid < TILE_N / PACK_RATIO_4BITS) && tile_manager.global_pred;
            if constexpr(HAS_NK_PRED) {
                pred = pred && tv.tid < ((n - tile_manager.tile_start_col * TILE_N) / PACK_RATIO_4BITS);
            }
            FragZeroLoading *gvm_addr = (FragZeroLoading*)zeros_loading + tv.tid;
            ldg_b32_reg_noasync(*((PackType*)&temp_zeros), gvm_addr, pred, true);
        }
    }

    __device__ __forceinline__ void sts_scales() {
        FragScaleLoading *scale_bsm = (FragScaleLoading*)(smem_base + 0x3000) + tv.tid;
        *scale_bsm = temp_scales;
    }

    __device__ __forceinline__ void sts_zeros() {
        if constexpr(w_type_id == sglang::kU4.id()) {
            bool pred = (tv.tid < TILE_N / PACK_RATIO_4BITS) && tile_manager.global_pred;
            if (pred) {
                float temp[PACK_RATIO_4BITS];
                decompress_zero_4bits(temp_zeros, temp);
                float *scale_bsm = (float*)(smem_base + 0x3400) + tv.tid * PACK_RATIO_4BITS;
                for (int i = 0; i < PACK_RATIO_4BITS; i++) {
                    *(scale_bsm + i) = temp[i];
                }
            }
        }
    }

    //TODO: implement when N_ITERS==1 or N_ITERS==3
    __device__ __forceinline__ void lds_scales() {
        if constexpr(N_ITERS==2) {
            *((half2*)local_dequanted_b[0]) = *((half2*)bsm_scales_ptr);
        } else if constexpr(N_ITERS==4) {
            *((PackTypeInt2*)local_dequanted_b[0]) = *((PackTypeInt2*)bsm_scales_ptr);
        }
    }

    __device__ __forceinline__ void pack_scales() {
        if constexpr(w_type_id == sglang::kU4B8.id()) {
            #pragma unroll
            for (int i = 0; i < N_ITERS; i++) {
                float s = local_dequanted_b[0][i];
                float z = -8 * s;
                local_scales[i] = {s, s};
                local_zeros[i] = {z, z};
            }
        } else if constexpr(w_type_id == sglang::kU4.id()) {
            #pragma unroll
            for (int i = 0; i < N_ITERS; i++) {
                float s = local_dequanted_b[0][i];
                float z = *(bsm_zeros_ptr + i);
                z = z * s;
                local_scales[i] = {s, s};
                local_zeros[i] = {z, z};
            }
        } else if constexpr(w_type_id == sglang::kU8B128.id()) {
            #pragma unroll
            for (int i = 0; i < N_ITERS; i++) {
                float s = local_dequanted_b[0][i];
                float z = -128 * s;
                local_scales[i] = {s, s};
                local_zeros[i] = {z, z};
            }
        } else if constexpr(w_type_id == sglang::kU8.id()) {
            // should apply zeros
        }
    }

    __device__ __forceinline__ void dequant(int kdx, int korder = 0) {
        if constexpr(w_type_id == sglang::kU4.id() || w_type_id == sglang::kU4B8.id()) {
            dequant_gptq_4bits<scalar_t>(local_b[kdx], local_dequanted_b[kdx], local_scales[kdx], local_zeros[kdx]);
        } else if constexpr(w_type_id == sglang::kU8.id() || w_type_id == sglang::kU8B128.id()) {
            dequant_gptq_8bits<scalar_t>(local_b[kdx], local_dequanted_b_8bits[kdx][korder], local_scales[kdx], local_zeros[kdx]);
        }
    }

    __device__ __forceinline__ void matmul(int mdx) {
        if constexpr(w_type_id == sglang::kU4.id() || w_type_id == sglang::kU4B8.id()) {
            #pragma unroll
            for (int i = 0; i < N_ITERS; i++) {
                mma_16x16x16<scalar_t>(local_a[mdx][0], *((PackTypeInt2*)local_dequanted_b[i]), *((PackTypeInt4*)output[mdx][i]));
                mma_16x16x16<scalar_t>(local_a[mdx][1], *((PackTypeInt2*)local_dequanted_b[i] + 1), *((PackTypeInt4*)output[mdx][i]));
            }
        } else if constexpr(w_type_id == sglang::kU8.id() || w_type_id == sglang::kU8B128.id()) {
            #pragma unroll
            for (int i = 0; i < N_ITERS; i++) {
                mma_16x16x16<scalar_t>(local_a[mdx][0], *((PackTypeInt2*)local_dequanted_b_8bits[i][0]), *((PackTypeInt4*)output[mdx][i]));
                mma_16x16x16<scalar_t>(local_a[mdx][1], *((PackTypeInt2*)local_dequanted_b_8bits[i][1]), *((PackTypeInt4*)output[mdx][i]));
            }
        }
    }

    __device__ __forceinline__ void clear_c() {
        #pragma unroll
        for (int miter = 0; miter < BLOCKS_M; miter++) {
            #pragma unroll
            for (int niter = 0; niter < N_ITERS; niter++) {
                #pragma unroll
                for (int miter2 = 0; miter2 < 4; miter2++) {
                    output[miter][niter][miter2] = 0;
                }
            }
        }
    }

    //functions for preloading next tile data
    __device__ __forceinline__ void init_address_pre(int _m, int _n, int _k, int _quant_group_power2, int _bidx, int _iters, int _size_atomic_cache, uint8_t *_smem_base) {
        tv.init();
        m = _m;
        n = _n;
        k = _k;
        quant_group_power2 = _quant_group_power2;
        bidx = _bidx;
        size_atomic_cache = _size_atomic_cache;
        smem_base = _smem_base;
        tile_manager.init(m, n, k, bidx, _iters);
        init_tile_pre(tile_manager.tile_start_col, tile_manager.tile_start_row);
    }

     __device__ __forceinline__ void init_tile_pre(int col, int row) {
        //Initialize start slice address and set them to A_loading and B_loading
        int offset_n = col * TILE_N;
        int offset_k = row * TILE_K;
        //A_loading address will always be valid
        A_loading = A + offset_k / (FragACount);
        if constexpr(w_type_id == sglang::kU4.id() || w_type_id == sglang::kU4B8.id()) {
            B_loading = B + (offset_k / PACK_RATIO_4BITS * n + offset_n) / (FragBCount) + tv.slot_idx * n + (tv.wave_idx * SLOT + tv.slot_tid)*N_ITERS;
        } else if constexpr(w_type_id == sglang::kU8.id() || w_type_id == sglang::kU8B128.id()) {
            B_loading = B + (offset_k / PACK_RATIO_8BITS * n + offset_n) / (FragBCount) + tv.slot_idx * n * 2 + (tv.wave_idx * SLOT + tv.slot_tid)*N_ITERS;
        }
        scales_loading = scales + ((offset_k >> quant_group_power2) * n + offset_n) / (sizeof(FragScaleLoading)/sizeof(scalar_t));
        if constexpr(w_type_id == sglang::kU4.id()) {
            zeros_loading = zeros + ((offset_k >> quant_group_power2) * n + offset_n) / PACK_RATIO_4BITS;
        }
    }

    __device__ __forceinline__ void next_tile_pre() {
        tile_manager.next_tile_pre();
        init_tile_pre(tile_manager.tile_start_col, tile_manager.tile_start_row);
    }

    __device__ __forceinline__ void init_bsm_addr() {
        bsm_a_ptr = (PackTypeInt4*)smem_base;           //use 8k bytes, will load at most 32x128*sizeof(half), either m32k128 or m128k32
        //remaining_bsm_ptr = (float*)(smem_base + 0x2000 + 0x1000); //3k bytes
        bsm_a_ptr += tv.slot_tid * (DOUBLE_PAD_SLICE_K / (sizeof(PackTypeInt4) / sizeof(scalar_t))) + tv.slot_idx;
        bsm_scales_ptr = (scalar_t*)(smem_base + 0x3000);      //use 128xsizeof(float)*2 = 1k
        bsm_scales_ptr += (tv.wave_idx * SLOT + tv.slot_tid) * N_ITERS;
        if constexpr(w_type_id == sglang::kU4.id()) {
            bsm_zeros_ptr = (float*)(smem_base + 0x3400);      //use 128xsizeof(float)*2 = 1k
            bsm_zeros_ptr += (tv.wave_idx * SLOT + tv.slot_tid) * N_ITERS;
        }
    }

    __device__ __forceinline__ void write_c(int offset, const float& v) {
        if constexpr(FP32_ATOMIC) {
            atomicAdd(C_temp + offset, v);
        } else {
            atomicAdd(C + offset, (scalar_t)v);
        }
    }

    __device__ __forceinline__ void write_c(int store_m, int n, int store_n, const float& v) {
        if constexpr(USE_ATOMIC_CACHE) {
            //When USE_ATOMIC_CACHE is true, FP32_ATOMIC must be true
            // if (store_m == 0 && store_n == 512) {
            //     printf("Storing m0 n1968, size_atomic_cache=%d,atomic_idx=%d, v = %f, tid = %d, blockIdx = %d, cidx = %d\n", size_atomic_cache, tile_manager.atomic_idx, v, threadIdx.x, blockIdx.x, store_m * n * size_atomic_cache + tile_manager.atomic_idx * n + store_n);
            // }
            // if (store_m * n * size_atomic_cache + tile_manager.atomic_idx * n + store_n == n*11+512) {
            //     printf("!!!Storing m0 n %d, size_atomic_cache=%d,atomic_idx=%d, v = %f, tid = %d, blockIdx = %d, cidx = %d\n", store_n, size_atomic_cache, tile_manager.atomic_idx, v, threadIdx.x, blockIdx.x, store_m * n * size_atomic_cache + tile_manager.atomic_idx * n + store_n);
            // }
            // int offset = store_m * n * size_atomic_cache + tile_manager.atomic_idx * n + store_n;
            // if (offset % 7168 == 0 && offset / 7168 < 11 && !(store_m == 0 && store_n == 0)) {
            //     printf("WARNING tid = %d, bdx = %d, store_m = %d, store_n = %d, offset = %d, n = %d, v = %f\n", threadIdx.x, blockIdx.x, store_m, store_n, offset, n, v);
            // }
            C_temp[store_m * n * size_atomic_cache + tile_manager.atomic_idx * n + store_n] = v;
        } else if constexpr(FP32_ATOMIC) {
            atomicAdd(C_temp + store_m * n + store_n, v);
        } else {
            atomicAdd(C + store_m * n + store_n, (scalar_t)v);
        }
    }

    __device__ __forceinline__ void reset_atomic_cache() {
        if constexpr(USE_ATOMIC_CACHE) {
            // if (blockIdx.x >= 21 && blockIdx.x < 34 && threadIdx.x == 0) {
            //     printf("bdx = %d, tile_start_row = %d, tile_start_col = %d, my_iters = %d, atomic_idx = %d, last_atomic_cache = %d\n",
            //         blockIdx.x, tile_manager.tile_start_row, tile_manager.tile_start_col, tile_manager.my_iters, tile_manager.atomic_idx, tile_manager.last_atomic_cache
            //     );
            // }
            if (!tile_manager.last_atomic_cache) {
                return;
            }
            int k_broad = tv.slot_idx * 4;
            int n_broad = (tv.wave_idx * SLOT + tv.slot_tid) * N_ITERS + tile_manager.tile_start_col * TILE_N;
            #pragma unroll
            for (int miter = 0; miter < BLOCKS_M; miter++) {
                #pragma unroll
                for (int niter = 0; niter < N_ITERS; niter++) {
                    int store_n = n_broad + niter;
                    #pragma unroll
                    for (int miter2 = 0; miter2 < 4; miter2++) {
                        int store_m = k_broad + miter * SLICE_M + miter2;
                        if constexpr(HAS_M_PRED && HAS_NK_PRED) {
                            if (store_m < m && store_n < n) {
                                // write_c(store_m * n + store_n, output[miter][niter][miter2]);
                                //Reset unused atomic cache
                                for (int i = tile_manager.atomic_idx + 1; i < size_atomic_cache; i++) {
                                    C_temp[store_m * n * size_atomic_cache + i * n + store_n] = 0;
                                }
                            }
                        } else if constexpr(HAS_M_PRED) {
                            if (store_m < m) {
                                // write_c(store_m * n + store_n, output[miter][niter][miter2]);
                                //Reset unused atomic cache
                                // if (blockIdx.x >= 21 && blockIdx.x < 34 && threadIdx.x == 0) {
                                //     printf("bdx=%d,tid=%d,atomic_cache_size=%d,atomic_idx=%d,store_m=%d,store_n=%d,reset cache\n", blockIdx.x, threadIdx.x, size_atomic_cache, tile_manager.atomic_idx, store_m, store_n);
                                // }
                                for (int i = tile_manager.atomic_idx + 1; i < size_atomic_cache; i++) {
                                    C_temp[store_m * n * size_atomic_cache + i * n + store_n] = 0;
                                }
                            }
                        } else if constexpr(HAS_NK_PRED) {
                            if (store_n < n) {
                                // write_c(store_m * n + store_n, output[miter][niter][miter2]);
                                //Reset unused atomic cache
                                for (int i = tile_manager.atomic_idx + 1; i < size_atomic_cache; i++) {
                                    C_temp[store_m * n * size_atomic_cache + i * n + store_n] = 0;
                                }
                            }
                        } else {
                            // write_c(store_m * n + store_n, output[miter][niter][miter2]);
                            //Reset unused atomic cache
                            for (int i = tile_manager.atomic_idx + 1; i < size_atomic_cache; i++) {
                                C_temp[store_m * n * size_atomic_cache + i * n + store_n] = 0;
                            }
                        }
                    }
                }
            }
        }
    }

    //atomic write to c
    __device__ __forceinline__ void write_c_pre() {
        int k_broad = tv.slot_idx * 4;
        int n_broad = (tv.wave_idx * SLOT + tv.slot_tid) * N_ITERS + tile_manager.tile_start_col_cache * TILE_N;
        #pragma unroll
        for (int miter = 0; miter < BLOCKS_M; miter++) {
            #pragma unroll
            for (int niter = 0; niter < N_ITERS; niter++) {
                int store_n = n_broad + niter;
                #pragma unroll
                for (int miter2 = 0; miter2 < 4; miter2++) {
                    int store_m = k_broad + miter * SLICE_M + miter2;
                    if constexpr(HAS_M_PRED && HAS_NK_PRED) {
                        if (store_m < m && store_n < n) {
			                // write_c(store_m * n + store_n, output[miter][niter][miter2]);
                            write_c(store_m, n, store_n, output[miter][niter][miter2]);
                        }
                    } else if constexpr(HAS_M_PRED) {
                        if (store_m < m) {
			                // write_c(store_m * n + store_n, output[miter][niter][miter2]);
                            write_c(store_m, n, store_n, output[miter][niter][miter2]);
                        }
                    } else if constexpr(HAS_NK_PRED) {
                        if (store_n < n) {
			                // write_c(store_m * n + store_n, output[miter][niter][miter2]);
                            write_c(store_m, n, store_n, output[miter][niter][miter2]);
                        }
                    } else {
			            // write_c(store_m * n + store_n, output[miter][niter][miter2]);
                        write_c(store_m, n, store_n, output[miter][niter][miter2]);
                    }
                }
            }
        }
        tile_manager.reset_atomic_idx();
    }

    __device__ __forceinline__ void write_c_pre(const float& s) {
        int k_broad = tv.slot_idx * 4;
        int n_broad = (tv.wave_idx * SLOT + tv.slot_tid) * N_ITERS + tile_manager.tile_start_col_cache * TILE_N;
        #pragma unroll
        for (int miter = 0; miter < BLOCKS_M; miter++) {
            #pragma unroll
            for (int niter = 0; niter < N_ITERS; niter++) {
                int store_n = n_broad + niter;
                #pragma unroll
                for (int miter2 = 0; miter2 < 4; miter2++) {
                    int store_m = k_broad + miter * SLICE_M + miter2;
                    if constexpr(HAS_M_PRED && HAS_NK_PRED) {
                        if (store_m < m && store_n < n) {
                            // write_c(store_m * n + store_n, output[miter][niter][miter2] * s);
                            write_c(store_m, n, store_n, output[miter][niter][miter2]*s);
			            }
                    } else if constexpr(HAS_M_PRED) {
                        if (store_m < m) {
                            // if (threadIdx.x == 0)
                            //     printf("bidx = %d, bidy = %d, tid = %d, will write value %f x %f\n", blockIdx.x, blockIdx.y, threadIdx.x, output[miter][niter][miter2], s);
                            // write_c(store_m * n + store_n, output[miter][niter][miter2] * s);
                            write_c(store_m, n, store_n, output[miter][niter][miter2]*s);
			            }
                    } else if constexpr(HAS_NK_PRED) {
                        if (store_n < n) {
                            // write_c(store_m * n + store_n, output[miter][niter][miter2] * s);
                            write_c(store_m, n, store_n, output[miter][niter][miter2]*s);
			            }
                    } else {
                        // write_c(store_m * n + store_n, output[miter][niter][miter2] * s);
                        write_c(store_m, n, store_n, output[miter][niter][miter2]*s);
		            }
                }
            }
        }
        tile_manager.reset_atomic_idx();
    }

    template<int K, bool KTAIL>
    __device__ __forceinline__ void on_sts_a() {
        if constexpr(K == 0) {
            sts_a(); // reorder data_a from reg(gvm resouce) to bsm
        }
    }

    template<int K, bool KTAIL>
    __device__ __forceinline__ void on_preload(int k_idx) {
        if constexpr(!KTAIL) {
            if constexpr(K == 0) {
                next_k0_pre(); // preload gvm a/b
            } else {
                next_k1_pre(); // preload gvm a/b
            }
            ldg_b(k_idx + K + 1); //preload b for next k
            if constexpr(K == 1) {
                ldg_a(k_idx + K + 1); //preload a for next k
            }
        } else {
            next_tile_pre();
            ldg_scales();
            ldg_b(0);
            ldg_a(0);
        }
    }

    template<int K, bool KTAIL>
    __device__ __forceinline__ void on_dequant_niters1(int k_idx) {
        if constexpr(w_type_id == sglang::kU4.id() || w_type_id == sglang::kU4B8.id()) {
            swap_b_cache(0);
            on_sts_a<K,KTAIL>(); // reorder data_a from reg(gvm resouce) to bsm
            on_preload<K,KTAIL>(k_idx);
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(0); //dequant b0
        } else if constexpr(w_type_id == sglang::kU8.id() || w_type_id == sglang::kU8B128.id()) {
            swap_b_cache(0);
            ldg_b(0, 1);
            dequant(0, 0);
            on_sts_a<K,KTAIL>(); // reorder data_a from reg(gvm resouce) to bsm
            swap_b_cache(0);
            on_preload<K,KTAIL>(k_idx);
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(0, 1); //dequant b0
        }
    }

    template<int K, bool KTAIL>
    __device__ __forceinline__ void on_dequant_niters2(int k_idx) {
        if constexpr(w_type_id == sglang::kU4.id() || w_type_id == sglang::kU4B8.id()) {
            swap_b_cache(0);
            dequant(0); //dequant b0
            on_sts_a<K,KTAIL>();
            swap_b_cache(1);
            on_preload<K,KTAIL>(k_idx);
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(1);   //dequant b64
        } else if constexpr(w_type_id == sglang::kU8.id() || w_type_id == sglang::kU8B128.id()) {
            swap_b_cache(0);
            swap_b_cache(1);
            ldg_b(0, 1);
            dequant(0); //dequant b0
            on_sts_a<K,KTAIL>();
            dequant(1);
            swap_b_cache(0);
            swap_b_cache(1);
            on_preload<K,KTAIL>(k_idx);
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(0, 1);   //dequant b64
            dequant(1, 1);   //dequant b64
        }
    }

    template<int K, bool KTAIL>
    __device__ __forceinline__ void on_dequant_niters3(int k_idx) {
        if constexpr(w_type_id == sglang::kU4.id() || w_type_id == sglang::kU4B8.id()) {
            swap_b_cache(0);
            dequant(0); //dequant b0
            on_sts_a<K,KTAIL>();
            swap_b_cache(1);
            dequant(1);
            swap_b_cache(2);
            on_preload<K,KTAIL>(k_idx);
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(2); //dequant b0
        } else if constexpr(w_type_id == sglang::kU8.id() || w_type_id == sglang::kU8B128.id()) {
            swap_b_cache(0);
            swap_b_cache(1);
            swap_b_cache(2);
            ldg_b(0, 1);
            dequant(0); //dequant b0
            on_sts_a<K,KTAIL>();
            swap_b_cache(1);
            dequant(1);
            dequant(2);
            swap_b_cache(0);
            swap_b_cache(1);
            swap_b_cache(2);
            on_preload<K,KTAIL>(k_idx);
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(0, 1); //dequant b0
            dequant(1, 1); //dequant b0
            dequant(2, 1); //dequant b0
        }
    }

    template<int K, bool KTAIL>
    __device__ __forceinline__ void on_dequant_niters4(int k_idx) {
        if constexpr(w_type_id == sglang::kU4.id() || w_type_id == sglang::kU4B8.id()) {
            swap_b_cache(0);
            dequant(0); //dequant b0
            on_sts_a<K,KTAIL>();
            swap_b_cache(1);
            dequant(1); //dequant b1
            swap_b_cache(2);
            swap_b_cache(3);
            dequant(2); //dequant b2
            on_preload<K,KTAIL>(k_idx);
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(3); //dequant b3
        } else if constexpr(w_type_id == sglang::kU8.id() || w_type_id == sglang::kU8B128.id()) {
            swap_b_cache(0);
            swap_b_cache(1);
            swap_b_cache(2);
            swap_b_cache(3);
            ldg_b(0, 1);
            dequant(0); //dequant b0
            on_sts_a<K,KTAIL>();
            dequant(1); //dequant b1
            dequant(2); //dequant b2
            dequant(3); //dequant b2
            swap_b_cache(0);
            swap_b_cache(1);
            swap_b_cache(2);
            swap_b_cache(3);
            on_preload<K,KTAIL>(k_idx);
            barrier_bsm; //sync threads for sts_a
            lds_a(0); //load first 16 batches into registers
            dequant(0, 1); //dequant b0
            dequant(1, 1); //dequant b1
            dequant(2, 1); //dequant b2
            dequant(3, 1); //dequant b3
        }
    }

    template<int K, bool KTAIL>
    __device__ __forceinline__ void on_dequant(int kdx) {
        if constexpr(N_ITERS == 1) on_dequant_niters1<K,KTAIL>(kdx);
        else if constexpr(N_ITERS == 2) on_dequant_niters2<K,KTAIL>(kdx);
        else if constexpr(N_ITERS == 3) on_dequant_niters3<K,KTAIL>(kdx);
        else if constexpr(N_ITERS == 4) on_dequant_niters4<K,KTAIL>(kdx);
    }
};

template<typename scalar_t,
    const sglang::ScalarTypeId w_type_id,
    const int THREADS,          // number of threads in a threadblock
    const int BLOCKS_M,         // number of 16x16 blocks in the m
                                // dimension (batchsize) of the
                                // threadblock
    const int BLOCKS_N,         // same for n dimension (output)
    const int BLOCKS_K,         // same for k dimension (reduction)
    const bool HAS_ACT_ORDER,   // whether act_order is enabled
    const bool HAS_ZP,          // whether zero-points are enabled
    const bool HAS_M_PRED = true,  //If we should use predictors to load m from gvm
    const bool HAS_NK_PRED = true,  //If we should use predictors to load nk from gvm
    const bool FP32_ATOMIC = true,
    const bool USE_ATOMIC_CACHE = true,
    const bool HAS_SIZEM_PTR = true
    >
__global__ void hgemm_gptq(
    const PackTypeInt4* __restrict__ A,  // fp16 input matrix of shape mxk
    const PackTypeInt4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    PackTypeInt4* __restrict__ C,        // fp16 output buffer of shape mxn
    PackTypeInt4* __restrict__ C_tmp,    // fp32 tmp output buffer (for reduce)
    const int* size_m_ptr,               // real prob_m that need be caculated
    const PackTypeInt4* __restrict__ scales_ptr,  // fp16 quantization scales of shape
                                          // (k/groupsize)xn
    const PackTypeInt4* __restrict__ zp_ptr,      // 4bit packed zero-points of shape
                                          // (k/groupsize)x(n/pack_factor)
    const int* __restrict__ g_idx,        // int32 group indices of shape k
    int prob_m,           // batch dimension m
    int prob_n,           // output dimension n
    int prob_k,           // reduction dimension k
    int quant_group_power2,
    int max_iters,        // max tile iterations for one block
    int size_atomic_cache,
    int* locks,           // extra global storage for barrier synchronization
    bool use_fp32_reduce  // whether to use fp32 global reduce
) {
    int bidx = blockIdx.x;
    int bidy = blockIdx.y; //Indicates index of FULL_M_BLOCKS
    if constexpr (HAS_SIZEM_PTR) {
        int real_prob_m = *size_m_ptr;
        //discard data if current m larger than real_prob_m
        if (real_prob_m <= blockIdx.y * (MAX_BLOCKS_M * SLICE_M)) return;
        prob_m = real_prob_m;
    }
    __shared__ uint8_t smem_base[0x4000]; //4x16x256 = 16Kbytes
    using LoadingManagerType = LoadingManager<scalar_t, w_type_id, THREADS, BLOCKS_M, BLOCKS_N, BLOCKS_K, HAS_ACT_ORDER, HAS_ZP, HAS_M_PRED, HAS_NK_PRED, FP32_ATOMIC, USE_ATOMIC_CACHE>;
    LoadingManagerType loading_manager;
    A += (uint64_t)blockIdx.y * (MAX_BLOCKS_M * SLICE_M) * prob_k / (sizeof(PackTypeInt4) / sizeof(scalar_t));
    C += (uint64_t)blockIdx.y * (MAX_BLOCKS_M * SLICE_M) * prob_n / (sizeof(PackTypeInt4) / sizeof(scalar_t));
    if constexpr(USE_ATOMIC_CACHE) {
        C_tmp += (uint64_t)blockIdx.y * (MAX_BLOCKS_M * SLICE_M) * prob_n * size_atomic_cache / (sizeof(PackTypeInt4) / sizeof(float));
    } else if constexpr(FP32_ATOMIC) {
        C_tmp += (uint64_t)blockIdx.y * (MAX_BLOCKS_M * SLICE_M) * prob_n / (sizeof(PackTypeInt4) / sizeof(float));
    }
    loading_manager.set_address(A, B, C, C_tmp, scales_ptr, zp_ptr);
    //loading_manager.init_address(prob_m, prob_n, prob_k, bidx, max_iters, smem_base);
    loading_manager.init_address_pre(std::min<int>(MAX_BLOCKS_M*SLICE_M, prob_m - blockIdx.y * (MAX_BLOCKS_M * SLICE_M)), prob_n, prob_k, quant_group_power2, bidx, max_iters, size_atomic_cache, smem_base);

    loading_manager.reset_atomic_cache();

    loading_manager.ldg_scales(); //Load all scales to bsm
    loading_manager.ldg_zp();
    loading_manager.ldg_b(0);    //load b in k0~31
    loading_manager.ldg_a(0);    //Load first k0~63 and all m
    loading_manager.clear_c();

    while (max_iters > 0) {
        loading_manager.init_bsm_addr(); //reset all bsm address for current tile
        loading_manager.sts_scales();
        loading_manager.sts_zeros();
        barrier_bsm;
        loading_manager.lds_scales(); //load scale0 and scale64
        loading_manager.pack_scales(); //pack scales into two v2f structure

        int k_idx = 0;
        if constexpr(BLOCKS_K / 2 - 1 > 0) {
            #pragma unroll BLOCKS_K / 2 - 1
            for (int kloop = 0; kloop < BLOCKS_K / 2 - 1; kloop++) {
                int m_idx = 0;
                loading_manager.template on_dequant<0, false>(k_idx);
                if constexpr(BLOCKS_M > 1) {
                    #pragma unroll BLOCKS_M - 1
                    for (; m_idx < BLOCKS_M - 1; m_idx++) {
                        loading_manager.lds_a(m_idx+1); //pre load next 16 batches
                        loading_manager.matmul(m_idx); //do matmul
                    }
                }
                barrier_bsm; //wait for the last 16 batches ready, becasue of k_idx will be change
                loading_manager.next_k1(); //modify gvm/bsm address of a and b
                loading_manager.matmul(m_idx); //do matmul
                m_idx = 0;
                loading_manager.template on_dequant<1, false>(k_idx);
                if constexpr(BLOCKS_M > 1) {
                    #pragma unroll BLOCKS_M - 1
                    for (; m_idx < BLOCKS_M - 1; m_idx++) {
                        loading_manager.lds_a(m_idx+1); //pre load next 16 batches
                        loading_manager.matmul(m_idx); //do matmul
                    }
                }
                barrier_bsm; //wait for the last 16 batches ready, becasue of k_idx will be change
                loading_manager.next_k0(); //modify gvm/bsm address of a and b
                loading_manager.matmul(m_idx); //do matmul
                k_idx += 2;
            }
        }

        int m_idx = 0;
        loading_manager.template on_dequant<0, false>(k_idx);
        if constexpr(BLOCKS_M > 1) {
            #pragma unroll BLOCKS_M - 1
            for (; m_idx < BLOCKS_M - 1; m_idx++) {
                loading_manager.lds_a(m_idx+1); //pre load next 16 batches
                loading_manager.matmul(m_idx); //do matmul
            }
        }
        barrier_bsm; //wait for the last 16 batches ready, becasue of k_idx will be change
        loading_manager.next_k1(); //modify gvm/bsm address of a and b
        loading_manager.matmul(m_idx); //do matmul
        m_idx = 0;
        loading_manager.template on_dequant<1, true>(k_idx);
        if constexpr(BLOCKS_M > 1) {
            #pragma unroll BLOCKS_M - 1
            for (; m_idx < BLOCKS_M - 1; m_idx++) {
                loading_manager.lds_a(m_idx+1); //pre load next 16 batches
                loading_manager.matmul(m_idx); //do matmul
            }
        }
        barrier_bsm; //wait for the last 16 batches ready, becasue of k_idx will be change
        //loading_manager.next_tile_pre(); should move into on_dequant ?
        loading_manager.matmul(m_idx); //do matmul

        max_iters--;

        if (loading_manager.tile_manager.need_save_data_pre()) {
            loading_manager.write_c_pre(); // reduce and write back
            loading_manager.clear_c();
        }

        barrier_bsm;
    }
}

//As we need to use CUDA_GRAPH, for particular data, we should keep the grid and block, input shape and output shape the same.
//Input: gate/up projs, with MAX_EXPERTS * prob_k * prob_k
//Input: selected_experts: int array indicates what expert is selected
//Input: experts_range: range that <= MAX_EXPERTS, if selected_expert >= expert_range, directly return
//Input: experts_per_token: how many experts should be selected in whole moe
//Input: origin_m: the original batch size, used for output offset
//Output: 2xexperts_per_token*origin_m*prob_n
//        where output[:, experts_count_in_range:experts_selected_per_token, :, :] are dirty data
template<typename scalar_t, typename scalar_output_t,
    const sglang::ScalarTypeId w_type_id,
    const int THREADS,          // number of threads in a threadblock
    const int BLOCKS_M,         // number of 16x16 blocks in the m
                                // dimension (batchsize) of the
                                // threadblock
    const int BLOCKS_N,         // same for n dimension (output)
    const int BLOCKS_K,         // same for k dimension (reduction)
    const bool HAS_ACT_ORDER,   // whether act_order is enabled
    const bool HAS_ZP,          // whether zero-points are enabled
    const bool HAS_M_PRED = true,  //If we should use predictors to load m from gvm
    const bool HAS_NK_PRED = true,  //If we should use predictors to load nk from gvm
    const bool FP32_ATOMIC = false,
    const bool USE_ATOMIC_CACHE = false
    >
__global__ void hgemm_gptq_fused_moe(
    const scalar_t* __restrict__ hidden_states,
    const PackType* __restrict__ gate_projs,
    const scalar_t* __restrict__ gate_projs_scales,
    const PackType* __restrict__ up_projs,
    const scalar_t* __restrict__ up_projs_scales,
    const int64_t* __restrict__ selected_experts,
    PackTypeInt4* C, PackTypeInt4* C_tmp, int experts_range, int experts_per_token,
    int origin_m, int prob_m, int prob_n, int prob_k, int max_iters, int size_atomic_cache, int quant_group_power2
) {
    int bidx = blockIdx.x; //Indicates which tile it is
    int bidy = blockIdx.y; //Indicates batch size
    int bidz = blockIdx.z; //Indicates projection matrix index and experts
    __shared__ uint8_t smem_base[0x4000]; //4x16x256 = 16Kbytes
    int seq_idx = bidy;
    //First experts_per_token blockz means blocks for gate projections
    //Second experts_per_token blockz means blocks for up projections
    int expert_rank = bidz >= experts_per_token ? 1 : 0;
    int expert_idx = bidz >= experts_per_token ? bidz - experts_per_token : bidz;
    int expert_offset = expert_idx + experts_per_token * seq_idx;
    int64_t expert = selected_experts[expert_offset]; // get out which expert these blocks should process
    //Sometimes, we may not have all experts stored on gpu
    if (expert >= experts_range) return;
    //Calculate offset b
    int64_t offset_b = 0, offset_b_scale = ((expert * prob_n * prob_k) >> quant_group_power2);
    if constexpr(w_type_id == sglang::kU4.id() || w_type_id == sglang::kU4B8.id()) {
        offset_b = expert * prob_n * prob_k / PACK_RATIO_4BITS;
    } else if constexpr(w_type_id == sglang::kU8B128.id() || w_type_id == sglang::kU8.id()) {
        offset_b = expert * prob_n * prob_k / PACK_RATIO_8BITS;
    }
    //offset to corresponding hidden_states
    int64_t offset_a = (int64_t)seq_idx * prob_m * prob_k / (sizeof(PackTypeInt4) / sizeof(scalar_t));
    int32_t _c_offset_experts = expert_rank * (origin_m * experts_per_token) + seq_idx * experts_per_token + expert_idx;

    uint64_t offset_c = (uint64_t)_c_offset_experts * prob_n / (sizeof(PackTypeInt4) / sizeof(scalar_output_t));
    const PackType* _B = bidz >= experts_per_token ? up_projs +  offset_b : gate_projs + offset_b;
    const scalar_t* _scales_ptr = bidz >= experts_per_token ? up_projs_scales + offset_b_scale: gate_projs_scales + offset_b_scale;

    const PackTypeInt4* A = (const PackTypeInt4*)hidden_states;
    const PackTypeInt4* B = (const PackTypeInt4*)_B;
    C += offset_c;
    if constexpr(USE_ATOMIC_CACHE) {
        C_tmp += (uint64_t)_c_offset_experts * prob_n * size_atomic_cache / (sizeof(PackTypeInt4) / sizeof(float));
    } else if constexpr(FP32_ATOMIC) {
        C_tmp += (uint64_t)_c_offset_experts * prob_n / (sizeof(PackTypeInt4) / sizeof(float));
    }
    A += offset_a;
    const PackTypeInt4* scales_ptr = (const PackTypeInt4*)_scales_ptr;
    const PackTypeInt4* zp_ptr = nullptr;
    //PackTypeInt4* C_tmp = nullptr;

    using LoadingManagerType = LoadingManager<scalar_t, w_type_id, THREADS, BLOCKS_M, BLOCKS_N, BLOCKS_K, HAS_ACT_ORDER, HAS_ZP, HAS_M_PRED, HAS_NK_PRED, FP32_ATOMIC, USE_ATOMIC_CACHE>;
    LoadingManagerType loading_manager;
    loading_manager.set_address(A, B, C, C_tmp, scales_ptr, zp_ptr);
    loading_manager.init_address_pre(prob_m, prob_n, prob_k, quant_group_power2, bidx, max_iters, size_atomic_cache, smem_base);

    loading_manager.ldg_scales(); //Load all scales to bsm
    loading_manager.ldg_zp();
    loading_manager.ldg_b(0);    //load b in k0~31
    loading_manager.ldg_a(0);    //Load first k0~63 and all m
    loading_manager.clear_c();

    while (max_iters > 0) {
        loading_manager.init_bsm_addr(); //reset all bsm address for current tile
        loading_manager.sts_scales();
        loading_manager.sts_zeros();
        barrier_bsm;
        loading_manager.lds_scales(); //load scale0 and scale64
        loading_manager.pack_scales(); //pack scales into two v2f structure

        int k_idx = 0;
        if constexpr(BLOCKS_K / 2 - 1 > 0) {
            #pragma unroll BLOCKS_K / 2 - 1
            for (int kloop = 0; kloop < BLOCKS_K / 2 - 1; kloop++) {
                int m_idx = 0;
                loading_manager.template on_dequant<0, false>(k_idx);
                if constexpr(BLOCKS_M > 1) {
                    #pragma unroll BLOCKS_M - 1
                    for (; m_idx < BLOCKS_M - 1; m_idx++) {
                        loading_manager.lds_a(m_idx+1); //pre load next 16 batches
                        loading_manager.matmul(m_idx); //do matmul
                    }
                }
                barrier_bsm; //wait for the last 16 batches ready, becasue of k_idx will be change
                loading_manager.next_k1(); //modify gvm/bsm address of a and b
                loading_manager.matmul(m_idx); //do matmul
                m_idx = 0;
                loading_manager.template on_dequant<1, false>(k_idx);
                if constexpr(BLOCKS_M > 1) {
                    #pragma unroll BLOCKS_M - 1
                    for (; m_idx < BLOCKS_M - 1; m_idx++) {
                        loading_manager.lds_a(m_idx+1); //pre load next 16 batches
                        loading_manager.matmul(m_idx); //do matmul
                    }
                }
                barrier_bsm; //wait for the last 16 batches ready, becasue of k_idx will be change
                loading_manager.next_k0(); //modify gvm/bsm address of a and b
                loading_manager.matmul(m_idx); //do matmul
                k_idx += 2;
            }
        }

        int m_idx = 0;
        loading_manager.template on_dequant<0, false>(k_idx);
        if constexpr(BLOCKS_M > 1) {
            #pragma unroll BLOCKS_M - 1
            for (; m_idx < BLOCKS_M - 1; m_idx++) {
                loading_manager.lds_a(m_idx+1); //pre load next 16 batches
                loading_manager.matmul(m_idx); //do matmul
            }
        }
        barrier_bsm; //wait for the last 16 batches ready, becasue of k_idx will be change
        loading_manager.next_k1(); //modify gvm/bsm address of a and b
        loading_manager.matmul(m_idx); //do matmul
        m_idx = 0;
        loading_manager.template on_dequant<1, true>(k_idx);
        if constexpr(BLOCKS_M > 1) {
            #pragma unroll BLOCKS_M - 1
            for (; m_idx < BLOCKS_M - 1; m_idx++) {
                loading_manager.lds_a(m_idx+1); //pre load next 16 batches
                loading_manager.matmul(m_idx); //do matmul
            }
        }
        barrier_bsm; //wait for the last 16 batches ready, becasue of k_idx will be change
        //loading_manager.next_tile_pre(); should move into on_dequant ?
        loading_manager.matmul(m_idx); //do matmul

        max_iters--;

        if (loading_manager.tile_manager.need_save_data_pre()) {
            loading_manager.write_c_pre(); // reduce and write back
            loading_manager.clear_c();
        }

        barrier_bsm;
    }
}

template<typename scalar_t, typename scalar_output_t,
    const sglang::ScalarTypeId w_type_id,
    const int THREADS,          // number of threads in a threadblock
    const int BLOCKS_M,         // number of 16x16 blocks in the m
                                // dimension (batchsize) of the
                                // threadblock
    const int BLOCKS_N,         // same for n dimension (output)
    const int BLOCKS_K,         // same for k dimension (reduction)
    const bool HAS_ACT_ORDER,   // whether act_order is enabled
    const bool HAS_ZP,          // whether zero-points are enabled
    const bool HAS_M_PRED = true,  //If we should use predictors to load m from gvm
    const bool HAS_NK_PRED = true,  //If we should use predictors to load nk from gvm
    const bool FP32_ATOMIC = false,
    const bool USE_ATOMIC_CACHE = false
    >
__global__ void hgemm_gptq_fused_moe_multi_scaled(
    const scalar_t* __restrict__ hidden_states,  //every experts have one hidden_states, aka 8x2048
    const PackType* __restrict__ down_projs,     //256x2048x7168, actually, only 8x2048x7168 is calculated, we calculate 8 times 1x2048 @ 2048x7168 gemm
    const scalar_t* __restrict__ down_projs_scales,
    const int64_t* __restrict__ selected_experts,
    const scalar_t* __restrict__ routing_weights,
    PackTypeInt4* C, PackTypeInt4* C_tmp, int experts_range, int experts_per_token,
    int origin_m, int prob_m, int prob_n, int prob_k, int max_iters, int size_atomic_cache, int quant_group_power2
) {
    int bidx = blockIdx.x; //Indicates which tile it is
    int bidy = blockIdx.y; //Indicates seq idx
    int bidz = blockIdx.z; //Indicates which expert it is, 0~7
    __shared__ uint8_t smem_base[0x4000]; //4x16x256 = 16Kbytes
    int seq_idx = bidy;
    int expert_idx = bidz;
    int expert_offset = seq_idx * experts_per_token + expert_idx;
    int64_t expert = selected_experts[expert_offset]; // get out which expert these blocks should process
    if (expert >= experts_range) return;
    const scalar_t* _A = hidden_states + seq_idx * experts_per_token * prob_m * prob_k + expert_idx * prob_m * prob_k; //m is always 1
    int64_t offset_b = 0;
    if constexpr(w_type_id == sglang::kU4.id() || w_type_id == sglang::kU4B8.id()) {
        offset_b = expert * prob_n * prob_k / PACK_RATIO_4BITS;
    } else if constexpr(w_type_id == sglang::kU8B128.id() || w_type_id == sglang::kU8.id()) {
        offset_b = expert * prob_n * prob_k / PACK_RATIO_8BITS;
    }
    const PackType* _B = down_projs + offset_b;
    const scalar_t* _scales_ptr = down_projs_scales + ((expert * prob_n * prob_k) >> quant_group_power2);
    using LoadingManagerType = LoadingManager<scalar_t, w_type_id, THREADS, BLOCKS_M, BLOCKS_N, BLOCKS_K, HAS_ACT_ORDER, HAS_ZP, HAS_M_PRED, HAS_NK_PRED, FP32_ATOMIC, USE_ATOMIC_CACHE>;
    LoadingManagerType loading_manager;
    const PackTypeInt4* A = (const PackTypeInt4*)_A;
    const PackTypeInt4* B = (const PackTypeInt4*)_B;
    C += seq_idx * prob_m * prob_n / (sizeof(PackTypeInt4) / sizeof(scalar_output_t));
    if constexpr(USE_ATOMIC_CACHE) {
        C_tmp += seq_idx * prob_m * prob_n * size_atomic_cache / (sizeof(PackTypeInt4) / sizeof(float));
    } else if constexpr(FP32_ATOMIC) {
        C_tmp += seq_idx * prob_m * prob_n / (sizeof(PackTypeInt4) / sizeof(float));
    }
    const PackTypeInt4* scales_ptr = (const PackTypeInt4*)_scales_ptr;
    const PackTypeInt4* zp_ptr = nullptr;
    //PackTypeInt4* C_tmp = nullptr;
    float routing_weight = (float)routing_weights[expert_offset];

    loading_manager.set_address(A, B, C, C_tmp, scales_ptr, zp_ptr);
    loading_manager.init_address_pre(prob_m, prob_n, prob_k, quant_group_power2, bidx, max_iters, size_atomic_cache, smem_base);

    loading_manager.ldg_scales(); //Load all scales to bsm
    loading_manager.ldg_zp();
    loading_manager.ldg_b(0);    //load b in k0~31
    loading_manager.ldg_a(0);    //Load first k0~63 and all m
    loading_manager.clear_c();

    while (max_iters > 0) {
        loading_manager.init_bsm_addr(); //reset all bsm address for current tile
        loading_manager.sts_scales();
        loading_manager.sts_zeros();
        barrier_bsm;
        loading_manager.lds_scales(); //load scale0 and scale64
        loading_manager.pack_scales(); //pack scales into two v2f structure

        int k_idx = 0;
        if constexpr(BLOCKS_K / 2 - 1 > 0) {
            #pragma unroll BLOCKS_K / 2 - 1
            for (int kloop = 0; kloop < BLOCKS_K / 2 - 1; kloop++) {
                int m_idx = 0;
                loading_manager.template on_dequant<0, false>(k_idx);
                if constexpr(BLOCKS_M > 1) {
                    #pragma unroll BLOCKS_M - 1
                    for (; m_idx < BLOCKS_M - 1; m_idx++) {
                        loading_manager.lds_a(m_idx+1); //pre load next 16 batches
                        loading_manager.matmul(m_idx); //do matmul
                    }
                }
                barrier_bsm; //wait for the last 16 batches ready, becasue of k_idx will be change
                loading_manager.next_k1(); //modify gvm/bsm address of a and b
                loading_manager.matmul(m_idx); //do matmul
                m_idx = 0;
                loading_manager.template on_dequant<1, false>(k_idx);
                if constexpr(BLOCKS_M > 1) {
                    #pragma unroll BLOCKS_M - 1
                    for (; m_idx < BLOCKS_M - 1; m_idx++) {
                        loading_manager.lds_a(m_idx+1); //pre load next 16 batches
                        loading_manager.matmul(m_idx); //do matmul
                    }
                }
                barrier_bsm; //wait for the last 16 batches ready, becasue of k_idx will be change
                loading_manager.next_k0(); //modify gvm/bsm address of a and b
                loading_manager.matmul(m_idx); //do matmul
                k_idx += 2;
            }
        }

        int m_idx = 0;
        loading_manager.template on_dequant<0, false>(k_idx);
        if constexpr(BLOCKS_M > 1) {
            #pragma unroll BLOCKS_M - 1
            for (; m_idx < BLOCKS_M - 1; m_idx++) {
                loading_manager.lds_a(m_idx+1); //pre load next 16 batches
                loading_manager.matmul(m_idx); //do matmul
            }
        }
        barrier_bsm; //wait for the last 16 batches ready, becasue of k_idx will be change
        loading_manager.next_k1(); //modify gvm/bsm address of a and b
        loading_manager.matmul(m_idx); //do matmul
        m_idx = 0;
        loading_manager.template on_dequant<1, true>(k_idx);
        if constexpr(BLOCKS_M > 1) {
            #pragma unroll BLOCKS_M - 1
            for (; m_idx < BLOCKS_M - 1; m_idx++) {
                loading_manager.lds_a(m_idx+1); //pre load next 16 batches
                loading_manager.matmul(m_idx); //do matmul
            }
        }
        barrier_bsm; //wait for the last 16 batches ready, becasue of k_idx will be change
        //loading_manager.next_tile_pre(); should move into on_dequant ?
        loading_manager.matmul(m_idx); //do matmul

        max_iters--;

        if (loading_manager.tile_manager.need_save_data_pre()) {
            loading_manager.write_c_pre(routing_weight); // reduce and write back
            loading_manager.clear_c();
        }

        barrier_bsm;
    }
}

} //end of namespace __hgemm_even_blocks_k

template<typename scalar_t,
    const sglang::ScalarTypeId w_type_id,
    int THREADS,
    int BLOCKS_M,
    int BLOCKS_N,
    int BLOCKS_K,
    bool HAS_ACT_ORDER,
    bool HAS_ZP,
    bool HAS_M_PRED,
    bool HAS_NK_PRED,
    bool FP32_ATOMIC,
    bool USE_ATOMIC_CACHE>
bool launch_gemm_gptq_kernel(const PackTypeInt4* A,
    const PackTypeInt4* B,
    PackTypeInt4* C,
    PackTypeInt4* C_temp,
    const int* size_m_ptr,
    const PackTypeInt4* scales,
    const PackTypeInt4* zeros,
    int* g_idx, int m, int n, int k, int quant_group, int chunks, int size_atomic_cache, cudaStream_t stream = nullptr) {
    int tiles_m = div_ceil(m, TILE_M);
    int tiles_n = div_ceil(n, TILE_N);
    int tiles_k = div_ceil(k, TILE_K);
    if (TILE_K > quant_group && TILE_K % quant_group != 0) {
        printf("Invalid TILE_K %d that can not be dived by QUANT_GROUP %d\n", TILE_K, quant_group);
        return false;
    }

    int total_tiles = tiles_n * tiles_k;
    int blocks = PEUS;
    int iters = div_ceil(total_tiles, PEUS);
    if (total_tiles < PEUS) {
        if (TILE_K < quant_group) {
            iters = quant_group / TILE_K;
            blocks = div_ceil(total_tiles, iters);
        } else {
            iters = 1;
            blocks = total_tiles;
        }
    } else {
        if (TILE_K < quant_group) {
            iters = div_ceil(iters, quant_group / TILE_K) * quant_group / TILE_K;
            blocks = div_ceil(total_tiles, iters);
        }
    }
    while (iters * blocks - total_tiles >= iters) {
        blocks -= 1;
    }

    if (total_tiles < blocks) {
        printf("total slice %d < blocks %d, Invalid configure\n", total_tiles, blocks);
        return false;
    }


    //It is better to do perm before launch kernel
    if (size_m_ptr == nullptr) {
        if constexpr(BLOCKS_K % 2 == 1) {
            __hgemm_singular_blocks_k::hgemm_gptq<scalar_t,
                w_type_id,
                THREADS,
                BLOCKS_M, BLOCKS_N, BLOCKS_K,
                HAS_ACT_ORDER, HAS_ZP, HAS_M_PRED, HAS_NK_PRED, FP32_ATOMIC, USE_ATOMIC_CACHE, false
                ><<<dim3(blocks, chunks, 1), THREADS, 0, stream>>>(A, B, C, C_temp, size_m_ptr, scales, zeros, g_idx, m, n, k, get_power2(quant_group), iters, size_atomic_cache, nullptr, false);
        } else {
            __hgemm_even_blocks_k::hgemm_gptq<scalar_t,
                w_type_id,
                THREADS,
                BLOCKS_M, BLOCKS_N, BLOCKS_K,
                HAS_ACT_ORDER, HAS_ZP, HAS_M_PRED, HAS_NK_PRED, FP32_ATOMIC, USE_ATOMIC_CACHE, false
                ><<<dim3(blocks, chunks, 1), THREADS, 0, stream>>>(A, B, C, C_temp, size_m_ptr, scales, zeros, g_idx, m, n, k, get_power2(quant_group), iters, size_atomic_cache, nullptr, false);
        }
    } else {
        if constexpr(BLOCKS_K % 2 == 1) {
            __hgemm_singular_blocks_k::hgemm_gptq<scalar_t,
                w_type_id,
                THREADS,
                BLOCKS_M, BLOCKS_N, BLOCKS_K,
                HAS_ACT_ORDER, HAS_ZP, HAS_M_PRED, HAS_NK_PRED, FP32_ATOMIC, USE_ATOMIC_CACHE, true
                ><<<dim3(blocks, chunks, 1), THREADS, 0, stream>>>(A, B, C, C_temp, size_m_ptr, scales, zeros, g_idx, m, n, k, get_power2(quant_group), iters, size_atomic_cache, nullptr, false);
        } else {
            __hgemm_even_blocks_k::hgemm_gptq<scalar_t,
                w_type_id,
                THREADS,
                BLOCKS_M, BLOCKS_N, BLOCKS_K,
                HAS_ACT_ORDER, HAS_ZP, HAS_M_PRED, HAS_NK_PRED, FP32_ATOMIC, USE_ATOMIC_CACHE, true
                ><<<dim3(blocks, chunks, 1), THREADS, 0, stream>>>(A, B, C, C_temp, size_m_ptr, scales, zeros, g_idx, m, n, k, get_power2(quant_group), iters, size_atomic_cache, nullptr, false);
        }
    }
    return true;
}


template<typename scalar_t, int QBITS>
bool __launch_gemm_gptq_fused_moe_kernel(
    const scalar_t* __restrict__ hidden_states,
    const PackType* __restrict__ gate_projs,
    const scalar_t* __restrict__ gate_projs_scales,
    const PackType* __restrict__ up_projs,
    const scalar_t* __restrict__ up_projs_scales,
    const int64_t* __restrict__ selected_experts,
    half* C, float* C_temp, int experts_range, int experts_per_token,
    int origin_m, int prob_m, int prob_n, int prob_k, int quant_group,
    cudaStream_t stream
) {
    int quant_group_power2 = get_power2(quant_group);
    if (quant_group_power2 != 6 && quant_group_power2 != 7) {
        printf("Invalid quant_group %d, quant_group_power = %d\n", quant_group, quant_group_power2);
        return false;
    }

    if (prob_m > MAX_BLOCKS_M * SLICE_M) {
        printf("prob_m %d > %d is not supported currently\n", prob_m, MAX_BLOCKS_M * SLICE_M);
        return false;
    }
    int BLOCKS_M = div_ceil(prob_m, SLICE_M);
    int BLOCKS_K = quant_group / SLICE_K;
    int BLOCKS_N = BLOCKS_M <= 2 ? 16 : 8;

    if (BLOCKS_K % 2 != 0) {
        printf("BLOCKS_K %d is singular and not supported yet!\n", BLOCKS_K);
        return false;
    }



    int tiles_m = div_ceil(prob_m, TILE_M);
    int tiles_n = div_ceil(prob_n, TILE_N);
    int tiles_k = div_ceil(prob_k, TILE_K);

    bool HAS_M_PRED = (prob_m % TILE_M) != 0;
    bool HAS_NK_PRED = ((prob_n % TILE_N) != 0) || ((prob_k % TILE_K) != 0);

    int total_tiles = tiles_n * tiles_k;
    int blocks = PEUS;
    int iters = div_ceil(total_tiles, PEUS);
    if (total_tiles < PEUS) {
        if (TILE_K < quant_group) {
            iters = quant_group / TILE_K;
            blocks = div_ceil(total_tiles, iters);
        } else {
            iters = 1;
            blocks = total_tiles;
        }
    } else {
        if (TILE_K < quant_group) {
            iters = div_ceil(iters, quant_group / TILE_K) * quant_group / TILE_K;
            blocks = div_ceil(total_tiles, iters);
        }
    }
    while (iters * blocks - total_tiles >= iters) {
        blocks -= 1;
    }

    dim3 block = dim3(blocks, origin_m, 2*experts_per_token);
    constexpr int THREADS = 256;
    bool HAS_ACT_ORDER = false;
    bool HAS_ZP = false;
    bool has_zp = false;
    bool has_act_order = false;
    constexpr bool fp32_atomic = true;


#define LAUNCH_FUSED_MOE(threads, bm, bn, bk, has_nk_pred, has_m_pred) \
    else if (THREADS == threads && BLOCKS_M == bm && BLOCKS_N == bn \
        && BLOCKS_K == bk  \
        && HAS_M_PRED == has_m_pred && HAS_NK_PRED == has_nk_pred) { \
            __hgemm_even_blocks_k::hgemm_gptq_fused_moe<scalar_t, half, sglang::kU4B8.id(), \
                    threads, bm, bn, bk, false, false, \
                    has_m_pred, has_nk_pred, fp32_atomic, false><<<block, threads, 0, stream>>>( \
                    hidden_states, gate_projs, gate_projs_scales, \
                    up_projs, up_projs_scales, \
                    selected_experts, (PackTypeInt4*)C, (PackTypeInt4*)C_temp, \
                    experts_range, experts_per_token, origin_m, \
                    prob_m, prob_n, prob_k, iters, 0, quant_group_power2); \
    }

    if (false) {

    }
    LAUNCH_FUSED_MOE(256, 1, 16, 2, false, true)
    LAUNCH_FUSED_MOE(256, 1, 16, 2, true, true)
    // LAUNCH_FUSED_MOE(256, 2, 16, 2, false, true)
    // LAUNCH_FUSED_MOE(256, 3, 8, 2, false, true)
    // LAUNCH_FUSED_MOE(256, 4, 8, 2, false, true)
    // LAUNCH_FUSED_MOE(256, 1, 16, 2, false, false)
    // LAUNCH_FUSED_MOE(256, 2, 16, 2, false, false)
    // LAUNCH_FUSED_MOE(256, 3, 8, 2, false, false)
    // LAUNCH_FUSED_MOE(256, 4, 8, 2, false, false)
    else {
        printf("prob_m=%d,prob_n=%d,prob_k=%d,BLOCKS_M=%d,BLOCKS_N=%d,BLOCKS_K=%d, quant_group=%d is not supported\n",
            prob_m, prob_n, prob_k, BLOCKS_M, BLOCKS_N, BLOCKS_K, quant_group);
        return false;
    }

#undef LAUNCH_FUSED_MOE
    return true;
}

template<typename scalar_t, int QBITS>
bool launch_gemm_gptq_fused_moe_kernel(
    const scalar_t* __restrict__ hidden_states,
    const PackType* __restrict__ gate_projs,
    const scalar_t* __restrict__ gate_projs_scales,
    const PackType* __restrict__ up_projs,
    const scalar_t* __restrict__ up_projs_scales,
    const int64_t* __restrict__ selected_experts,
    half* C, float* C_tmp, int experts_range, int experts_per_token,
    int m, int n, int k, int quant_group,
    cudaStream_t stream
) {
    return __launch_gemm_gptq_fused_moe_kernel<scalar_t, QBITS>(
        hidden_states, gate_projs, gate_projs_scales, up_projs, up_projs_scales,
        selected_experts, C, C_tmp,
        experts_range, experts_per_token,
        m, 1, n, k, quant_group,
        stream
    );
    // return ret;
}


template<typename scalar_t, int QBITS>
bool __launch_gemm_gptq_fused_moe_muti_scaled_kernel(
    const scalar_t* __restrict__ hidden_states,
    const PackType* __restrict__ down_projs,
    const scalar_t* __restrict__ down_projs_scales,
    const int64_t* __restrict__ selected_experts,
    const scalar_t* __restrict__ routing_weights,
    half* C, float* C_temp, int experts_range, int experts_per_token,
    int origin_m, int prob_m, int prob_n, int prob_k, int quant_group,
    cudaStream_t stream
) {
    int quant_group_power2 = get_power2(quant_group);
    if (quant_group_power2 != 6 && quant_group_power2 != 7) {
        printf("Invalid quant_group %d, quant_group_power = %d\n", quant_group, quant_group_power2);
        return false;
    }

    if (prob_m > MAX_BLOCKS_M * SLICE_M) {
        printf("prob_m %d > %d is not supported currently\n", prob_m, MAX_BLOCKS_M * SLICE_M);
        return false;
    }
    int BLOCKS_M = div_ceil(prob_m, SLICE_M);
    int BLOCKS_K = quant_group / SLICE_K;
    int BLOCKS_N = BLOCKS_M <= 2 ? 16 : 8;

    if (BLOCKS_K % 2 != 0) {
        printf("BLOCKS_K %d is singular and not supported yet!\n", BLOCKS_K);
        return false;
    }

    int tiles_m = div_ceil(prob_m, TILE_M);
    int tiles_n = div_ceil(prob_n, TILE_N);
    int tiles_k = div_ceil(prob_k, TILE_K);

    bool HAS_M_PRED = (prob_m % TILE_M) != 0;
    bool HAS_NK_PRED = ((prob_n % TILE_N) != 0) || ((prob_k % TILE_K) != 0);

    int total_tiles = tiles_n * tiles_k;
    int blocks = PEUS;
    int iters = div_ceil(total_tiles, PEUS);
    if (total_tiles < PEUS) {
        if (TILE_K < quant_group) {
            iters = quant_group / TILE_K;
            blocks = div_ceil(total_tiles, iters);
        } else {
            iters = 1;
            blocks = total_tiles;
        }
    } else {
        if (TILE_K < quant_group) {
            iters = div_ceil(iters, quant_group / TILE_K) * quant_group / TILE_K;
            blocks = div_ceil(total_tiles, iters);
        }
    }
    while (iters * blocks - total_tiles >= iters) {
        blocks -= 1;
    }

    dim3 block = dim3(blocks, origin_m, experts_per_token);
    constexpr int THREADS = 256;
    bool HAS_ACT_ORDER = false;
    bool HAS_ZP = false;
    bool has_zp = false;
    bool has_act_order = false;
    constexpr bool fp32_atomic = true;


#define LAUNCH_FUSED_MOE(threads, bm, bn, bk, has_nk_pred, has_m_pred) \
    else if (THREADS == threads && BLOCKS_M == bm && BLOCKS_N == bn \
        && BLOCKS_K == bk  \
        && HAS_M_PRED == has_m_pred && HAS_NK_PRED == has_nk_pred) { \
            __hgemm_even_blocks_k::hgemm_gptq_fused_moe_multi_scaled<scalar_t, half, sglang::kU4B8.id(), \
                    threads, bm, bn, bk, false, false, \
                    has_m_pred, has_nk_pred, fp32_atomic, false><<<block, threads, 0, stream>>>( \
                    hidden_states, down_projs, down_projs_scales, \
                    selected_experts, routing_weights, (PackTypeInt4*)C, (PackTypeInt4*)C_temp, \
                    experts_range, experts_per_token, \
                    origin_m, prob_m, prob_n, prob_k, iters, 0, quant_group_power2); \
    }

    if (false) {

    }
    LAUNCH_FUSED_MOE(256, 1, 16, 2, false, true)
    LAUNCH_FUSED_MOE(256, 1, 16, 2, true, true)
    // LAUNCH_FUSED_MOE(256, 2, 16, 2, false, true)
    // LAUNCH_FUSED_MOE(256, 3, 8, 2, false, true)
    // LAUNCH_FUSED_MOE(256, 4, 8, 2, false, true)
    // LAUNCH_FUSED_MOE(256, 1, 16, 2, false, false)
    // LAUNCH_FUSED_MOE(256, 2, 16, 2, false, false)
    // LAUNCH_FUSED_MOE(256, 3, 8, 2, false, false)
    // LAUNCH_FUSED_MOE(256, 4, 8, 2, false, false)
    else {
        printf("prob_m=%d,prob_n=%d,prob_k=%d,BLOCKS_M=%d,BLOCKS_N=%d,BLOCKS_K=%d, quant_group=%d is not supported\n",
            prob_m, prob_n, prob_k, BLOCKS_M, BLOCKS_N, BLOCKS_K, quant_group);
        return false;
    }

#undef LAUNCH_FUSED_MOE
    return true;
}

template<typename scalar_t, int QBITS>
bool launch_gemm_gptq_fused_moe_muti_scaled_kernel(
    const scalar_t* __restrict__ hidden_states,
    const PackType* __restrict__ down_projs,
    const scalar_t* __restrict__ down_projs_scales,
    const int64_t* __restrict__ selected_experts,
    const scalar_t* __restrict__ routing_weights,
    half* C, float* C_temp, int experts_range, int experts_per_token,
    int m, int n, int k, int quant_group,
    cudaStream_t stream
) {
    return __launch_gemm_gptq_fused_moe_muti_scaled_kernel<scalar_t, QBITS>(
        hidden_states, down_projs, down_projs_scales,
        selected_experts, routing_weights, C, C_temp,
        experts_range, experts_per_token,
        m, 1, n, k, quant_group,
        stream
    );
    //return ret;
}

}

