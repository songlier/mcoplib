// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
// clang-format off
// adapted from: https://github.com/soundOfDestiny/cutlass/blob/a4208aa6958864923505cade9c63eb2a6daf16e5/include/cutlass/gemm/collective/fp8_accumulation.hpp



#pragma once

#include "cute/algorithm/clear.hpp"
#include "cute/tensor.hpp"

//////////////////////////////////////////////////////////////////////////////
///////////////////////////////////FP8 Accumulation///////////////////////////
//////////////////////////////////////////////////////////////////////////////
/// This class provides API to promote (add) or scale (multiply_add) the results
/// from the tensor core accumulators to the main accumulators when the number 
/// of MMAs reaches the max number of MMA interval specified by user, after that
/// the tensor core accumulators are zeroed.
//////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

template <
    class EngineAccum,
    class LayoutAccum>
struct GmmaFP8AccumulationWithScale {  
  using TensorAccum = cute::Tensor<EngineAccum, LayoutAccum>;
  using ElementAccumulator = typename EngineAccum::value_type;

  static_assert(is_static<LayoutAccum>::value, "Accumulator Layout should be static");
  static_assert(is_rmem<TensorAccum>::value , "Accumulator tensor must be rmem resident.");

private:
  TensorAccum& accum_;
  TensorAccum accum_temp_;

  uint32_t accum_promotion_interval_;         // defines the max num of executed MMAs after which accum should be promoted.
  uint32_t mma_count_per_mainloop_iteration_; // num of MMAs per k_tile of mainloop
  uint32_t mma_count_;                        // current executed MMAs
  uint32_t reset_accum_flag_;                 // accum needs to be zeroed or not. 

  // promote or `add` the partial accumulators to main accumulator (FADD).
  CUTLASS_DEVICE
  void promote_core() {
    warpgroup_wait<0>();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum_); ++i) {
      accum_(i) += accum_temp_(i);
    }
  }

  // `multiply` scale the partial accumulators and `add` to main accumulator (FFMA).
  template <
    class EngineScale,
    class LayoutScale>
  CUTLASS_DEVICE
  void scale_core(const cute::Tensor<EngineScale, LayoutScale> &scale) {
    using TensorScale = cute::Tensor<EngineScale, LayoutScale>;

    static_assert(is_static<LayoutScale>::value, "Scale Layout should be static");
    static_assert(is_rmem<TensorScale>::value , "Scale tensor must be rmem resident.");

    static_assert(LayoutAccum{}.shape() == LayoutScale{}.shape(), "Accumulator and scale must have same shape.");

    warpgroup_wait<0>();
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accum_); ++i) {
      accum_(i) += accum_temp_(i) * scale(i);
    }
  }

public:
  CUTLASS_DEVICE
  GmmaFP8AccumulationWithScale(
      TensorAccum &accum,
      uint32_t accum_promotion_interval,
      uint32_t mma_count_per_mainloop_iteration)
      : accum_(accum), 
        accum_promotion_interval_(accum_promotion_interval),
        mma_count_per_mainloop_iteration_(mma_count_per_mainloop_iteration),
        mma_count_(0), 
        reset_accum_flag_(0) 
  {
    accum_temp_ = cute::make_fragment_like(accum);
  }

  //
  // Methods (Common)
  //

  CUTLASS_DEVICE 
  TensorAccum& operator()() {
    return accum_temp_;
  }

  /// prepare the MMA accumulators when initialization or zeroing is required.
  CUTLASS_DEVICE
  bool prepare_if_needed() { 
    return reset_accum_flag_;
  }

  //
  // Methods (for FADD version)
  //

  /// promote (add) the results from the MMA accumulators to main accumulator if needed.
  CUTLASS_DEVICE
  void promote_if_needed() {
    mma_count_ += mma_count_per_mainloop_iteration_;
    reset_accum_flag_ = __shfl_sync(0xffffffff, mma_count_ == accum_promotion_interval_, 0);
    if (reset_accum_flag_) {
      promote_core();
      mma_count_ = 0;
    }
  }

  /// promote (add) the residue results from the MMA accumulators to main accumulator if needed.
  CUTLASS_DEVICE
  void promote_residue_if_needed() {
    if (__shfl_sync(0xffffffff, mma_count_ > 0, 0)) {
      promote_core();
    }
  }

  //
  // Methods (for FFMA version)
  //

  /// scale (multiply_add) the results from the MMA accumulators to main accumulator if needed.
  template <
    class EngineScale,
    class LayoutScale>
  CUTLASS_DEVICE
  void scale_if_needed(const cute::Tensor<EngineScale, LayoutScale> &scale) {
    mma_count_ += mma_count_per_mainloop_iteration_;
    reset_accum_flag_ = __shfl_sync(0xffffffff, mma_count_ == accum_promotion_interval_, 0);
    if (reset_accum_flag_) {
      scale_core(scale);
      mma_count_ = 0;
    }
  }

  /// scale (multiply_add) the residue results from the MMA accumulators to main accumulator if needed.
  template <
    class EngineScale,
    class LayoutScale>
  CUTLASS_DEVICE
  void scale_residue_if_needed(const cute::Tensor<EngineScale, LayoutScale> &scale) {
    if (__shfl_sync(0xffffffff, mma_count_ > 0, 0)) {
      scale_core(scale);
    }
  }
};

} // namespace cutlass::gemm::collective
