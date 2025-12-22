
#pragma once

#include <torch/library.h>
#include <torch/extension.h>
#include <torch/torch.h>

torch::Tensor gptq_marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                               torch::Tensor& b_scales, torch::Tensor& g_idx,
                               torch::Tensor& perm, torch::Tensor& workspace,
                               int64_t num_bits, torch::Tensor& size_m_tensor,
                               int64_t size_m, int64_t size_n,
                               int64_t size_k, int sms, bool is_k_full, 
                               at::ScalarType dtype=torch::kBFloat16,
                               bool use_atomic_cache = true);

torch::Tensor gptq_marlin_gemm_legacy(torch::Tensor& a, torch::Tensor& b_q_weight,
                               torch::Tensor& b_scales, torch::Tensor& g_idx,
                               torch::Tensor& perm, torch::Tensor& workspace,
                               int64_t num_bits,
                               int64_t size_m, int64_t size_n,
                               int64_t size_k, bool is_k_full,
                               at::ScalarType dtype=torch::kBFloat16,
                               bool use_atomic_cache = true);