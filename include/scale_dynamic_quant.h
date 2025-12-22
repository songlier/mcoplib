#pragma once

#include <ATen/ATen.h>
#include <torch/extension.h>

std::tuple<at::Tensor, at::Tensor> scale_dynamic_quant(
    const at::Tensor& hidden_states,
    const at::Tensor& smooth_scales,
    at::ScalarType dst_dtype = at::ScalarType::Char
);