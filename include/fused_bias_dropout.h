#include <ATen/ATen.h>

at::Tensor fused_bias_dropout(at::Tensor input, 
                            at::Tensor residual, 
                            float dropout_prob);
