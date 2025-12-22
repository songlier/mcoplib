#include <ATen/ATen.h>

void all_reduce_max(at::Tensor input, at::Tensor output);
void all_reduce_sum(at::Tensor input, at::Tensor output);