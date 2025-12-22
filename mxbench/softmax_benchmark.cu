// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
/*
 * Softmax Performance Benchmark using NVBench
 */

#include <nvbench/nvbench.cuh>
#include <nvbench/main.cuh>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>

// Softmax kernel implementation
__global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;

        // Find max value in the row (simplified - would need reduction in practice)
        float max_val = input[idx];
        for (int c = 0; c < cols; c++) {
            max_val = fmaxf(max_val, input[row * cols + c]);
        }

        // Compute exp(x - max) and sum
        float exp_val = expf(input[idx] - max_val);
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            sum += expf(input[row * cols + c] - max_val);
        }

        // Final softmax
        output[idx] = exp_val / sum;
    }
}

// More efficient softmax with shared memory
__global__ void softmax_optimized_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float row_max[256];
    __shared__ float row_sum[256];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= rows) return;

    // Each thread processes multiple elements
    int elements_per_thread = (cols + blockDim.x - 1) / blockDim.x;
    int start_col = tid * elements_per_thread;
    int end_col = min(start_col + elements_per_thread, cols);

    // Find max in the row
    float max_val = -std::numeric_limits<float>::infinity();
    for (int col = start_col; col < end_col; col++) {
        max_val = fmaxf(max_val, input[row * cols + col]);
    }

    // Reduce to find global max
    row_max[tid] = max_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            row_max[tid] = fmaxf(row_max[tid], row_max[tid + stride]);
        }
        __syncthreads();
    }

    float global_max = row_max[0];
    __syncthreads();

    // Compute exp values and sum
    float local_sum = 0.0f;
    for (int col = start_col; col < end_col; col++) {
        float exp_val = expf(input[row * cols + col] - global_max);
        output[row * cols + col] = exp_val; // Store intermediate exp values
        local_sum += exp_val;
    }

    // Reduce to find global sum
    row_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            row_sum[tid] += row_sum[tid + stride];
        }
        __syncthreads();
    }

    float global_sum = row_sum[0];

    // Normalize
    for (int col = start_col; col < end_col; col++) {
        output[row * cols + col] /= global_sum;
    }
}

// Benchmark function for basic softmax
void softmax_benchmark_basic(nvbench::state &state)
{
    const auto batch_size = static_cast<int>(state.get_int64("BatchSize"));
    const auto seq_length = static_cast<int>(state.get_int64("SeqLength"));

    // Allocate device memory
    thrust::device_vector<float> input(batch_size * seq_length);
    thrust::device_vector<float> output(batch_size * seq_length);

    // Initialize with some data
    thrust::sequence(input.begin(), input.end());
    thrust::transform(input.begin(), input.end(), input.begin(),
        [] __host__ __device__ (float x) { return (x / 10000.0f) * 2.0f - 1.0f; });

    // Add throughput information
    state.add_element_count(batch_size * seq_length, "Elements");
    state.add_global_memory_reads<float>(batch_size * seq_length, "InputSize");
    state.add_global_memory_writes<float>(batch_size * seq_length, "OutputSize");

    // Configure kernel launch
    const int block_size = 256;
    const int grid_size = batch_size;

    state.exec([batch_size, seq_length, grid_size, block_size,
                input_ptr = thrust::raw_pointer_cast(input.data()),
                output_ptr = thrust::raw_pointer_cast(output.data())]
               (nvbench::launch &launch) {
        softmax_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(
            input_ptr, output_ptr, batch_size, seq_length);
    });
}

// Benchmark function for optimized softmax
void softmax_benchmark_optimized(nvbench::state &state)
{
    const auto batch_size = static_cast<int>(state.get_int64("BatchSize"));
    const auto seq_length = static_cast<int>(state.get_int64("SeqLength"));

    // Allocate device memory
    thrust::device_vector<float> input(batch_size * seq_length);
    thrust::device_vector<float> output(batch_size * seq_length);

    // Initialize with some data
    thrust::sequence(input.begin(), input.end());
    thrust::transform(input.begin(), input.end(), input.begin(),
        [] __host__ __device__ (float x) { return (x / 10000.0f) * 2.0f - 1.0f; });

    // Add throughput information
    state.add_element_count(batch_size * seq_length, "Elements");
    state.add_global_memory_reads<float>(batch_size * seq_length, "InputSize");
    state.add_global_memory_writes<float>(batch_size * seq_length, "OutputSize");

    // Configure kernel launch
    const int block_size = 256;
    const int grid_size = batch_size;

    state.exec([batch_size, seq_length, grid_size, block_size,
                input_ptr = thrust::raw_pointer_cast(input.data()),
                output_ptr = thrust::raw_pointer_cast(output.data())]
               (nvbench::launch &launch) {
        softmax_optimized_kernel<<<grid_size, block_size, 0, launch.get_stream()>>>(
            input_ptr, output_ptr, batch_size, seq_length);
    });
}

// Register benchmarks
NVBENCH_BENCH(softmax_benchmark_basic)
    .add_int64_axis("BatchSize", {32, 64, 128, 256, 512})
    .add_int64_axis("SeqLength", {128, 256, 512, 1024, 2048});

NVBENCH_BENCH(softmax_benchmark_optimized)
    .add_int64_axis("BatchSize", {32, 64, 128, 256, 512})
    .add_int64_axis("SeqLength", {128, 256, 512, 1024, 2048});


NVBENCH_MAIN