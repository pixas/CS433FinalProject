#include "padding.hpp"
#include "common.hpp"

__global__ void batch_padding_kernel(
    half *input, half *output, 
    int height, int width, 
    int aligned_height, int aligned_width, 
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / (height * width);
    int row_idx = (idx % (height * width)) / width;
    int col_idx = idx % width;

    if (idx < batch_size * height * width) 
        output[batch_idx * aligned_height * aligned_width + row_idx * aligned_width + col_idx] = input[idx];
}

void batch_padding(
    half *input, half *output, 
    int height, int width, 
    int aligned_height, int aligned_width, 
    int batch_size
) {
    dim3 block(CUDA_NUM_THREADS), grid((batch_size * height * width + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
    batch_padding_kernel<<<grid, block>>>(input, output, height, width, aligned_height, aligned_width, batch_size);
    CUDA_POST_KERNEL_CHECK;
}

__global__ void batch_unpadding_kernel(
    float *input, float *output, 
    int height, int width, 
    int aligned_height, int aligned_width, 
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / (height * width);
    int row_idx = (idx % (height * width)) / width;
    int col_idx = idx % width;

    if (idx < batch_size * height * width) 
        output[idx] = input[batch_idx * aligned_height * aligned_width + row_idx * aligned_width + col_idx];
}

void batch_unpadding(
    float *input, float *output, 
    int height, int width, 
    int aligned_height, int aligned_width, 
    int batch_size
) {
    dim3 block(CUDA_NUM_THREADS), grid((batch_size * height * width + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
    batch_unpadding_kernel<<<grid, block>>>(input, output, height, width, aligned_height, aligned_width, batch_size);
    CUDA_POST_KERNEL_CHECK;
}
