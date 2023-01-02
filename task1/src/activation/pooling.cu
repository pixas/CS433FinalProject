#include "model_utils.hpp"
#include "common.hpp"
#include <cuda_runtime_api.h>
#include <iostream>
#include <cstdlib>
#include <float.h>



/**
 * @brief maxpool2d kernel function
 * @param input       input tensor    shape (b, dim, height, width), allocated on GPU
 * @param output      output tensor   shape (b, dim, height, width), allocated on GPU
 * @param channels    input tensor channel
 * @param height    matrix height
 * @param width     matrix width
 * @param kernel_h  kernel height
 * @param kernel_w  kernel width
 * @param padding   padding size
 * @param stride    stride for max pooling sliding window
 */
template<typename T>
__global__ void max_pool_2d_kernel(T *input, T *output, int channels, int height, int width, int kernel_h, int kernel_w, int padding, int stride) {
    int batch_idx = blockIdx.z;

    int target_height = (height + 2 * padding - 1) / stride;
    int target_width = (width + 2 * padding - 1) / stride;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < target_height && col < target_width) {
        int hstart = row * stride - padding;
        int wstart = col * stride - padding;
        const int hend = min(hstart + kernel_h, height);
        const int wend = min(wstart + kernel_w, width);
        hstart = max(0, hstart);
        wstart = max(0, wstart);
        for (int dim = 0; dim < channels; ++dim) {
            int cur_idx = batch_idx * channels * height * width + dim * height * width;
            T max_value = -1e9;
            int max_idx = -1;
            for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                    int final_idx = cur_idx + h * width + w;
                    if (input[final_idx] > max_value) {
                        max_idx = final_idx;
                        max_value = input[max_idx];
                    }
                }
            }
            int cur_index = batch_idx * channels * target_height * target_width + dim * target_height * target_width + row * target_width + col;
            output[cur_index] = max_value;
        }
    }
}




template<typename T>
void max_pool_2d(T *input, T *output, int batch_size, int channels, int height, int width, int kernel_h, int kernel_w, int padding, int stride) {
    dim3 block(TC_TILE_WIDTH, TC_TILE_WIDTH);
    int target_height = (height + 2 * padding - 1) / stride;
    int target_width = (width + 2 * padding - 1) / stride;
    int grid_x = (target_width + block.x - 1) / block.x;
    int grid_y = (target_height + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y, batch_size);
    max_pool_2d_kernel<<<grid, block>>>(input, output, channels, height, width, kernel_h, kernel_w, padding, stride);

}

template void max_pool_2d<float>(float *input, float *output, int batch_size, int channels, int height, int width, int kernel_h, int kernel_w, int padding, int stride);



/**
 * @brief adaptive mean sum kernel function
 * @param input       input tensor    shape (b, dim, height, width), allocated on GPU
 * @param output      output tensor   shape (b, dim, height, width), allocated on GPU
 * @param channels    input tensor channel
 * @param height    matrix height
 * @param width     matrix width
 * @param output_height output height for padding
 * @param output_width output width for padding
 */
 template<typename T>
 __global__ void adaptive_pool_sum_kernel(T *input, T* sum, int channels, int height, int width, int output_height = 1, int output_width = 1) {
    int batch_idx = blockIdx.z;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        for (int dim = 0; dim < channels; ++dim) {
            // printf("%d %d %d %d\n", batch_idx, dim, row, col);
            int cur_idx = batch_idx * channels * height * width + dim * height * width + row * width + col;
            T cur_value = input[cur_idx];
            // sum[batch_idx * channels + dim] = cur_value;
            // sum[batch_idx * channels + dim] += cur_value;
            atomicAdd(&sum[batch_idx * channels + dim], cur_value);
        }
    }
 
 }
  
 /**
 * @brief relu activation function
 * @param input       input tensor    shape (b, dim, height, width), allocated on GPU
 * @param output      output tensor   shape (b, dim, height, width), allocated on GPU
 * @param channels    input tensor channel
 * @param height    matrix height
 * @param width     matrix width
 * @param output_height output height for padding
 * @param output_width output width for padding
 */
 template<typename T>
 __global__ void adaptive_pool_mean_kernel(T* output, int batch_size, int channels, int height, int width) {
    // int total_points = height * width;
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int col = blockIdx.x * blockDim.x + threadIdx.x;
    // if (row < batch_size && col < channels) {
    //     int cur_id = row * channels + col;
    //     output[cur_id] /= float(total_points);
    // }

    int total_points = height * width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batch_size * channels) {
        // printf("%d %.3f\n", idx, output[idx]);
        output[idx] /= float(total_points);
    }
}
 
 
 
template<typename T>
void adaptive_mean_pool(T *input, T *output, int batch_size, int channels, int height, int width) {
    dim3 block(TC_TILE_WIDTH, TC_TILE_WIDTH);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, batch_size);

    adaptive_pool_sum_kernel<<<grid, block>>>(input, output, channels, height, width);
    dim3 block2, grid2;
    if (batch_size * channels > CUDA_NUM_THREADS) {
        block2.x = CUDA_NUM_THREADS;
        grid2.x = (batch_size * channels + block2.x - 1) / block2.x;
    } else {
        block2.x = batch_size * channels;
        grid2.x = 1;
    }
    adaptive_pool_mean_kernel<<<grid2, block2>>>(output, batch_size, channels, height, width);
    CUDA_POST_KERNEL_CHECK;
}
 
template void adaptive_mean_pool<float>(float * input, float * output, int batch_size, int channels, int height, int width);
