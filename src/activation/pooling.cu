#include "model_utils.hpp"
#include <cuda_runtime_api.h>
#include <iostream>
#include <cstdlib>
#include <float.h>



/**
 * @brief relu activation function
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
    // int batch_size = gridDim.z;
    int batch_idx = blockIdx.z;
    int target_height = height / stride;
    int target_width = width / stride;
    // int block_idx = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
    // int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    // int current_id = thread_idx + block_idx * blockDim.x * blockDim.y;

    // int n_points = height * width;
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

            T max_value = -FLT_MAX;
            int max_idx = -1;
            for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                    if (input[cur_idx + h * width + w] > max_value) {
                        max_idx = cur_idx + h * width + w;
                        max_value = input[max_idx];
                    }
                }
            }
            int cur_index = batch_idx * channels * target_height * target_height + dim * target_height * target_width + row * target_width + col;
            output[cur_index] = max_value;
        }
    }

}




template<typename T>
void max_pool_2d(T *input, T *output, int batch_size, int channels, int height, int width, int kernel_h, int kernel_w, int padding, int stride) {
    dim3 block(TC_TILE_WIDTH, TC_TILE_WIDTH);
    dim3 grid((width / stride + block.x - 1) / block.x, (height / stride + block.y - 1) / block.y, batch_size);
    max_pool_2d_kernel<<<grid, block>>>(input, output, channels, height, width, kernel_h, kernel_w, padding, stride);

}

template void max_pool_2d(float *input, float *output, int batch_size, int channels, int height, int width, int kernel_h, int kernel_w, int padding, int stride);



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
 __global__ void adaptive_pool_sum_kernel(T *input, T* sum, int channels, int height, int width, int output_height = 1, int output_width = 1) {
     // int batch_size = gridDim.z;
     int batch_idx = blockIdx.z;
 
 
     // int n_points = height * width;
     int row = blockIdx.y * blockDim.y + threadIdx.y;
     int col = blockIdx.x * blockDim.x + threadIdx.x;
 
     if (row < height && col < width) {
         for (int dim = 0; dim < channels; ++dim) {
             int cur_idx = batch_idx * channels * height * width + dim * height * width + row * width + col;
             T cur_value = input[cur_idx];
             sum[batch_idx * channels + dim] += cur_value;
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
     int total_points = height * width;
     int row = blockIdx.y * blockDim.y + threadIdx.y;
     int col = blockIdx.x * blockDim.x + threadIdx.x;
     if (row < batch_size && col < channels) {
         int cur_id = row * channels + col;
         output[cur_id] /= float(total_points);
     }
 
 }
 
 
 
 
 template<typename T>
 void adaptive_mean_pool(T *input, T *output, int batch_size, int channels, int height, int width) {
     dim3 block(TC_TILE_WIDTH, TC_TILE_WIDTH);
     dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, batch_size);
 
     adaptive_pool_sum_kernel<<<grid, block>>>(input, output, channels, height, width);
     dim3 grid2(1, 1);
     dim3 block2(channels, batch_size);
     adaptive_pool_mean_kernel<<<grid2, block2>>>(output, batch_size, channels, height, width);
 
 }
 
 template void adaptive_mean_pool(float * input, float * output, int batch_size, int channels, int height, int width);
  
  