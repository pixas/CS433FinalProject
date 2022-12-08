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
__global__ void batch_norm_2d_kernel(T *input, T *output, int batch_size, int channels, int height, int width, T* running_mean, T * running_var, T * weight, T * bias) {
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

    if (row < height && col < width) {
        // int hstart = row * stride - padding;
        // int wstart = col * stride - padding;
        // const int hend = min(hstart + kernel_h, height);
        // const int wend = min(wstart + kernel_w, width);
        // hstart = max(0, hstart);
        // wstart = max(0, wstart);
        // for (int dim = 0; dim < channels; ++dim) {
        //     int cur_idx = batch_idx * channels * height * width + dim * height * width;

        //     T max_value = -FLT_MAX;
        //     int max_idx = -1;
        //     for (int h = hstart; h < hend; ++h) {
        //         for (int w = wstart; w < wend; ++w) {
        //             if (input[cur_idx + h * width + w] > max_value) {
        //                 max_idx = cur_idx + h * width + w;
        //                 max_value = input[max_idx];
        //             }
        //         }
        //     }
        //     int cur_index = batch_idx * channels * target_height * target_height + dim * target_height * target_width + row * target_width + col;
        //     output[cur_index] = max_value;
        for (int dim = 0; dim < channels; ++dim) {
            int cur_idx = batch_idx * channels * height * width + dim * height * width;
            T* normed_input = (input[cur_idx + row * width + col] - running_mean[dim]) / (running_var[dim] + 1e-5);
            T* true_out = (normed_input * weight[dim]) + bias[dim];
            output[cur_idx + row * width + col] = true_out;
        }
        
    }

}




template<typename T>
void batch_norm_2d(T *input, T *output, int batch_size, int channels, int height, int width, T* running_mean, T * running_var, T * weight, T * bias) {
    dim3 block(TC_TILE_WIDTH, TC_TILE_WIDTH);
    dim3 grid((width / stride + block.x - 1) / block.x, (height / stride + block.y - 1) / block.y, batch_size);
    batch_norm_2d_kernel<<<grid, block>>>(input, output, channels, height, width, running_mean, running_var, weight, bias);

}

template void batch_norm_2d(float *input, float *output, int batch_size, int channels, int height, int width, float* running_mean, float * running_var, float * weight, float * bias);

