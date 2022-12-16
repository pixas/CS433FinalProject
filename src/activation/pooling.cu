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
    // int batch_size = gridDim.z;
    int batch_idx = blockIdx.z;

    int target_height = (height + 2 * padding - 1) / stride;
    int target_width = (width + 2 * padding - 1) / stride;
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
            T max_value = -1e9;
            int max_idx = -1;
            for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                    int final_idx = cur_idx + h * width + w;
                    // printf("%d %d %d %d\n", (final_idx / (channels * height * width)), (final_idx % (channels * height * width) / (height * width)), ((final_idx % (channels * height * width) % (height * width)) / width), ((final_idx % (channels * height * width) % (height * width)) % width));
                    if (input[final_idx] > max_value) {
                        max_idx = final_idx;
                        max_value = input[max_idx];
                    }
                }
            }
            // printf("%d %d %d %d %.1f\n", batch_idx, dim, row, col, max_value);
            int cur_index = batch_idx * channels * target_height * target_width + dim * target_height * target_width + row * target_width + col;
            output[cur_index] = max_value;
            // printf("%.2f\n", output[cur_index]);
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
    // dim3 grid2(1, 1);
    // dim3 block2(channels, batch_size);
    // printf("%.3f\n", output[0]);
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
  

// #include <cassert>
// #include <iostream>
// #include <random>
// #include <vector>

// constexpr int BATCH_SIZE = 128;
// constexpr int CHANNELS = 512;
// constexpr int HEIGHT = 28;
// constexpr int WIDTH = 28;

// int main() {
//     int stride = 2;
//     int kernel = 3;
//     int padding = 1;
//     std::vector<float> input(BATCH_SIZE * CHANNELS * HEIGHT * WIDTH);
//     std::vector<float> expected_output(BATCH_SIZE * CHANNELS * 1);
//     int out_h = 1;
//     int out_w = 1;
//     // Set random number generator for input data
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<> dis(0.0, 1.0);
//     for (int i = 0; i < BATCH_SIZE * CHANNELS * HEIGHT * WIDTH; ++i) {
//         input[i] = dis(gen);
//     }
//     std::cout << "Initialized over\n";
//     for (int i = 0; i < HEIGHT; ++i) {
//         for (int j = 0; j < WIDTH; ++j) {
//             std::cout << input[1 * CHANNELS * HEIGHT * WIDTH + 1 * HEIGHT * WIDTH + i * WIDTH + j] << (j == WIDTH - 1 ? "\n" : " ");
//         }
//     }
//     std::cout << BATCH_SIZE << " " << HEIGHT << " " << WIDTH << " " << CHANNELS << std::endl;
//     // Create a vector of random input values
//     for (int i = 0; i < BATCH_SIZE; ++i)
//     {
//         for (int d = 0; d < CHANNELS; ++d) {
//             float total_sum = 0;
//             for (int r = 0; r < HEIGHT; ++r) {
//                 for (int c = 0; c < WIDTH; ++c) {
//                     total_sum += input[i * CHANNELS * HEIGHT * WIDTH + d * HEIGHT * WIDTH + r * WIDTH + c];
//                 }
//             }
//             total_sum = total_sum / float(HEIGHT * WIDTH);
//             expected_output[i * CHANNELS + d] = total_sum;
//         }
//     }
//     for (int i = 0; i < BATCH_SIZE; ++i) {
//         for (int j = 0; j < CHANNELS; ++j) {
//             std::cout << expected_output[i * CHANNELS + j] <<  (j == CHANNELS - 1 ? "\n" : " ");
//         }
//     }
//     std::cout << "Create expected output\n";
//     std::vector<float> output(BATCH_SIZE * CHANNELS * out_h * out_w);
//     float * d_input;
//     cudaMalloc((void**)&d_input, sizeof(float) * input.size());
//     cudaMemcpy(d_input, input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice);
//     float * d_output;
//     cudaMalloc((void**)&d_output, sizeof(float) * BATCH_SIZE * CHANNELS * out_h * out_w);

//     adaptive_mean_pool(d_input, d_output, BATCH_SIZE, CHANNELS, HEIGHT, WIDTH);
//     float * host_output = new float[output.size()];
//     cudaMemcpy(host_output, d_output, sizeof(int) * output.size(), cudaMemcpyDeviceToHost);
//     output.assign(host_output, host_output + output.size());
//     for (int i = 0; i < BATCH_SIZE; ++i) {
//         for (int j = 0; j < CHANNELS; ++j) {
//             std::cout << output[i * CHANNELS + j] <<  (j == CHANNELS - 1 ? "\n" : " ");
//         }
//     }
//     for (int i = 0; i < BATCH_SIZE; ++i) {
//         for (int j = 0; j < CHANNELS; ++j) {
//             std::cout << ((expected_output[i * CHANNELS + j] - output[i * CHANNELS + j]) < abs(1e-5)) <<  (j == CHANNELS - 1 ? "\n" : " ");
//         }
//     }
//     assert(output == expected_output);

//     cudaFree(d_input);
//     cudaFree(d_output);

//     return 0;
// }

// #include <cassert>
// #include <iostream>
// #include <random>
// #include <vector>

// constexpr int BATCH_SIZE = 2;
// constexpr int CHANNELS = 4;
// constexpr int HEIGHT = 14;
// constexpr int WIDTH = 14;

// int main() {
//     int stride = 2;
//     int kernel = 3;
//     int padding = 1;
//     std::vector<float> input(BATCH_SIZE * CHANNELS * HEIGHT * WIDTH);
//     std::vector<float> expected_output(BATCH_SIZE * CHANNELS * (HEIGHT + 2 * padding - 1) / stride * (WIDTH + 2 * padding - 1) / stride);
//     int out_h = (HEIGHT + 2 * padding - 1) / stride;
//     int out_w = (WIDTH + 2 * padding - 1) / stride;
//     // Set random number generator for input data
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<> dis(0.0, 1.0);
//     for (int i = 0; i < BATCH_SIZE * CHANNELS * HEIGHT * WIDTH; ++i) {
//         input[i] = dis(gen);
//     }
//     std::cout << "Initialized over\n";
//     for (int i = 0; i < HEIGHT; ++i) {
//         for (int j = 0; j < WIDTH; ++j) {
//             std::cout << input[1 * CHANNELS * HEIGHT * WIDTH + 1 * HEIGHT * WIDTH + i * WIDTH + j] << (j == WIDTH - 1 ? "\n" : " ");
//         }
//     }
//     std::cout << BATCH_SIZE << " " << HEIGHT << " " << WIDTH << " " << CHANNELS << std::endl;
//     // Create a vector of random input values
//     for (int i = 0; i < BATCH_SIZE; ++i)
//     {
//         for (int r = 0; r < HEIGHT; r += stride) {
//             for (int c = 0; c < WIDTH; c += stride) {
//                 for (int d = 0; d < CHANNELS; ++d) {
//                     int h_start = max(0, r - kernel / 2);
//                     int w_start = max(0, c - kernel / 2);
//                     int h_end = min(HEIGHT, r + kernel / 2 + 1);
//                     int w_end = min(WIDTH, c + kernel / 2 + 1);
//                     float max_value = FLT_MIN;
//                     for (int h = h_start; h < h_end; ++h) {
//                         for (int w = w_start; w < w_end; ++w) {
//                             if (input[i * CHANNELS * HEIGHT * WIDTH + d * HEIGHT * WIDTH + h * WIDTH + w] > max_value) {
//                                 max_value = input[i * CHANNELS * HEIGHT * WIDTH + d * HEIGHT * WIDTH + h * WIDTH + w];
//                             }

//                         }
//                     }
//                     expected_output[i * CHANNELS * out_h * out_w + d * out_h * out_w + (r / stride) * out_w + (c / stride)] = max_value;
//                 }
//             }
//         }
//     }
//     for (int i = 0; i < out_h; ++i) {
//         for (int j = 0; j < out_w; ++j) {
//             std::cout << expected_output[1 * CHANNELS * out_h * out_w +1 * out_h * out_w + i * out_w + j] <<  (j == out_w - 1 ? "\n" : " ");
//         }
//     }
//     std::cout << "Create expected output\n";
//     std::vector<float> output(BATCH_SIZE * CHANNELS * out_h * out_w);
//     float * d_input;
//     cudaMalloc((void**)&d_input, sizeof(float) * input.size());
//     cudaMemcpy(d_input, input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice);
//     float * d_output;
//     cudaMalloc((void**)&d_output, sizeof(float) * BATCH_SIZE * CHANNELS * out_h * out_w);

//     max_pool_2d(d_input, d_output, BATCH_SIZE, CHANNELS, HEIGHT, WIDTH, 3, 3, 1, 2);
//     float * host_output = new float[output.size()];
//     cudaMemcpy(host_output, d_output, sizeof(int) * output.size(), cudaMemcpyDeviceToHost);
//     output.assign(host_output, host_output + output.size());
//     for (int i = 0; i < out_h; ++i) {
//         for (int j = 0; j < out_w; ++j) {
//             std::cout << output[1 * CHANNELS * out_h * out_w + 1 * out_h * out_w + i * out_w + j] <<  (j == out_w - 1 ? "\n" : " ");
//         }
//     }
//     assert(output == expected_output);
//     // Create a vector to store the output
//     // std::vector<int> output(input.size() / 4);

//     // // Create device pointers for the input and output
//     // float * d_input;
//     // int * d_output;

//     // Allocate memory on the device for the input and output
//     // cudaMalloc(&d_input, input.size() * sizeof(float));
//     // cudaMalloc(&d_output, output.size() * sizeof(int));

//     // // Copy the input to the device
//     // cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

//     // // Call the argmax function
//     // argmax(d_input, d_output, 4, 4);

//     // // Copy the output back to the host
//     // cudaMemcpy(output.data(), d_output, output.size() * sizeof(int), cudaMemcpyDeviceToHost);

//     // // Print the output
//     // for (int i = 0; i < output.size(); i++) {
//     //     std::cout << output[i] << std::endl;
//     // }

//     // Free the memory on the device
//     cudaFree(d_input);
//     cudaFree(d_output);

//     return 0;
// }