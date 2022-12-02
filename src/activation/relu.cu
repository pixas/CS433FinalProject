#include "model_utils.hpp"

#include <iostream>



/**
 * @brief relu activation function
 * @param input       input tensor    shape (b, dim, height, width), allocated on GPU
 * @param output      output tensor   shape (b, dim, height, width), allocated on GPU
 * @param batch_size batch size for tensor
 * @param height    matrix height
 * @param width     matrix width
 * @param input_dim input tensor's dimension
 */
template<typename T>
__global__ void relu_kernel(T *input, T *output, int height, int width, int input_dim) {
//  int batch_size = gridDim.z;
    int batch_idx = blockIdx.z;

    // int block_idx = gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x;
    // int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    // int current_id = thread_idx + block_idx * blockDim.x * blockDim.y;

//  int n_points = height * width;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        for (int dim = 0; dim < input_dim; ++dim) {
            int cur_idx = batch_idx * input_dim * height * width + dim * height * width + row * width + col;
            T cur_value = input[cur_idx];
            output[cur_idx] = max(cur_value, 0.);
        }
    }

}

/**
* @brief relu activation function
* @param input       input tensor    shape (b, dim, height * width), allocated on GPU
* @param output      output tensor   shape (b, dim, height * width), allocated on GPU
* @param batch_size batch size for tensor
* @param height    matrix height
* @param width     matrix width
* @param input_dim input tensor's dimension
*/
template<typename T>
void relu(T *input, T *output, int batch_size, int height, int width, int input_dim) {

    dim3 block(TC_TILE_WIDTH, TC_TILE_WIDTH);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, batch_size);
    relu_kernel<<<grid, block>>>(input, output, height, width, input_dim);
    
}

template void relu<float>(float *input, float * output, int batch_size, int height, int width, int input_dim);

 