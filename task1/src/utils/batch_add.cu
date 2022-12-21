#include "batch_add.hpp"
#include "common.hpp"

/**
 * @brief batch addition core
 * @param mat_a       input matrix    shape (batch, channel, height, width), allocated on GPU
 * @param mat_b       input matrix    shape (batch, channel, height, width), allocated on GPU
 * @param mat_out     output matrix   shape (batch, channel, height, width), allocated on GPU
 * @param batch       batch size
 * @param channel     channel size
 * @param height      height size
 * @param width       width size
 */
template<typename T>
__global__ void batch_add_kernel(T *mat_a, T *mat_b, T *mat_out, int batch, int channel, int height, int width) {
    int channel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_idx = blockIdx.z;
    int idx = batch_idx * channel * height * width + channel_idx * height * width + matrix_idx;
    int row = matrix_idx / width;
    int col = matrix_idx % width;

    if (row < height && col < width)
        mat_out[idx] = mat_a[idx] + mat_b[idx];
}

/**
 * @brief batch addition
 * @param mat_a       input matrix    shape (batch, channel, height, width), allocated on GPU
 * @param mat_b       input matrix    shape (batch, channel, height, width), allocated on GPU
 * @param mat_out     output matrix   shape (batch, channel, height, width), allocated on GPU
 * @param batch       batch size
 * @param channel     channel size
 * @param height      height size
 * @param width       width size
 */
template<typename T>
void batch_add(T *mat_a, T *mat_b, T *mat_out, int batch, int channel, int height, int width) {
    dim3 block(TC_TILE_WIDTH, TC_TILE_WIDTH);
    dim3 grid((channel + block.x - 1) / block.x, (height * width + block.y - 1) / block.y, batch);

    batch_add_kernel<T><<<grid, block>>>(mat_a, mat_b, mat_out, batch, channel, height, width);
    CUDA_POST_KERNEL_CHECK;
}

template void batch_add<float>(float *mat_a, float *mat_b, float *mat_out, int batch, int channel, int height, int width);
