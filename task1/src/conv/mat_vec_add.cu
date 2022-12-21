#include "mat_vec_add.hpp"
#include "common.hpp"

/**
 * @brief matrix vector addition core
 * @param mat       input matrix    shape (height, width), allocated on GPU
 * @param vec       vector to add   shape (height, 1), allocated on GPU
 * @param res_mat   result matrix   shape (height, width), allocated on GPU
 * @param height    matrix height
 * @param width     matrix width
 */
template<typename T>
__global__ void mat_vec_add_kernel(T *mat, T *vec, T *res_mat, int height, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width)
        res_mat[row * width + col] = mat[row * width + col] + vec[row];
}

/**
 * @brief matrix vector addition
 * @param mat       input matrix    shape (height, width), allocated on GPU
 * @param vec       vector to add   shape (height, 1), allocated on GPU
 * @param res_mat   result matrix   shape (height, width), allocated on GPU
 * @param height    matrix height
 * @param width     matrix width
 */
template<typename T>
void mat_vec_add(T *mat, T *vec, T *res_mat, int height, int width) {
    dim3 block(TC_TILE_WIDTH, TC_TILE_WIDTH);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    mat_vec_add_kernel<<<grid, block>>>(mat, vec, res_mat, height, width);
    CUDA_POST_KERNEL_CHECK;
}

template void mat_vec_add<float>(float *mat, float *vec, float *res_mat, int height, int width);

/**
 * @brief matrix vector addition core for batch
 * @param mat           input matrix    shape (batch, height, width), allocated on GPU
 * @param vec           vector to add   shape (height, 1), allocated on GPU
 * @param res_mat       result matrix   shape (batch, height, width), allocated on GPU
 * @param height        matrix height
 * @param width         matrix width
 * @param batch_size    batch size
 */
template<typename T>
__global__ void batch_mat_vec_add_kernel(T *mat, T *vec, T *res_mat, int height, int width, int batch_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.z;

    if (row < height && col < width)
        res_mat[batch_idx * height * width + row * width + col] = mat[batch_idx * height * width + row * width + col] + vec[row];
}

/**
 * @brief matrix vector addition for batch
 * @param mat           input matrix    shape (batch, height, width), allocated on GPU
 * @param vec           vector to add   shape (height, 1), allocated on GPU
 * @param res_mat       result matrix   shape (batch, height, width), allocated on GPU
 * @param height        matrix height
 * @param width         matrix width
 * @param batch_size    batch size
 */
template<typename T>
void batch_mat_vec_add(T *mat, T *vec, T *res_mat, int height, int width, int batch_size) {
    dim3 block(TC_TILE_WIDTH, TC_TILE_WIDTH);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, batch_size);
    batch_mat_vec_add_kernel<<<grid, block>>>(mat, vec, res_mat, height, width, batch_size);
    CUDA_POST_KERNEL_CHECK;
}

template void batch_mat_vec_add<float>(float *mat, float *vec, float *res_mat, int height, int width, int batch_size);