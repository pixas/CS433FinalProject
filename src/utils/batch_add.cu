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

// float mat_a[4 * 3 * 4 * 4] = {
//     // img 1
//     // channel 1
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,
//     //channel 2
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,
//     // channel 3
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,

//     // img 2
//     // channel 1
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,
//     //channel 2
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,
//     // channel 3
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,

//     // img 3
//     // channel 1
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,
//     //channel 2
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,
//     // channel 3
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,

//     // img 4
//     // channel 1
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,
//     //channel 2
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,
//     // channel 3
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,
// };

// float mat_b[4 * 3 * 4 * 4] = {
//     // img 1
//     // channel 1
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,
//     //channel 2
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,
//     // channel 3
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,

//     // img 2
//     // channel 1
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,
//     //channel 2
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,
//     // channel 3
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,

//     // img 3
//     // channel 1
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,
//     //channel 2
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,
//     // channel 3
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,

//     // img 4
//     // channel 1
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,
//     //channel 2
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,
//     // channel 3
//     0, 1, 2 ,3,
//     4, 5, 6, 7,
//     8, 9, 10, 11,
//     12, 13, 14, 15,
// };

// #include <iostream>
// int main() {
//     float *mat_a_gpu, *mat_b_gpu, *mat_out_gpu;
//     cudaMalloc(&mat_a_gpu, sizeof(float) * 4 * 3 * 4 * 4);
//     cudaMalloc(&mat_b_gpu, sizeof(float) * 4 * 3 * 4 * 4);
//     cudaMalloc(&mat_out_gpu, sizeof(float) * 4 * 3 * 4 * 4);

//     cudaMemcpy(mat_a_gpu, mat_a, sizeof(float) * 4 * 3 * 4 * 4, cudaMemcpyHostToDevice);
//     cudaMemcpy(mat_b_gpu, mat_b, sizeof(float) * 4 * 3 * 4 * 4, cudaMemcpyHostToDevice);

//     batch_add(mat_a_gpu, mat_b_gpu, mat_out_gpu, 4, 3, 4, 4);

//     float mat_out[4 * 3 * 4 * 4];
//     cudaMemcpy(mat_out, mat_out_gpu, sizeof(float) * 4 * 3 * 4 * 4, cudaMemcpyDeviceToHost);

//     for (int i = 0; i < 4; ++i) {
//         for (int j = 0; j < 3; ++j) {
//             std::cout << "channel" << j << std::endl;
//             for (int k = 0; k < 4; ++k) {
//                 for (int l = 0; l < 4; ++l) {
//                     std::cout << mat_out[i * 3 * 4 * 4 + j * 4 * 4 + k * 4 + l] << " ";
//                 }
//                 std::cout << std::endl;
//             }
//             std::cout << std::endl;
//         }
//         std::cout << std::endl;
//     }
// }