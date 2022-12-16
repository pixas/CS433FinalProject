#include "mma.h"
#include "common.hpp"
#include "gemm.hpp"

using namespace nvcuda;

/**
 * @brief pad the input image with zeros to align with the align_units
 * @param input the input matrix (height x width), allocated on GPU
 * @param output the output matrix (aligned_height x aligned_width), allocated on GPU
 * @param height input matrix height
 * @param width input matrix width
 * @param align_units the alignment unit
 */
template<typename T>
void padding(T *input, T *output, int height, int width, int align_units) {
    int aligned_width = ALIGN(width, align_units);
    for (int i = 0; i < height; i++)
        cudaMemcpy(output + i * aligned_width, input + i * width, width * sizeof(T), cudaMemcpyDeviceToDevice);
}

/**
 * @brief remove the padded zeros in the input image, allocated on GPU
 * @param input the input matrix (aligned_height x aligned_width), allocated on GPU
 * @param output the output matrix (height x width)
 * @param height input matrix height
 * @param width input matrix width
 * @param align_units the alignment unit
 */
template<typename T>
void unpadding(T *input, T *output, int height, int width, int align_units) {
    int aligned_width = ALIGN(width, align_units);
    for (int i = 0; i < height; i++)
        cudaMemcpy(output + i * width, input + i * aligned_width, width * sizeof(T), cudaMemcpyDeviceToDevice);
}

/**
 * @brief gemm kernel
 * @param matrix_a the first matrix (m x k), allocated on GPU
 * @param matrix_b the second matrix (k x n), allocated on GPU
 * @param matrix_c the output matrix (m x n), allocated on GPU
 * @param m the first matrix height
 * @param k the first matrix width = the second matrix height
 * @param n the second matrix width
 * @note  m, n, k must be multiples of 16
 */
__global__ void wmma_gemm_kernel(
    half *matrix_a, half *matrix_b, float *matrix_c, 
    int m, int k, int n
) {
    int k_tile_dim = k / TC_TILE_WIDTH, n_tile_dim = n / TC_TILE_WIDTH;
    int tile_idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int m_tile_idx = tile_idx / n_tile_dim;   // row index in matrix_c in tile units
    int n_tile_idx = tile_idx % n_tile_dim;   // column index in matrix_c in tile units
    float *matrix_c_target_tile_ptr = matrix_c + m_tile_idx * n_tile_dim * TC_TILE_WIDTH * TC_TILE_WIDTH + n_tile_idx * TC_TILE_WIDTH;

    // Declare and initialize the fragments
    wmma::fragment<wmma::matrix_a, TC_TILE_WIDTH, TC_TILE_WIDTH, TC_TILE_WIDTH, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TC_TILE_WIDTH, TC_TILE_WIDTH, TC_TILE_WIDTH, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TC_TILE_WIDTH, TC_TILE_WIDTH, TC_TILE_WIDTH, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over the k dimension to perform the matrix multiplication in tile units
    for (int k_tile_idx = 0; k_tile_idx < k_tile_dim; k_tile_idx++) {
        // Calculate the pointers to load from
        half *matrix_a_tile_ptr = matrix_a + m_tile_idx * k_tile_dim * TC_TILE_WIDTH * TC_TILE_WIDTH + k_tile_idx * TC_TILE_WIDTH;
        half *matrix_b_tile_ptr = matrix_b + k_tile_idx * n_tile_dim * TC_TILE_WIDTH * TC_TILE_WIDTH + n_tile_idx * TC_TILE_WIDTH;

        // Load the inputs
        wmma::load_matrix_sync(a_frag, matrix_a_tile_ptr, k);
        wmma::load_matrix_sync(b_frag, matrix_b_tile_ptr, n);

        // Perform the matrix multiplication
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    
    // Store the output
    wmma::store_matrix_sync(matrix_c_target_tile_ptr, acc_frag, n, wmma::mem_row_major);
}

/**
 * @brief gemm with wmma
 * @param matrix_a the first matrix (m x k), allocated on GPU
 * @param matrix_b the second matrix (k x n), allocated on GPU
 * @param matrix_c the output matrix (m x n), allocated on GPU
 * @param m the first matrix height
 * @param k the first matrix width = the second matrix height
 * @param n the second matrix width
 * @note  m, n, k can be any size
 */
void wmma_gemm(
    half *matrix_a, half *matrix_b, float *matrix_c, 
    int m, int k, int n
) {
    int aligned_m = ALIGN(m, TC_TILE_WIDTH), aligned_k = ALIGN(k, TC_TILE_WIDTH), aligned_n = ALIGN(n, TC_TILE_WIDTH);
    half *aligned_matrix_a, *aligned_matrix_b;
    float *aligned_matrix_c;
    cudaMalloc(&aligned_matrix_a, aligned_m * aligned_k * sizeof(half));
    cudaMalloc(&aligned_matrix_b, aligned_k * aligned_n * sizeof(half));
    cudaMalloc(&aligned_matrix_c, aligned_m * aligned_n * sizeof(float));

    padding(matrix_a, aligned_matrix_a, m, k, TC_TILE_WIDTH);
    padding(matrix_b, aligned_matrix_b, k, n, TC_TILE_WIDTH);

    // each warp to compute one tile
    int num_warps = (aligned_m / TC_TILE_WIDTH) * (aligned_n / TC_TILE_WIDTH);

    dim3 dimGrid, dimBlock;
    if (num_warps * WARP_SIZE > CUDA_NUM_THREADS) {
        dimGrid.x = (num_warps * WARP_SIZE + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
        dimBlock.x = CUDA_NUM_THREADS;
    } else {
        dimGrid.x = 1;
        dimBlock.x = num_warps * WARP_SIZE;
    }

    wmma_gemm_kernel<<<dimGrid, dimBlock>>>(
        aligned_matrix_a, aligned_matrix_b, aligned_matrix_c, 
        aligned_m, aligned_k, aligned_n
    );
    cudaDeviceSynchronize();

    unpadding(aligned_matrix_c, matrix_c, m, n, TC_TILE_WIDTH);

    cudaFree(aligned_matrix_a);
    cudaFree(aligned_matrix_b);
    cudaFree(aligned_matrix_c);
}

/**
 * @brief right batch matrix multiplication kernel
 * @param matrix_a the first matrix (m x k), allocated on GPU
 * @param batch_matrix_b the second matrix of whole batch (b x k x n), allocated on GPU
 * @param batch_matrix_c the output matrix of whole batch(b x m x n), allocated on GPU
 * @param m the first matrix height
 * @param k the first matrix width = the second matrix height
 * @param n the second matrix width
 * @note  m, n, k must be multiples of 16
 */
__global__ void wmma_rbmm_kernel(
    half *matrix_a, half *batch_matrix_b, float *batch_matrix_c,
    int m, int k, int n
) {
    int batch_idx = blockIdx.y;
    int m_tile_dim = m / TC_TILE_WIDTH, k_tile_dim = k / TC_TILE_WIDTH, n_tile_dim = n / TC_TILE_WIDTH;
    int tile_idx = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    if (tile_idx >= m_tile_dim * n_tile_dim)
        return;
    
    int m_tile_idx = tile_idx / n_tile_dim;   // row index in matrix_c in tile units
    int n_tile_idx = tile_idx % n_tile_dim;   // column index in matrix_c in tile units
    float *matrix_c_target_tile_ptr = batch_matrix_c + batch_idx * m * n + m_tile_idx * n_tile_dim * TC_TILE_WIDTH * TC_TILE_WIDTH + n_tile_idx * TC_TILE_WIDTH;

    // Declare and initialize the fragments
    wmma::fragment<wmma::matrix_a, TC_TILE_WIDTH, TC_TILE_WIDTH, TC_TILE_WIDTH, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TC_TILE_WIDTH, TC_TILE_WIDTH, TC_TILE_WIDTH, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TC_TILE_WIDTH, TC_TILE_WIDTH, TC_TILE_WIDTH, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over the k dimension to perform the matrix multiplication in tile units
    for (int k_tile_idx = 0; k_tile_idx < k_tile_dim; k_tile_idx++) {
        // Calculate the pointers to load from
        half *matrix_a_tile_ptr = matrix_a + m_tile_idx * k_tile_dim * TC_TILE_WIDTH * TC_TILE_WIDTH + k_tile_idx * TC_TILE_WIDTH;
        half *matrix_b_tile_ptr = batch_matrix_b + batch_idx * k * n + k_tile_idx * n_tile_dim * TC_TILE_WIDTH * TC_TILE_WIDTH + n_tile_idx * TC_TILE_WIDTH;

        // Load the inputs
        wmma::load_matrix_sync(a_frag, matrix_a_tile_ptr, k);
        wmma::load_matrix_sync(b_frag, matrix_b_tile_ptr, n);

        // Perform the matrix multiplication
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
    
    // Store the output
    wmma::store_matrix_sync(matrix_c_target_tile_ptr, acc_frag, n, wmma::mem_row_major);
}

/**
 * @brief right batch matrix multiplication
 * @param matrix_a the first matrix (m x k), allocated on GPU
 * @param batch_matrix_b the second matrix of whole batch (b x k x n), allocated on GPU
 * @param batch_matrix_c the output matrix of whole batch (b x m x n), allocated on GPU
 * @param batch_size the batch size
 * @param m the first matrix height
 * @param k the first matrix width = the second matrix height
 * @param n the second matrix width
 * @note  m, n, k can be any size
 */
void wmma_rbmm(
    half *matrix_a, half *batch_matrix_b, float *batch_matrix_c,
    int batch_size, int m, int k, int n
) {
    int aligned_m = ALIGN(m, TC_TILE_WIDTH);
    int aligned_k = ALIGN(k, TC_TILE_WIDTH);
    int aligned_n = ALIGN(n, TC_TILE_WIDTH);

    MALLOC_ERR_DECLARATION;

    half *aligned_matrix_a, *aligned_batch_matrix_b;
    float *aligned_batch_matrix_c;
    cudaMalloc(&aligned_matrix_a, aligned_m * aligned_k * sizeof(half));
    cudaMalloc(&aligned_batch_matrix_b, batch_size * aligned_k * aligned_n * sizeof(half));
    cudaMalloc(&aligned_batch_matrix_c, batch_size * aligned_m * aligned_n * sizeof(float));

    CUDA_POST_MALLOC_CHECK;

    // batch padding
    padding(matrix_a, aligned_matrix_a, m, k, TC_TILE_WIDTH);
    for (int i = 0; i < batch_size; i++)
        padding(batch_matrix_b + i * k * n, aligned_batch_matrix_b + i * aligned_k * aligned_n, k, n, TC_TILE_WIDTH);

    // each warp to compute one tile
    int num_warps = (aligned_m / TC_TILE_WIDTH) * (aligned_n / TC_TILE_WIDTH);

    dim3 dimGrid, dimBlock;
    if (num_warps * WARP_SIZE > CUDA_NUM_THREADS) {
        dimGrid.x = (num_warps * WARP_SIZE + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
        dimBlock.x = CUDA_NUM_THREADS;
    } else {
        dimGrid.x = 1;
        dimBlock.x = num_warps * WARP_SIZE;
    }
    dimGrid.y = batch_size;

    wmma_rbmm_kernel<<<dimGrid, dimBlock>>>(
        aligned_matrix_a, aligned_batch_matrix_b, aligned_batch_matrix_c, 
        aligned_m, aligned_k, aligned_n
    );
    cudaDeviceSynchronize();
    CUDA_POST_KERNEL_CHECK;

    // batch unpadding
    for (int i = 0; i < batch_size; i++)
        unpadding(aligned_batch_matrix_c + i * aligned_m * aligned_n, batch_matrix_c + i * m * n, m, n, TC_TILE_WIDTH);
    
    cudaFree(aligned_matrix_a);
    cudaFree(aligned_batch_matrix_b);
    cudaFree(aligned_batch_matrix_c);
}

// float kernels_o[3 * 3 * 3 * 3] = {
//     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
//     3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
// };

// float col_o[27 * 4] = {
//     0, 1, 4, 5,
//     1, 2, 5, 6,
//     2, 3, 6, 7,
//     4, 5, 8, 9,
//     5, 6, 9, 10,
//     6, 7, 10, 11,
//     8, 9, 12, 13,
//     9, 10, 13, 14,
//     10, 11, 14, 15,

//     0, 1, 4, 5,
//     1, 2, 5, 6,
//     2, 3, 6, 7,
//     4, 5, 8, 9,
//     5, 6, 9, 10,
//     6, 7, 10, 11,
//     8, 9, 12, 13,
//     9, 10, 13, 14,
//     10, 11, 14, 15, 

//     0, 1, 4, 5,
//     1, 2, 5, 6,
//     2, 3, 6, 7,
//     4, 5, 8, 9,
//     5, 6, 9, 10,
//     6, 7, 10, 11,
//     8, 9, 12, 13,
//     9, 10, 13, 14,
//     10, 11, 14, 15,
// };

// int main() {
//     half col_batch[3 * 27 * 4];
//     half kernels[3 * 27];
//     float output[3 * 3 * 4];

//     for (int i = 0; i < 3; i++) {
//         for (int j = 0; j < 27; j++) {
//             for (int k = 0; k < 4; k++) {
//                 col_batch[i * 27 * 4 + j * 4 + k] = __float2half(col_o[j * 4 + k]);
//                 // printf("%f ", __half2float(col_batch[i * 27 * 4 + j * 4 + k]));
//             }
//             // printf("\n");
//         }
//         // printf("\n");
//     }
//     for (int i = 0; i < 3 * 27; i++) {
//         kernels[i] = __float2half(kernels_o[i]);
//     }

//     half *col_batch_gpu, *kernels_gpu;
//     float *output_gpu;
//     cudaMalloc(&col_batch_gpu, 3 * 27 * 4 * sizeof(half));
//     cudaMalloc(&kernels_gpu, 3 * 27 * sizeof(half));
//     cudaMalloc(&output_gpu, 3 * 3 * 4 * sizeof(float));
//     cudaMemcpy(col_batch_gpu, col_batch, 3 * 27 * 4 * sizeof(half), cudaMemcpyHostToDevice);
//     cudaMemcpy(kernels_gpu, kernels, 3 * 27 * sizeof(half), cudaMemcpyHostToDevice);

//     wmma_rbmm(kernels_gpu, col_batch_gpu, output_gpu, 3, 3, 27, 4);

//     cudaMemcpy(output, output_gpu, 3 * 3 * 4 * sizeof(float), cudaMemcpyDeviceToHost);

//     for (int i = 0; i < 3; i++) {
//         for (int j = 0; j < 3; j++) {
//             for (int k = 0; k < 4; k++) {
//                 printf("%f ", output[i * 3 * 4 + j * 4 + k]);
//             }
//             printf("\n");
//         }
//         printf("\n");
//     }
// }