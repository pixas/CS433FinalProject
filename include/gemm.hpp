#ifndef _GOURP14_GEMM_HPP_
#define _GROUP14_GEMM_HPP_

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
); 

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
);

#endif /* _GROUP_GEMM_HPP */