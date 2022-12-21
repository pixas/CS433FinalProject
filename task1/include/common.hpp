#ifndef _GROUP14_COMMON_HPP_
#define _GROUP14_COMMON_HPP_

#define CUDA_NUM_THREADS 512
#define CUDA_GET_BLOCKS(N) ((N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS)

#define WARP_SIZE 32

#define TC_TILE_WIDTH 16    // Tensor Core tile width

#define ALIGN(x, y) ((x + y - 1) / y * y)

#include <stdio.h>
// #define MALLOC_ERR_DECLARATION cudaError_t malloc_err;
// #define CUDA_POST_KERNEL_CHECK cudaError_t kernel_err = cudaGetLastError(); if (kernel_err != cudaSuccess) { printf("CUDA kernel failed : %s\n%s at L%d in %s\n", cudaGetErrorString(kernel_err), __FILE__, __LINE__, __FUNCTION__); exit(-1); }
// #define CUDA_POST_MALLOC_CHECK malloc_err = cudaGetLastError(); if (malloc_err != cudaSuccess) { printf("CUDA malloc failed : %s\n%s at L%d in %s\n", cudaGetErrorString(malloc_err), __FILE__, __LINE__, __FUNCTION__); exit(-1); }
#define MALLOC_ERR_DECLARATION 
#define CUDA_POST_KERNEL_CHECK 
#define CUDA_POST_MALLOC_CHECK 

#endif