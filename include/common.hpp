#ifndef _GROUP14_COMMON_HPP_
#define _GROUP14_COMMON_HPP_

#define CUDA_NUM_THREADS 512
#define CUDA_GET_BLOCKS(N) ((N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS)

#define WARP_SIZE 32

#define TC_TILE_WIDTH 16    // Tensor Core tile width

#define ALIGN(x, y) ((x + y - 1) / y * y)

#endif