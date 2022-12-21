#include "mma.h"
#include "float_half_trans.hpp"
#include "common.hpp"

using namespace nvcuda;

__global__ void float2half_kernel(float *in, half *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = __float2half(in[idx]);
}

__global__ void half2float_kernel(half *in, float *out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    out[idx] = __half2float(in[idx]);
}

void float2half(float *in, half *out, int n) {
    dim3 block(CUDA_NUM_THREADS), grid((n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
    float2half_kernel<<<grid, block>>>(in, out, n);
}

void half2float(half *in, float *out, int n) {
    dim3 block(CUDA_NUM_THREADS), grid((n + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
    half2float_kernel<<<grid, block>>>(in, out, n);
}
