#include "model_utils.hpp"
#include "common.hpp"

template<typename T>
__global__ void argmax_kernel(T *input, int *output, int batch_size, int num_classes) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    // Check if the index is within the bounds of the batch size
    if (idx >= batch_size)
        return;

    // Find the maximum value and its index within the input array
    T max_val = input[idx * num_classes];
    int max_idx = 0;
    for (int i = 1; i < num_classes; ++i) {
        T val = input[idx * num_classes + i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    // Store the maximum index in the output array
    output[idx] = max_idx;
}

template<typename T>
void argmax(T * input, int * output, int batch_size, int num_classes) {
    dim3 threadsPerBlock(1, 1);
    dim3 numBlocks(batch_size, 1);

    // Launch the kernel
    argmax_kernel<T><<<numBlocks, threadsPerBlock>>>(input, output, batch_size, num_classes);
    cudaDeviceSynchronize();
    CUDA_POST_KERNEL_CHECK;
}

template void argmax<float>(float * input, int * output, int batch_size, int num_classes);
