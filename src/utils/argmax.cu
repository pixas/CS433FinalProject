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
    int * h_output = (int *)malloc(sizeof(int) * batch_size);
    cudaMemcpy(h_output, output, sizeof(int) * batch_size, cudaMemcpyDeviceToHost);

    argmax_kernel<T><<<numBlocks, threadsPerBlock>>>(input, output, batch_size, num_classes);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, output, sizeof(int) * batch_size, cudaMemcpyDeviceToHost);
    CUDA_POST_KERNEL_CHECK;
}

template void argmax<float>(float * input, int * output, int batch_size, int num_classes);

// #include <vector>
// #include <iostream>
// int main() {
//     // Create a vector of random input values
//     std::vector<float> input = {
//         0.1, 0.2, 0.3, 0.4, 
//         0.5, 0.6, 0.7, 0.8, 
//         0.9, 0.1, 0.2, 0.3, 
//         0.4, 0.5, 0.6, 0.7, 
//         // 0.8, 0.9, 0.1, 0.2, 
//         // 0.3, 0.4, 0.5, 0.6, 
//         // 0.7, 0.8, 0.9, 0.1, 
//         // 0.2, 0.3, 0.4, 0.5, 
//         // 0.6, 0.7, 0.8, 0.9
//     };

//     // Create a vector to store the output
//     std::vector<int> output(input.size() / 4);

//     // Create device pointers for the input and output
//     float * d_input;
//     int * d_output;

//     // Allocate memory on the device for the input and output
//     cudaMalloc(&d_input, input.size() * sizeof(float));
//     cudaMalloc(&d_output, output.size() * sizeof(int));

//     // Copy the input to the device
//     cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);

//     // Call the argmax function
//     argmax(d_input, d_output, 4, 4);

//     // Copy the output back to the host
//     cudaMemcpy(output.data(), d_output, output.size() * sizeof(int), cudaMemcpyDeviceToHost);

//     // Print the output
//     for (int i = 0; i < output.size(); i++) {
//         std::cout << output[i] << std::endl;
//     }

//     // Free the memory on the device
//     cudaFree(d_input);
//     cudaFree(d_output);

//     return 0;
// }
#include <cassert>
#include <iostream>
#include <random>
#include <vector>



constexpr int BATCH_SIZE = 200;
constexpr int NUM_CLASSES = 1000;

void test_argmax_kernel()
{
// Set up input and expected output
std::vector<float> input(BATCH_SIZE * NUM_CLASSES);
std::vector<int> expected_output(BATCH_SIZE);

// Set random number generator for input data
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

// Generate random input data and expected output
for (int i = 0; i < BATCH_SIZE; ++i)
{
int max_index = 0;
float max_value = 0.0;
for (int j = 0; j < NUM_CLASSES; ++j)
{
  input[i * NUM_CLASSES + j] = dis(gen);

  if (input[i * NUM_CLASSES + j] > max_value)
  {
    max_value = input[i * NUM_CLASSES + j];
    max_index = j;
  }
}

expected_output[i] = max_index;
}

// Run kernel and check output
std::vector<int> output(BATCH_SIZE);
float * d_input;
cudaMalloc((void**)&d_input, sizeof(float) * input.size());
cudaMemcpy(d_input, input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice);
int * d_output;
cudaMalloc((void**)&d_output, sizeof(int) * BATCH_SIZE);

argmax(d_input, d_output, BATCH_SIZE, NUM_CLASSES);
int * host_output = new int[BATCH_SIZE];
cudaMemcpy(host_output, d_output, sizeof(int) * BATCH_SIZE, cudaMemcpyDeviceToHost);
output.assign(host_output, host_output + BATCH_SIZE);
assert(output == expected_output);
}

// int main()
// {
// test_argmax_kernel();
// std::cout << "All tests passed!" << std::endl;
// return 0;
// }