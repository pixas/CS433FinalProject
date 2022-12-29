# CS433FinalProject

## Overview
SJTU CS433 final project.
This lab has two tasks: use tensor cores in volta architecture to implement gemm operators; simulate a volta architecture GPU with C++ and corresponding tensor cores' functionalities.

### Task1

See details in `task1/README.md`
We use C++ to load ResNet18's parameters and use CUDA cores and Tensor cores to build a ResNet18 network.
We randomly select 5000 images and complete correct inference with little wrong predictions, which are all caused by precision errors.

### Task2

We use provided `task2/include/tensor_core.hpp` to implement `task2/src/utils/tensor_core.cpp`, where we implement most functionalities of Volta GPU, including HMMA. 
We replace the `wmma_rbmm_kernel` with our implemented `sim_wmma` and complete infernce of 10 images.
The inference result shows that our implementation is correct.