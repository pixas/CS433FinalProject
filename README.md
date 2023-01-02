# CS433FinalProject

## Overview
SJTU CS433 final project.
This lab has two tasks: use tensor cores in volta architecture to implement gemm operators; simulate a volta architecture GPU with C++ and corresponding tensor cores' functionalities.

### Task1

See details in `task1/README.md`
We use C++ to load ResNet18's parameters and use CUDA cores and Tensor cores to build a ResNet18 network.
We randomly select 5000 images and complete correct inference with little wrong predictions, which are all caused by precision errors.

### File Structure
```
D:.
│  benchmark.sh
│  buildenv.mk
│  Makefile
│  README.md
│  resnet18.onnx
│  select_file_list.txt
│
├─include
│      batch_add.hpp
│      common.hpp
│      conv.hpp
│      data_utils.hpp
│      float_half_trans.hpp
│      gemm.hpp
│      im2col.hpp
│      mat_vec_add.hpp
│      model_utils.hpp
│      padding.hpp
│
├─src
│  │  inference.cu
│  │  oracle.cpp
│  │  validate.py
│  │
│  ├─activation
│  │      pooling.cu
│  │      relu.cu
│  │
│  ├─conv
│  │      conv.cu
│  │      float_half_trans.cu
│  │      gemm.cu
│  │      im2col.cu
│  │      mat_vec_add.cu
│  │      padding.cu
│  │
│  └─utils
│          argmax.cu
│          batch_add.cu
│          load_parameters.cpp
│
└─target
    ├─benchmark
    │      benchmark_b16.txt
    │      benchmark_b32.txt
    │      benchmark_b64.txt
    │      benchmark_oracle.txt
    │      output_b128_opt.txt
    │      output_b256_opt.txt
    │      output_b32_baseline.txt
    │      output_b32_opt.txt
    │      output_b64_opt.txt
    │
    ├─bin
    │      inference
    │      oracle
    │
    └─output
            error_file_list.txt
            error_file_list_opt.txt
            error_list_predictions.txt
            err_list_final_output
            oracle_predictions.txt
            predictions.txt
```
### Task2

We use provided `task2/include/tensor_core.hpp` to implement `task2/src/utils/tensor_core.cpp`, where we implement most functionalities of Volta GPU, including HMMA. 
We replace the `wmma_rbmm_kernel` with our implemented `sim_wmma` and complete infernce of 10 images.
The inference result shows that our implementation is correct.

### SASS file
We provide the row major based sass file in `ref/Lab1/test.sass`, which is generated based on `ref/Lab1/tensor_core.cu`.

### File Structure
```
D:.
│  app
│  app.sass
│  benchmark.sh
│  buildenv.mk
│  Makefile
│  README.md
│  resnet18.onnx
│  select_file_list.txt
│  test
│
├─include
│      batch_add.hpp
│      common.hpp
│      conv.hpp
│      data_utils.hpp
│      float_half_trans.hpp
│      gemm.hpp
│      im2col.hpp
│      mat_vec_add.hpp
│      model_utils.hpp
│      padding.hpp
│      tensor_core.hpp
│
├─src
│  │  inference.cu
│  │  oracle.cpp
│  │  validate.py
│  │
│  ├─activation
│  │      pooling.cu
│  │      relu.cu
│  │
│  ├─conv
│  │      conv.cu
│  │      float_half_trans.cu
│  │      gemm.cu
│  │      im2col.cu
│  │      mat_vec_add.cu
│  │      padding.cu
│  │
│  └─utils
│          argmax.cu
│          batch_add.cu
│          load_parameters.cpp
│          tensor_core.cpp
│
└─target
    ├─benchmark
    │      benchmark_b128.txt
    │      benchmark_b16.txt
    │      benchmark_b256.txt
    │      benchmark_b32.txt
    │      benchmark_b64.txt
    │      benchmark_oracle.txt
    │      output_b128_opt.txt
    │      output_b256_opt.txt
    │      output_b32_baseline.txt
    │      output_b32_opt.txt
    │      output_b64_opt.txt
    │      sim_wmma.txt
    │
    ├─bin
    │      inference
    │      oracle
    │
    └─output
            error_file_list.txt
            error_file_list_opt.txt
            error_list_predictions.txt
            err_list_final_output
            oracle_predictions.txt
            predictions.txt
```