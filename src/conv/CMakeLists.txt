cmake_minimum_required(VERSION 2.8)

project(libconv)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_70)

if (NOT DEFINED CS433_GROUP14_TC_ROOT_DIR)
    set (CS433_GROUP14_TC_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../..)
endif()

if (NOT DEFINED CS433_GROUP14_TC_INCLUDE_DIR)
    set (CS433_GROUP14_TC_INCLUDE_DIR ${CS433_GROUP14_TC_ROOT_DIR}/include)
endif()

if (NOT DEFINED CUDA_INCLUDE_DIRS)
    set(CUDA_INCLUDE_DIRS ${CS433_GROUP14_TC_INCLUDE_DIR})
endif()

cuda_add_executable(
    libconv.a
    im2col.cu ${CS433_GROUP14_TC_INCLUDE_DIR}/im2col.hpp
    gemm.cu ${CS433_GROUP14_TC_INCLUDE_DIR}/gemm.hpp
    conv.cu ${CS433_GROUP14_TC_INCLUDE_DIR}/conv.hpp
)