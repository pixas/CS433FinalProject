ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
TARGET_DIR := $(ROOT_DIR)/target/bin
INCLUDE_DIR := $(ROOT_DIR)/include
MAKE := make
MKDIR := mkdir -p
CP := cp -f
RM := rm -f

NVCC := nvcc
NVCC_FLAGS := -O3 -arch=sm_70 -I$(INCLUDE_DIR)

CUDA_LIB_PATH := /usr/local/cuda/lib64
CUDA_LIB := -L$(CUDA_LIB_PATH) -lcudart -lcublas -lcudnn

CXX := g++
CXX_FLAGS := -O3 -I$(INCLUDE_DIR) $(CUDA_LIB)

OPENCV_LIB := $(shell pkg-config --libs opencv4)
OPENCV_INC := $(shell pkg-config --cflags opencv4)
