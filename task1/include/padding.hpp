#ifndef _GOURP14_PADDING_HPP_
#define _GROUP14_PADDING_HPP_

#include "mma.h"
#include <cuda_runtime.h>

using namespace nvcuda;

void batch_padding(
    half *input, half *output, 
    int height, int width, 
    int aligned_height, int aligned_width, 
    int batch_size
);

void batch_unpadding(
    float *input, float *output, 
    int height, int width, 
    int aligned_height, int aligned_width, 
    int batch_size
);

#endif  /* _GROUP14_PADDING_HPP_ */