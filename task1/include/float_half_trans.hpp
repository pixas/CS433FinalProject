#ifndef _GROUP_14_FLOAT_HALF_TRANS_HPP_
#define _GROUP_14_FLOAT_HALF_TRANS_HPP_

#include <cuda_runtime.h>
#include "mma.h"

using namespace nvcuda;

void float2half(float *in, half *out, int n); 
void half2float(half *in, float *out, int n);

#endif