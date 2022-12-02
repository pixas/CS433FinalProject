#ifndef _GROUP14_IM2COL_HPP_
#define _GOURP14_IM2COL_HPP_

#include "mma.h"

/**
 * @brief im2col for 2d convolution
 * @param data_im       input img data (whole batch), allocated on GPU  shape (batch_size, channels, height, width)
 * @param batch_size    batch size
 * @param channels      input img channels = kernel channels
 * @param height        input img height
 * @param width         input img width
 * @param kernel_h      kernel height
 * @param kernel_w      kernel width
 * @param pad_h         padding height
 * @param pad_w         padding width
 * @param stride_h      stride height
 * @param stride_w      stride width
 * @param dilation_h    dilation height
 * @param dilation_w    dilation width
 * @param height_col    output col height
 * @param width_col     output col width
 * @param data_col      output col data, allocated on GPU
 * @return None.
 */
void im2col_gpu(
    half *data_im,
    const int batch_size, const int channels, 
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    half *data_col
);

#endif /* _GROUP14_IM2COL_HPP_ */