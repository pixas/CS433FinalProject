#ifndef _GROUP14_CONV_HPP_
#define _GROUP14_CONV_HPP_

/**
 * @brief 2d convolution with im2col and wmma
 * @param data_input    input img data, allocated on GPU
 * @param batch_size    batch size
 * @param channels      input img channels = kernel channels
 * @param height        input img height
 * @param width         input img width
 * @param data_kernel   kernel data, allocated on GPU, shape (channels, kernel_h, kernel_w)
 * @param num_kernels   number of kernels
 * @param kernel_h      kernel height
 * @param kernel_w      kernel width
 * @param pad_h         padding height
 * @param pad_w         padding width
 * @param stride_h      stride height
 * @param stride_w      stride width
 * @param dialtion_h    dialtion height
 * @param dialtion_w    dialtion width
 * @param data_bias     bias data, allocated on GPU, shape (num_kernels, 1)
 * @param data_output   output data, allocated on GPU
 * @return None.
 */
void conv(
    half *data_input,
    const int batch_size, const int channels, 
    const int height, const int width,
    half *data_kernel,
    const int num_kernels, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float *data_bias,
    float *data_output
);

#endif /* _GROUP14_CONV_HPP_ */