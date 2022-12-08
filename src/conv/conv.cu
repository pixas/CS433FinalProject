#include "im2col.hpp"
#include "gemm.hpp"
#include "mat_vec_add.hpp"
#include "float_half_trans.hpp"
#include "conv.hpp"

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
    float *data_input,
    const int batch_size, const int channels, 
    const int height, const int width,
    float *data_kernel,
    const int num_kernels, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float *data_bias,
    float *data_output
) {
    const int eq_kernel_h = kernel_h + (kernel_h - 1) * (dilation_h - 1);   // equalized kernel height
    const int eq_kernel_w = kernel_w + (kernel_w - 1) * (dilation_w - 1);   // equalized kernel width

    const int output_h = (height + 2 * pad_h - eq_kernel_h) / stride_h + 1; // output height (without consideration of channel)
    const int output_w = (width + 2 * pad_w - eq_kernel_w) / stride_w + 1;  // output weight (without consideration of channel)

    /* float2half begin */
    half *data_input_half, *data_kernel_half;
    cudaMalloc((void**)&data_input_half, batch_size * channels * height * width * sizeof(half));
    cudaMalloc((void**)&data_kernel_half, channels * kernel_h * kernel_w * num_kernels * sizeof(half));
    float2half(data_input, data_input_half, batch_size * channels * height * width);
    float2half(data_kernel, data_kernel_half, channels * kernel_h * kernel_w * num_kernels);
    /* float2half end */

    /* im2col begin */
    const int col_size = batch_size * channels * kernel_h * kernel_w * output_h * output_w; // size of columns
    half *data_col;
    cudaMalloc((void **)&data_col, sizeof(half) * col_size);

    im2col_gpu(
        data_input_half,
        batch_size, channels, 
        height, width,
        kernel_h, kernel_w,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        data_col
    );
    /* im2col end */

    /* matrix multiplication begin */
    float *tmp_mult_res;
    cudaMalloc((void **)&tmp_mult_res, sizeof(float) * batch_size * num_kernels * output_h * output_w);
    wmma_rbmm(
        data_kernel_half, data_col, tmp_mult_res,
        batch_size, num_kernels, channels * kernel_h * kernel_w, output_h * output_w
    );
    /* matrix multiplication end */

    /* add bias begin */
    batch_mat_vec_add<float>(
        tmp_mult_res, data_bias, data_output,
        num_kernels, output_h * output_w, batch_size
    );
    /* add bias end */

    cudaFree(tmp_mult_res);
    cudaFree(data_col);
    cudaFree(data_input_half);
    cudaFree(data_kernel_half);
}
