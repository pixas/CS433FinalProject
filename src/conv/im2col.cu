#include "im2col.hpp"
#include "common.hpp"
#include <iostream>

/* Reference: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu */

/**
 * @brief im2col kernel for 2d convolution
 * @param n             number of points in output col
 * @param data_im       input img data (whole batch), allocated on GPU  shape (batch_size, channels, height, width)
 * @param channels      number of channels in input img
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
__global__ void im2col_gpu_kernel(
    const int n, half* data_im,
    const int channels, const int height, const int width, 
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    half* data_col
) {
    int batch_idx = blockIdx.y;
    data_im += batch_idx * channels * height * width;
    data_col += batch_idx * channels * kernel_h * kernel_w * height_col * width_col;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        const int h_idx = idx / width_col;              // height index in col in kernel units                  range [0, kernel_h * channels)
        const int h_col = h_idx % height_col;           // height index of the channel in col in kernel units   range [0, kernel_h)
        const int w_col = idx % width_col;              // width index in col in kernel units                   range [0, kernel_w)
        const int c_im = h_idx / height_col;            // channel index in img in kernel units                 range [0, channels)
        const int c_col = c_im * kernel_h * kernel_w;   // channel index in col in pixel units                  range [0, (channels - 1) * kernel_h * kernel_w]
        const int h_offset = h_col * stride_h - pad_h;  // height offset of the channel in img in pixel units   range [0, height)
        const int w_offset = w_col * stride_w - pad_w;  // width offset in img in pixel units                   range [0, width)

        half *data_col_ptr = data_col + (c_col * height_col + h_col) * width_col + w_col;  // pointer to the start pixel of the kernel of the thread in col
        half *data_im_ptr = data_im + (c_im * height + h_offset) * width + w_offset;       // pointer to the start pixel of the channel of the thread in img 
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                int h_im = h_offset + i * dilation_h;   // height index of the pixel in img in pixel units
                int w_im = w_offset + j * dilation_w;   // width index of the pixel in img in pixel units

                *data_col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ? data_im_ptr[i * dilation_h * width + j * dilation_w] : __float2half(0);   // check if the index is valid
                data_col_ptr += height_col * width_col; // move to the next pixel in col
            }
        }
    }
}


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
) {
    const int eq_kernel_h = kernel_h + (kernel_h - 1) * (dilation_h - 1);       // equivalent kernel height
    const int eq_kernel_w = kernel_w + (kernel_w - 1) * (dilation_w - 1);       // equivalent kernel width
    // kernel units: height = kernel_h * kernel_w, width = 1
    const int height_col = (height + 2 * pad_h - eq_kernel_h) / stride_h + 1;   // output col height in kernel units
    const int width_col = (width + 2 * pad_w - eq_kernel_w) / stride_w + 1;     // output col width in kernel units
    const int num_kernels = channels * height_col * width_col;                  // number of kernels in output col

    // printf("height_col: %d, width_col: %d, num_kernels: %d\n", height_col, width_col, num_kernels);

    dim3 grid(CUDA_GET_BLOCKS(num_kernels), batch_size);

    im2col_gpu_kernel<<<grid, CUDA_NUM_THREADS>>>(
        num_kernels, data_im,
        channels, height, width,
        kernel_h, kernel_w,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        height_col, width_col,
        data_col
    );
    cudaPeekAtLastError();
}

// float im[3][3][16] = {
//     // img 0, batch_idx = 0
//     {
//         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 
//         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 
//         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 
//     },
//     // img 1, batch_idx = 1
//     {
//         {49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64}, 
//         {65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80}, 
//         {81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96}
//     },
//     // img 2, batch_idx = 2
//     {
//         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 
//         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 
//         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, 
//     }
// };

// int main() {
//     half im_half[3][3][16];
//     for (int i = 0; i < 3; i++) {
//         for (int j = 0; j < 3; j++) {
//             for (int k = 0; k < 16; k++) {
//                 im_half[i][j][k] = __float2half(im[i][j][k]);
//             }
//         }
//     }

//     half *im_gpu;
//     cudaMalloc(&im_gpu, sizeof(half) * 3 * 3 * 16);
//     cudaMemcpy(im_gpu, im_half, sizeof(half) * 3 * 3 * 16, cudaMemcpyHostToDevice);

//     half *col_gpu;
//     cudaMalloc(&col_gpu, sizeof(half) * 3 * 108);

//     im2col_gpu(
//         im_gpu, 
//         3, 3,   // batch_size, channels
//         4, 4,   // height, width
//         3, 3,   // kernel_h, kernel_w
//         0, 0,   // pad_h, pad_w
//         1, 1,   // stride_h, stride_w
//         1, 1,   // dilation_h, dilation_w
//         col_gpu
//     );

//     half col[3][108];
//     cudaMemcpy(col, col_gpu, sizeof(half) * 3 * 108, cudaMemcpyDeviceToHost);
//     for (int i = 0; i < 3; i++) {
//         for (int j = 0; j < 27; j++) {
//             for (int k = 0; k < 4; k++) {
//                 printf("%f ", __half2float(col[i][j * 4 + k]));
//             }
//             printf("\n");
//         }
//         printf("\n");
//     }

//     cudaFree(im_gpu);
//     cudaFree(col_gpu);

//     return 0;
// }