#include "common.hpp"
#include "mma.h"
#include "conv.hpp"
#include <iostream>

using namespace nvcuda;

using namespace std;

float kernels_o[3 * 27] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
};

float im[3 * 3 * 4 * 4] = {
    //img 1
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,

    //img 2
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,

    //img 3
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
};

float bias_o[3] = { 1, 2, 3 };

int main() {
    float *kernels = new float[3 * 27];
    float *im_h = new float[3 * 3 * 4 * 4];
    
    for (int i = 0; i < 3 * 27; i++) {
        kernels[i] = (kernels_o[i]);
    }
    for (int i = 0; i < 3 * 3 * 4 * 4; i++) {
        im_h[i] = (im[i]);
    }

    float *d_kernels, *d_im;
    float *d_bias;
    float *d_out;
    cudaMalloc(&d_kernels, 3 * 27 * sizeof(float));
    cudaMalloc(&d_im, 3 * 3 * 4 * 4 * sizeof(float));
    cudaMalloc(&d_bias, 3 * sizeof(float));
    cudaMalloc(&d_out, 3 * 3 * 2 * 2 * sizeof(float));

    cudaMemcpy(d_kernels, kernels, 3 * 27 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_im, im_h, 3 * 3 * 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias_o, 3 * sizeof(float), cudaMemcpyHostToDevice);

    conv(
        d_im, 
        3, 3, 4, 4,
        d_kernels,
        3, 3, 3,
        0, 0, 
        1, 1,
        1, 1,
        d_bias,
        d_out
    );

    float *out = new float[3 * 3 * 2 * 2];
    cudaMemcpy(out, d_out, 3 * 3 * 2 * 2 * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 3; ++i) {
        cout << "img" << i << endl;
        for (int j = 0; j < 3; ++j) {
            cout << "channel" << j << endl;
            for (int k = 0; k < 2; ++k) {
                for (int l = 0; l < 2; ++l) {
                    cout << out[i * 3 * 2 * 2 + j * 2 * 2 + k * 2 + l] << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;
    }

    cudaFree(d_kernels);
    cudaFree(d_im);
    cudaFree(d_bias);
    cudaFree(d_out);
}