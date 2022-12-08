// 存放整个模型，include 卷积等算子，有一个
#include "include/conv.hpp"
#include "include/model_utils.hpp"
#include "include/data_utils"
#include <cuda_runtime_api.h>
#include <unordered_map>
#include "include/float_half_trans.hpp"

class Conv{
    private:
        // half * weight;
        float *weight;
        float *bias;
        int stride;
        int kernel_h;
        int kernel_w;
        int input_channel;
        int output_channel;
        int pad_h;
        int pad_w;
        int bsz;
        int pad_w;
        bool on_cuda;
        // half *cuda_weight;
        float *cuda_weight;
        float *cuda_bias;
    public:
        Conv(int batch_size, int in_dim, int out_dim, int stride=1, int kh=3, int kw=3) {
            this->stride = stride;
            kernel_h = kh;
            bsz = batch_size;
            kernel_w = kw;
            input_channel = in_dim;
            output_channel = out_dim;
            pad_h = kernel_h / 2;
            pad_w = kernel_w / 2;
            weight = new float[input_channel * output_channel * kernel_h * kernel_w];
            bias = new float[output_channel];
            on_cuda = false;
        }

        void load_from_statedict(std::unordered_map<std::string, float *>& state_dict) {
            // float2half(state_dict["weight"], weight, input_channel * output_channel * kernel_h * kernel_w);
            memcpy(weight, state_dict["weight"], sizeof(float) * input_channel * output_channel * kernel_h * kernel_w);
            memcpy(bias, state_dict["bias"], sizeof(float) * output_channel);
        }

        void forward(float* input_data, float* output_data, int height, int width) {
            if (on_cuda) {
                conv(input_data, bsz, input_channel, height, width, cuda_weight,
                    output_channel, kernel_h, kernel_w, pad_h, pad_w, 
                    stride, stride, 1, 1, cuda_bias, output_data);
            }
            else {
                conv(input_data, bsz, input_channel, height, width, weight,
                output_channel, kernel_h, kernel_w, pad_h, pad_w, 
                stride, stride, 1, 1, bias, output_data);

            }
        }

        void cuda() {
            float * cuda_weight;
            cudaMalloc((void**)&cuda_weight, sizeof(float) * input_channel * output_channel * kernel_h * kernel_w);
            cudaMemcpy(cuda_weight, weight, sizeof(float) * input_channel * output_channel * kernel_h * kernel_w, cudaMemcpyHostToDevice);

            float * cuda_bias;
            cudaMalloc((void**)&cuda_bias, sizeof(float) * output_channel);
            cudaMemcpy(cuda_bias, bias, sizeof(float) * output_channel, cudaMemcpyHostToDevice);
            on_cuda = true;
        }
        ~Conv() {
            if (on_cuda) {
                cudaFree(cuda_weight);
                cudaFree(cuda_bias);
            }
            delete [] bias;
            delete [] weight;
        }
}


class ResNet18{
    private:
        
};