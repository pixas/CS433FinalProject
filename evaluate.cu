// 存放整个模型，include 卷积等算子，有一个
#include <cuda_runtime_api.h>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <string>

#include "conv.hpp"
#include "model_utils.hpp"
#include "data_utils.hpp"
#include "float_half_trans.hpp"
#include "batch_add.hpp"
#include "gemm.hpp"
#include "mat_vec_add.hpp"
#include "common.hpp"

class Conv2d{
    private:
        int stride;
        int kernel_h;
        int kernel_w;
        int input_channel;
        int output_channel;
        int pad_h;
        int pad_w;
        int bsz;
        bool on_cuda;
        float *cuda_weight;
        float *cuda_bias;
    public:
        Conv2d(int batch_size, int in_dim, int out_dim, int pad = -1, int stride = 1, int kh = 3, int kw = 3) {
            this->stride = stride;
            kernel_h = kh;
            bsz = batch_size;
            kernel_w = kw;
            input_channel = in_dim;
            output_channel = out_dim;
            if (pad == -1) {
                pad_h = kernel_h / 2;
                pad_w = kernel_w / 2;
            }
            else {
                pad_h = pad;
                pad_w = pad;
            }
            cudaMalloc((void**)&cuda_weight, sizeof(float) * input_channel * output_channel * kernel_h * kernel_w);
            cudaMalloc((void**)&cuda_bias, sizeof(float) * output_channel);
            on_cuda = false;
        }

        void load_from_statedict(std::unordered_map<std::string, float *>& state_dict) {
            MALLOC_ERR_DECLARATION;
            cudaMemcpy(cuda_weight, state_dict["weight"], sizeof(float) * input_channel * output_channel * kernel_h * kernel_w, cudaMemcpyHostToDevice);
            cudaMemcpy(cuda_bias, state_dict["bias"], sizeof(float) * output_channel, cudaMemcpyHostToDevice);
            CUDA_POST_MALLOC_CHECK;
        }

        void forward(float* input_data, float* output_data, int height, int width) {
            if (on_cuda) {
                conv(
                    input_data, bsz, input_channel, height, width, cuda_weight,
                    output_channel, kernel_h, kernel_w, pad_h, pad_w, 
                    stride, stride, 1, 1, cuda_bias, output_data
                );
            } else {
                printf("ERROR: not on cuda\n");
            }
        }

        void cuda() {
            MALLOC_ERR_DECLARATION;
            cudaMalloc((void**)&cuda_weight, sizeof(float) * input_channel * output_channel * kernel_h * kernel_w);
            cudaMalloc((void**)&cuda_bias, sizeof(float) * output_channel);
            CUDA_POST_MALLOC_CHECK;
            on_cuda = true;
        }
        ~Conv2d() {
            cudaFree(cuda_weight);
            cudaFree(cuda_bias);
        }
};


class ReLU{
private:
    int bsz;
    int in_channel;
public:
    ReLU(int batch_size, int in_channel) {
        bsz = batch_size;
        this->in_channel = in_channel;
    }
    void forward(float * input, float * output, int height, int width) {
        relu(input, output, bsz, height, width, in_channel);
    }
    ~ReLU() {}
    void cuda() {}
};



class BasicBlock{
private:
    int bsz;
    int in_channel;
    int out_channel;
    Conv2d * downsample;
    int stride;
    Conv2d * conv1;
    ReLU * relu;
    Conv2d * conv2;
    bool on_cuda;
    int output_h;
    int output_w;
public:
    BasicBlock(int batch_size, int in_channel, int out_channel, int stride=1, Conv2d** downsample_module=NULL) {
        if (downsample_module != NULL) {
            downsample = *downsample_module;
        }
        else {
            downsample = NULL;
        }
        bsz = batch_size;
        this->in_channel = in_channel;
        this->out_channel = out_channel;
        this->stride = stride;
        conv1 = new Conv2d(bsz, in_channel, out_channel, -1, stride);
        relu = new ReLU(bsz, out_channel);
        conv2 = new Conv2d(bsz, out_channel, out_channel);  
    }
    void forward(float * input, float * output, int height, int width) {
        float * identity;
        float * out;
        output_h = height / stride;
        output_w = width / stride;

        MALLOC_ERR_DECLARATION;
        cudaMalloc((void**)&identity, sizeof(float) * bsz * out_channel * output_h * output_w);
        cudaMalloc((void**)&out, sizeof(float) * bsz * out_channel * output_h * output_w);
        CUDA_POST_MALLOC_CHECK;
        
        conv1->forward(input, out, height, width);
        relu->forward(out, out, output_h, output_w);
        conv2->forward(out, output, output_h, output_w);
        if (downsample != NULL) {
            downsample->forward(input, identity, height, width);
        } else {
            cudaMemcpy(identity, input, sizeof(float) * bsz * out_channel * output_h * output_w , cudaMemcpyDeviceToDevice);
        }
        batch_add(output, identity, output, bsz, out_channel, output_h, output_w);
        relu->forward(output, output, output_h, output_w);     
        cudaFree(identity);
           
    }
    int get_output_height() {
        return output_h;
    }    
    int get_output_width() {
        return output_w;
    }

    void cuda() {
        if (downsample != NULL) {
            downsample->cuda();
        }
        relu->cuda();
        conv1->cuda();
        conv2->cuda();
        on_cuda = true;
    }
    ~BasicBlock(){
        conv1->~Conv2d();
        // delete conv1;
        conv2->~Conv2d();
        // delete conv2;
        relu->~ReLU();
        // delete relu;
        // if (downsample != NULL) {
        //     downsample->~Conv2d();
        // }
    }
    void load_from_statedict(std::unordered_map<std::string, float *>& state_dict1, std::unordered_map<std::string, float *>& state_dict2) {
        conv1->load_from_statedict(state_dict1);
        conv2->load_from_statedict(state_dict2);
    }
};

class Linear{
private:
    int bsz;
    float * cuda_weight;
    half * cuda_weight_half;
    float * cuda_bias;
    int in_channel;
    int out_channel;
public:
    Linear(int batch_size, int in_channel, int out_channel) {
        bsz = batch_size;
        this->in_channel = in_channel;
        this->out_channel = out_channel;
    }
    void load_from_statedict(std::unordered_map<std::string, float *>& state_dict) {
        cudaMemcpy(cuda_weight, state_dict["weight"], sizeof(float) * in_channel * out_channel, cudaMemcpyHostToDevice);
        for (int i = 0; i < bsz; ++i)
            cudaMemcpy(cuda_bias + i * out_channel, state_dict["bias"], sizeof(float) * out_channel, cudaMemcpyHostToDevice);
        float2half(cuda_weight, cuda_weight_half, in_channel * out_channel);
    }
    void forward(float * input, float * output, int height, int width) {
        half * temp_input;

        cudaMalloc((void**)&temp_input, sizeof(half) * bsz * in_channel * height * width);
        
        float2half(input, temp_input, bsz * in_channel * height * width);
        wmma_rbmm(cuda_weight_half, temp_input, output, bsz, out_channel, in_channel, height * width);
        batch_add(output, cuda_bias, output, bsz, out_channel, 1, 1);

        cudaFree(temp_input);
    }
    void cuda() {
        MALLOC_ERR_DECLARATION;
        cudaMalloc((void**)&cuda_weight, sizeof(float) * in_channel * out_channel);
        cudaMalloc((void**)&cuda_bias, bsz * sizeof(float) * out_channel);
        cudaMalloc((void**)&cuda_weight_half, sizeof(half) * in_channel * out_channel);
        CUDA_POST_MALLOC_CHECK;
    }
    ~Linear() {
        cudaFree(cuda_weight);
        cudaFree(cuda_bias);    
        cudaFree(cuda_weight_half);
    }
};

class ResNet18{
    private:
        int bsz;
        Conv2d * conv1;
        ReLU * relu1;
        Conv2d * downsample1;
        Conv2d * downsample2;
        Conv2d * downsample3;
        BasicBlock ** block;
        Linear * output_project;
        int out_channels[10] = {
            64, 64,     // conv1, maxpool
            64, 64,     // block0, block1
            128, 128,   // block2, block3
            256, 256,   // block4, block5
            512, 512    // block6, block7
        };
        // output height of each layer
        int height_list[10] = {
            112, 56,    // conv1,  maxpool
            56, 56,     // block0, block1
            28, 28,     // block2, block3
            14, 14,     // block4, block5
            7, 7        // block6, block7
        };
        // output width of each layer
        int width_list[10] = {
            112, 56,    // conv1,  maxpool
            56, 56,     // block0, block1
            28, 28,     // block2, block3
            14, 14,     // block4, block5
            7, 7        // block6, block7
        };
    public:
        ResNet18(int batch_size, int num_classes=1000) {
            bsz = batch_size;
            conv1 = new Conv2d(batch_size, 3, 64, -1, 2, 7, 7);
            relu1 = new ReLU(batch_size, 64);
            block = new BasicBlock*[8];
            downsample1 = new Conv2d(batch_size, 64, 128, 0, 2, 1, 1);
            downsample2 = new Conv2d(batch_size, 128, 256, 0, 2, 1, 1);
            downsample3 = new Conv2d(batch_size, 256, 512, 0, 2, 1, 1);

            block[0] = new BasicBlock(batch_size, 64, 64);
            block[1] = new BasicBlock(batch_size, 64, 64);
            block[2] = new BasicBlock(batch_size, 64, 128, 2, &downsample1);
            block[3] = new BasicBlock(batch_size, 128, 128);
            block[4] = new BasicBlock(batch_size, 128, 256, 2, &downsample2);
            block[5] = new BasicBlock(batch_size, 256, 256);
            block[6] = new BasicBlock(batch_size, 256, 512, 2, &downsample3);
            block[7] = new BasicBlock(batch_size, 512, 512);
            
            output_project = new Linear(batch_size, 512, num_classes);
        }

        void forward(float * input, float * output, int height, int width) {
            float ** out_list = new float*[11];

            for (int i = 0; i < 10; ++i) {
                cudaMalloc((void**)&(out_list[i]), sizeof(float) * bsz * out_channels[i] * height_list[i] * width_list[i]);
            }
            cudaMalloc((void**)&(out_list[10]), sizeof(float) * bsz * out_channels[9]);
            conv1->forward(input, out_list[0], height, width);
            relu1->forward(out_list[0], out_list[0], height_list[0], width_list[0]);
            max_pool_2d(out_list[0], out_list[1], bsz, out_channels[1], height_list[0], width_list[0], 3, 3, 1, 2);
            for (int i = 0; i < 8; ++i) {
                block[i]->forward(
                    out_list[i+1], out_list[i+2],
                    height_list[i+1], width_list[i+1]
                );
                
            }
            adaptive_mean_pool(out_list[9], out_list[10], bsz, out_channels[9], height_list[9], width_list[9]);
            // {
            //     cudaMemcpy(debug, output, sizeof(float) * bsz * out_channels[9] * 1, cudaMemcpyDeviceToHost);
            //     // for (int i = 0; i < height_list[9]; ++i) {
            //     //     for (int j = 0; j < width_list[9]; ++j) {
            //     //         printf("%.3f%s", debug[0 + 0 + i * width_list[9] + j], (j == width_list[9] - 1 ? "\n": " "));
            //     //     }
            //     // }
            //     for (int i = 0; i < out_channels[9]; ++i) {
            //         printf("%.3f%s", debug[i], (i == out_channels[9] - 1? "\n" : " "));
            //     }

            // }
            output_project->forward(out_list[10], output, 1, 1);
            for (int i = 0; i < 10; ++i) {
                cudaFree(out_list[i]);
            }
        }

        void cuda() {
            conv1->cuda();
            relu1->cuda();
            downsample1->cuda();
            downsample2->cuda();
            downsample3->cuda();
            for (int i = 0; i < 8; ++i)
                block[i]->cuda();
            output_project->cuda();
        }

        ~ResNet18(){
            conv1->~Conv2d();
            delete conv1;
            relu1->~ReLU();
            delete relu1;
            downsample1->~Conv2d();
            delete downsample1;
            downsample2->~Conv2d();
            delete downsample2;
            downsample3->~Conv2d();
            delete downsample3;
            for (int i = 0; i < 8; ++i) {
                block[i]->~BasicBlock();
                delete block[i];
            }
            delete [] block;
            output_project->~Linear();
            delete output_project;
        }
        void load_from_statedict(std::unordered_map<std::string,std::unordered_map<std::string, float *>>& state_dict) {
            conv1->load_from_statedict(state_dict["onnx_node!Conv_0"]);
            downsample1->load_from_statedict(state_dict["onnx_node!Conv_16"]);
            downsample2->load_from_statedict(state_dict["onnx_node!Conv_27"]);
            downsample3->load_from_statedict(state_dict["onnx_node!Conv_38"]);
            output_project->load_from_statedict(state_dict["onnx_node!Gemm_48"]);

            block[0]->load_from_statedict(state_dict["onnx_node!Conv_3"], state_dict["onnx_node!Conv_5"]);
            block[1]->load_from_statedict(state_dict["onnx_node!Conv_8"], state_dict["onnx_node!Conv_10"]);
            block[2]->load_from_statedict(state_dict["onnx_node!Conv_13"], state_dict["onnx_node!Conv_15"]);
            block[3]->load_from_statedict(state_dict["onnx_node!Conv_19"], state_dict["onnx_node!Conv_21"]);
            block[4]->load_from_statedict(state_dict["onnx_node!Conv_24"], state_dict["onnx_node!Conv_26"]);
            block[5]->load_from_statedict(state_dict["onnx_node!Conv_30"], state_dict["onnx_node!Conv_32"]);
            block[6]->load_from_statedict(state_dict["onnx_node!Conv_35"], state_dict["onnx_node!Conv_37"]);
            block[7]->load_from_statedict(state_dict["onnx_node!Conv_41"], state_dict["onnx_node!Conv_43"]);
        }
};


int main(int argc, char const *argv[]) {
    MALLOC_ERR_DECLARATION;
    cv::String model_name = "resnet18.onnx";
    std::unordered_map<std::string, std::unordered_map<std::string, float *>> state_dict = obtain_layer_info(model_name);
    int batch_size = atoi(argv[1]);
    const int channels = 3;
    const int height = 224;
    const int width = 224;
    const int num_classes = 1000;
    ResNet18 model(batch_size);
    model.cuda();
    CUDA_POST_MALLOC_CHECK;
    model.load_from_statedict(state_dict);
    DataLoader dt = DataLoader("/home/group14/CS433FinalProject/select_file_list.txt", batch_size);
    printf("DataLoader initialized\n");

    std::ofstream file("predictions.txt");

    std::vector<std::string> file_list;
    float * batched_images = (float*)malloc(sizeof(float) * batch_size * channels * height * width);

    float * predictions;
    float * cuda_images;
    int * predicted_classes;
    int * host_predicted_classes = new int[batch_size];
    // float * debug = new float[1 * 512 * 1 * 1];
    cudaMalloc((void**)&predictions, sizeof(float) * batch_size * num_classes);
    cudaMalloc((void**)&cuda_images, sizeof(float) * batch_size * channels * height * width);
    cudaMalloc((void**)&predicted_classes, sizeof(int) * batch_size);
    CUDA_POST_MALLOC_CHECK;
    memset(host_predicted_classes, 0, sizeof(int) * batch_size);

        while (dt.load_batch_data(batched_images, file_list, &batch_size) != -1) {
            // printf("IMG\n");
            // for (int i = 0; i < channels; ++i) {
            //     printf("channel %d\n", i);
            //     for (int j = 0; j< height; ++j) {
            //         for (int k = 0; k < width; ++k)
            //             printf("%f, ", batched_images[i * height * width + j * width + k]);
            //         printf("\n");
            //     }
            //     printf("\n");
            // }
            cudaMemcpy(cuda_images, batched_images, sizeof(float) * batch_size * channels * height * width, cudaMemcpyHostToDevice);
            // add a timing module to wrap forward
            model.forward(cuda_images, predictions, height, width);
            // float * temp = new float[batch_size * num_classes];
            // cudaMemcpy(temp, predictions, sizeof(float) * batch_size * num_classes, cudaMemcpyDeviceToHost);
            // for (int i = 0; i < batch_size; ++i) {
            //     float max_val = temp[i * num_classes];
            //     int max_idx = 0;
            //     for (int j = 0; j < num_classes; ++j) {
            //         if (temp[i * num_classes + j] > max_val) {
            //             max_val = temp[i * num_classes + j];
            //             max_idx = j;
            //         }
            //     }
            //     printf("%d %d %f\n", i, max_idx, max_val);
            // }
            // add a timing module to wrap forward
            argmax(predictions, predicted_classes, batch_size, num_classes);
            cudaMemcpy(host_predicted_classes, predicted_classes, sizeof(int) * batch_size, cudaMemcpyDeviceToHost);
            for (int i = 0; i < batch_size; ++i) {
                // printf("%d %d\n", i, host_predicted_classes[i]);
                file << file_list[i] << " " << std::to_string(host_predicted_classes[i]) << std::endl;
            }
            // break;
        }

    
    cudaFree(cuda_images);
    cudaFree(predictions);
    cudaFree(predicted_classes);
    delete [] host_predicted_classes;
    free(batched_images);
    file.close();

    return 0;
}