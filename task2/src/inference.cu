#include <cuda_runtime_api.h>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <string>

#include "conv.hpp"
#include "im2col.hpp"
#include "model_utils.hpp"
#include "data_utils.hpp"
#include "float_half_trans.hpp"
#include "batch_add.hpp"
#include "gemm.hpp"
#include "mat_vec_add.hpp"
#include "common.hpp"
#include "padding.hpp"
#include "tensor_core.hpp"
#include <chrono>

/* Global Variable Begin */
std::chrono::duration<double> conv1_time{0};
std::chrono::duration<double> relu1_time{0};
std::chrono::duration<double> max_pool_time{0};
std::chrono::duration<double> block_time{0};
std::chrono::duration<double> mean_pool_time{0};
std::chrono::duration<double> linear_time{0};

std::chrono::duration<double> float2half_time{0};
std::chrono::duration<double> im2col_time{0};
std::chrono::duration<double> wmma_rbmm_time{0};
std::chrono::duration<double> batch_mat_vec_add_time{0};

std::chrono::duration<double> padding_time{0};
std::chrono::duration<double> rbmm_time{0};
std::chrono::duration<double> unpadding_time{0};
/* Global Varaiable End */

// Right Batch Matrix Multiplication Layer
class RBMM {
private:
    int bsz;    // batch size

    int m;
    int k;
    int n;

    int aligned_m;
    int aligned_k;
    int aligned_n;

    half *aligned_matrix_a;
    half *aligned_batch_matrix_b;
    float *aligned_batch_matrix_c;

    int num_warps;

    float * float_aligned_matrix_a;
    float * float_aligned_batch_matrix_b;

    float * sim_aligned_matrix_a;
    float * sim_aligned_matrix_b;
    float * sim_aligned_matrix_c;

    float * sim_aligned_matrix_d;
    simVolta::GPU volta;
    
public:
    RBMM(int batch_size, int m, int k, int n) {
        this->bsz = batch_size;
        this->m = m;
        this->k = k;
        this->n = n;

        // align m, k, n to TC_TILE_WIDTH (16)
        aligned_m = ALIGN(m, TC_TILE_WIDTH);
        aligned_k = ALIGN(k, TC_TILE_WIDTH);
        aligned_n = ALIGN(n, TC_TILE_WIDTH);

        num_warps = (aligned_m / TC_TILE_WIDTH) * (aligned_n / TC_TILE_WIDTH);
    }

    void forward(half *matrix_a, half *batch_matrix_b, float *batch_matrix_c) {
        // padding matrix_a
        auto padding_start = std::chrono::high_resolution_clock::now();
        batch_padding(matrix_a, aligned_matrix_a, m, k, aligned_m, aligned_k, 1);
        batch_padding(batch_matrix_b, aligned_batch_matrix_b, k, n, aligned_k, aligned_n, bsz);
        auto padding_end = std::chrono::high_resolution_clock::now();
        padding_time += padding_end - padding_start;
        
        dim3 dimGrid, dimBlock;
        if (num_warps * WARP_SIZE > CUDA_NUM_THREADS) {
            dimGrid.x = (num_warps * WARP_SIZE + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
            dimBlock.x = CUDA_NUM_THREADS;
        } else {
            dimGrid.x = 1;
            dimBlock.x = num_warps * WARP_SIZE;
        }
        dimGrid.y = bsz;

        auto rbmm_start = std::chrono::high_resolution_clock::now();

        half2float(aligned_matrix_a, float_aligned_matrix_a, aligned_m * aligned_k);
        half2float(aligned_batch_matrix_b, float_aligned_batch_matrix_b, bsz * aligned_k * aligned_n);
        cudaMemcpy(sim_aligned_matrix_a, float_aligned_matrix_a, sizeof(float) * aligned_m * aligned_k, cudaMemcpyDeviceToHost);

        simVolta::dim3 simDimGrid, simDimBlock;
        for (int b = 0; b < bsz; ++b) {
            // cudaMemcpy(sim_aligned_matrix_c, aligned_batch_matrix_c + b * aligned_m * aligned_n, sizeof(float) * aligned_m * aligned_n, cudaMemcpyDeviceToHost);
            cudaMemcpy(sim_aligned_matrix_b, float_aligned_batch_matrix_b + b * aligned_k * aligned_n, sizeof(float) * aligned_k * aligned_n, cudaMemcpyDeviceToHost);      
            //  call the gemm function
            simVolta::sim_gemm(sim_aligned_matrix_a, sim_aligned_matrix_b, sim_aligned_matrix_c, sim_aligned_matrix_d, aligned_m, aligned_k, aligned_n, volta, simDimGrid, simDimBlock);
            cudaMemcpy(aligned_batch_matrix_c + b * aligned_m * aligned_n, sim_aligned_matrix_c, sizeof(float) * aligned_m * aligned_n, cudaMemcpyHostToDevice);
        }
        auto rbmm_end = std::chrono::high_resolution_clock::now();
        rbmm_time += rbmm_end - rbmm_start;

        // unpadding batch_matrix_c
        auto unpadding_start = std::chrono::high_resolution_clock::now();
        batch_unpadding(aligned_batch_matrix_c, batch_matrix_c, m, n, aligned_m, aligned_n, bsz);
        auto unpadding_end = std::chrono::high_resolution_clock::now();
        unpadding_time += unpadding_end - unpadding_start;
    }

    void cuda() {
        MALLOC_ERR_DECLARATION;
        simVolta::simMalloc((void **)&sim_aligned_matrix_a, sizeof(float) * aligned_m * aligned_k, volta);
        simVolta::simMalloc((void **)&sim_aligned_matrix_b, sizeof(float) * aligned_k * aligned_n, volta);
        simVolta::simMalloc((void **)&sim_aligned_matrix_c, sizeof(float) * aligned_m * aligned_n, volta);
        
        cudaMalloc((void **)&aligned_matrix_a, sizeof(half) * aligned_m * aligned_k);
        cudaMalloc((void **)&aligned_batch_matrix_b, sizeof(half) * aligned_k * aligned_n * bsz);
        cudaMalloc((void **)&aligned_batch_matrix_c, sizeof(float) * aligned_m * aligned_n * bsz);
        
        cudaMalloc((void **)&float_aligned_matrix_a, sizeof(float) * aligned_m * aligned_k);
        cudaMalloc((void **)&float_aligned_batch_matrix_b, sizeof(float) * aligned_k * aligned_n * bsz);
        
        CUDA_POST_MALLOC_CHECK;
    }

    ~RBMM() {
        volta.SIM_EXIT_INSTR();
        cudaFree(aligned_matrix_a);
        cudaFree(aligned_batch_matrix_b);
        cudaFree(aligned_batch_matrix_c);
        cudaFree(float_aligned_matrix_a);
        cudaFree(float_aligned_batch_matrix_b);
    }
};

class Conv2d{
private:
    int stride;
    int kernel_h;
    int kernel_w;
    int input_h;
    int input_w;
    int input_channel;
    int output_channel;
    int pad_h;
    int pad_w;
    int dilation;
    int bsz;
    bool on_cuda;
    float *cuda_weight;
    float *cuda_bias;

    int output_h;
    int output_w;
    int col_size;
    half *cuda_weight_half;
    half *cuda_input_half;
    half *cuda_col_half;
    float *cuda_intermediate_mult_res;

    RBMM *rbmm;
    
public:
    Conv2d(int batch_size, int in_dim, int out_dim, int in_h, int in_w, int pad = -1, int stride = 1, int kh = 3, int kw = 3) {
        bsz = batch_size;
        kernel_h = kh;
        kernel_w = kw;
        input_channel = in_dim;
        output_channel = out_dim;
        input_h = in_h;
        input_w = in_w;
        if (pad == -1) {
            pad_h = kernel_h / 2;
            pad_w = kernel_w / 2;
        }
        else {
            pad_h = pad;
            pad_w = pad;
        }
        this->stride = stride;
        this->dilation = 1;
        on_cuda = false;

        // calculate output height and width
        int eq_kernel_h = kernel_h + (kernel_h - 1) * (dilation - 1);   // equalized kernel height
        int eq_kernel_w = kernel_w + (kernel_w - 1) * (dilation - 1);   // equalized kernel width
        output_h = (input_h + 2 * pad_h - eq_kernel_h) / stride + 1;
        output_w = (input_w + 2 * pad_w - eq_kernel_w) / stride + 1;
        col_size = bsz * input_channel * kernel_h * kernel_w * output_h * output_w;

        rbmm = new RBMM(bsz, output_channel, input_channel * kernel_h * kernel_w, output_h * output_w);
    }

    void load_from_statedict(std::unordered_map<std::string, float *>& state_dict) {
        MALLOC_ERR_DECLARATION;
        cudaMemcpy(cuda_weight, state_dict["weight"], sizeof(float) * input_channel * output_channel * kernel_h * kernel_w, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_bias, state_dict["bias"], sizeof(float) * output_channel, cudaMemcpyHostToDevice);
        float2half(cuda_weight, cuda_weight_half, input_channel * output_channel * kernel_h * kernel_w);
        CUDA_POST_MALLOC_CHECK;
    }

    void forward(float* input_data, float* output_data) {
        if (on_cuda) {
            // input float to half
            auto float2half_start = std::chrono::high_resolution_clock::now();
            float2half(input_data, cuda_input_half, bsz * input_channel * input_h * input_w);
            auto float2half_end = std::chrono::high_resolution_clock::now();
            float2half_time += float2half_end - float2half_start;

            // im2col
            auto im2col_start = std::chrono::high_resolution_clock::now();
            im2col_gpu(
                cuda_input_half,
                bsz, input_channel, input_h, input_w,
                kernel_h, kernel_w,
                pad_h, pad_w,
                stride, stride,
                dilation, dilation,
                cuda_col_half
            );
            auto im2col_end = std::chrono::high_resolution_clock::now();
            im2col_time += im2col_end - im2col_start;

            // gemm
            auto wmma_rbmm_start = std::chrono::high_resolution_clock::now();
            rbmm->forward(cuda_weight_half, cuda_col_half, cuda_intermediate_mult_res);
            auto wmma_rbmm_end = std::chrono::high_resolution_clock::now();
            wmma_rbmm_time += wmma_rbmm_end - wmma_rbmm_start;

            // add bias
            auto batch_mat_vec_add_start = std::chrono::high_resolution_clock::now();
            batch_mat_vec_add<float>(
                cuda_intermediate_mult_res, cuda_bias, output_data,
                output_channel, output_h * output_w, bsz
            );
            auto batch_mat_vec_add_end = std::chrono::high_resolution_clock::now();
            batch_mat_vec_add_time += batch_mat_vec_add_end - batch_mat_vec_add_start;

            CUDA_POST_KERNEL_CHECK;
        } else {
            printf("ERROR: not on cuda\n");
        }
    }

    void cuda() {
        MALLOC_ERR_DECLARATION;
        cudaMalloc((void**)&cuda_weight, sizeof(float) * input_channel * output_channel * kernel_h * kernel_w);
        cudaMalloc((void**)&cuda_bias, sizeof(float) * output_channel);
        cudaMalloc((void**)&cuda_weight_half, sizeof(half) * input_channel * output_channel * kernel_h * kernel_w);   
        cudaMalloc((void**)&cuda_input_half, sizeof(half) * bsz * input_channel * input_h * input_w);
        cudaMalloc((void**)&cuda_col_half, sizeof(half) * col_size);
        cudaMalloc((void**)&cuda_intermediate_mult_res, sizeof(float) * bsz * output_channel * output_h * output_w);
        rbmm->cuda();
        CUDA_POST_MALLOC_CHECK;
        on_cuda = true;
    }

    ~Conv2d() {
        cudaFree(cuda_weight);
        cudaFree(cuda_bias);
        cudaFree(cuda_weight_half);
        cudaFree(cuda_input_half);
        cudaFree(cuda_col_half);
        cudaFree(cuda_intermediate_mult_res);

        // delete rbmm;
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
    int height;
    int width;

    float * identity, *out;
public:
    BasicBlock(int batch_size, int in_channel, int out_channel, int stride=1, int __height=-1, int __width=-1, Conv2d** downsample_module=NULL) {
        if (downsample_module != NULL) {
            downsample = *downsample_module;
        }
        else {
            downsample = NULL;
        }
        height = __height;
        width = __width;
        bsz = batch_size;
        this->in_channel = in_channel;
        this->out_channel = out_channel;
        this->stride = stride;
        output_h = height / stride;
        output_w = width / stride;

        conv1 = new Conv2d(bsz, in_channel, out_channel, height, width, -1, stride);
        relu = new ReLU(bsz, out_channel);
        conv2 = new Conv2d(bsz, out_channel, out_channel, output_h, output_w);  
    }
    void forward(float * input, float * output) {
        conv1->forward(input, out);
        relu->forward(out, out, output_h, output_w);
        conv2->forward(out, output);
        if (downsample != NULL) {
            downsample->forward(input, identity);
        } else {
            cudaMemcpy(identity, input, sizeof(float) * bsz * out_channel * output_h * output_w , cudaMemcpyDeviceToDevice);
        }
        batch_add(output, identity, output, bsz, out_channel, output_h, output_w);
        relu->forward(output, output, output_h, output_w);

    }
    
    void cuda() {
        if (downsample != NULL) {
            downsample->cuda();
        }
        relu->cuda();
        conv1->cuda();
        conv2->cuda();
        cudaMalloc((void**)&identity, sizeof(float) * bsz * out_channel * output_h * output_w);
        cudaMalloc((void**)&out, sizeof(float) * bsz * out_channel * output_h * output_w);
        on_cuda = true;
    }
    ~BasicBlock(){
        cudaFree(identity);
        cudaFree(out);
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
    int height;
    int width;

    half * temp_input;

    RBMM *rbmm;
public:
    Linear(int batch_size, int in_channel, int out_channel, int __height, int __width) {
        bsz = batch_size;
        this->in_channel = in_channel;
        this->out_channel = out_channel;
        height = __height;
        width = __width;
        rbmm = new RBMM(bsz, out_channel, in_channel, height * width);
    }
    void load_from_statedict(std::unordered_map<std::string, float *>& state_dict) {
        cudaMemcpy(cuda_weight, state_dict["weight"], sizeof(float) * in_channel * out_channel, cudaMemcpyHostToDevice);
        for (int i = 0; i < bsz; ++i)
            cudaMemcpy(cuda_bias + i * out_channel, state_dict["bias"], sizeof(float) * out_channel, cudaMemcpyHostToDevice);
        float2half(cuda_weight, cuda_weight_half, in_channel * out_channel);
    }
    void forward(float * input, float * output) {
        float2half(input, temp_input, bsz * in_channel * height * width);
        rbmm->forward(cuda_weight_half, temp_input, output);
        batch_add(output, cuda_bias, output, bsz, out_channel, 1, 1);
    }
    void cuda() {
        MALLOC_ERR_DECLARATION;
        cudaMalloc((void**)&temp_input, sizeof(half) * bsz * in_channel * height * width);
        cudaMalloc((void**)&cuda_weight, sizeof(float) * in_channel * out_channel);
        cudaMalloc((void**)&cuda_bias, bsz * sizeof(float) * out_channel);
        cudaMalloc((void**)&cuda_weight_half, sizeof(half) * in_channel * out_channel);
        rbmm->cuda();
        CUDA_POST_MALLOC_CHECK;
    }
    ~Linear() {
        cudaFree(temp_input);
        cudaFree(cuda_weight);
        cudaFree(cuda_bias);    
        cudaFree(cuda_weight_half);
        // delete rbmm;
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
        float **out_list;   // intermediate output

    public:
        ResNet18(int batch_size, int num_classes=1000) {
            bsz = batch_size;
            conv1 = new Conv2d(batch_size, 3, 64, 224, 224, -1, 2, 7, 7);
            relu1 = new ReLU(batch_size, 64);
            block = new BasicBlock*[8];
            downsample1 = new Conv2d(batch_size, 64, 128, 56, 56, 0, 2, 1, 1);
            downsample2 = new Conv2d(batch_size, 128, 256, 28, 28, 0, 2, 1, 1);
            downsample3 = new Conv2d(batch_size, 256, 512, 14, 14, 0, 2, 1, 1);

            block[0] = new BasicBlock(batch_size, 64, 64, 1, height_list[1], width_list[1]);
            block[1] = new BasicBlock(batch_size, 64, 64, 1, height_list[2], width_list[2]);
            block[2] = new BasicBlock(batch_size, 64, 128, 2, height_list[3], width_list[3], &downsample1);
            block[3] = new BasicBlock(batch_size, 128, 128, 1, height_list[4], width_list[4]);
            block[4] = new BasicBlock(batch_size, 128, 256, 2, height_list[5], width_list[5], &downsample2);
            block[5] = new BasicBlock(batch_size, 256, 256, 1, height_list[6], width_list[6]);
            block[6] = new BasicBlock(batch_size, 256, 512, 2, height_list[7], width_list[7], &downsample3);
            block[7] = new BasicBlock(batch_size, 512, 512, 1, height_list[8], width_list[8]);
            
            output_project = new Linear(batch_size, 512, num_classes, 1, 1);
            out_list = new float*[11];
        }

        void forward(float * input, float * output, int height, int width) {
            auto conv_start = std::chrono::high_resolution_clock::now();
            conv1->forward(input, out_list[0]);
            auto conv_end = std::chrono::high_resolution_clock::now();
            conv1_time += conv_end - conv_start;

            auto relu1_start = std::chrono::high_resolution_clock::now();
            relu1->forward(out_list[0], out_list[0], height_list[0], width_list[0]);
            auto relu1_end = std::chrono::high_resolution_clock::now();
            relu1_time += relu1_end - relu1_start;

            auto maxpool_start = std::chrono::high_resolution_clock::now();
            max_pool_2d(out_list[0], out_list[1], bsz, out_channels[1], height_list[0], width_list[0], 3, 3, 1, 2);
            auto maxpool_end = std::chrono::high_resolution_clock::now();
            max_pool_time += maxpool_end - maxpool_start;

            auto block_start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 8; ++i) {
                block[i]->forward(
                    out_list[i+1], out_list[i+2]
                );
            }
            auto block_end = std::chrono::high_resolution_clock::now();
            block_time += block_end - block_start;
            
            auto adaptive_mean_pool_start = std::chrono::high_resolution_clock::now();
            adaptive_mean_pool(out_list[9], out_list[10], bsz, out_channels[9], height_list[9], width_list[9]);
            auto adaptive_mean_pool_end = std::chrono::high_resolution_clock::now();
            mean_pool_time += adaptive_mean_pool_end - adaptive_mean_pool_start;

            auto linear_start = std::chrono::high_resolution_clock::now();
            output_project->forward(out_list[10], output);
            auto linear_end = std::chrono::high_resolution_clock::now();
            linear_time += linear_end - linear_start;
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
            for (int i = 0; i < 10; ++i) {
                cudaMalloc((void**)&(out_list[i]), sizeof(float) * bsz * out_channels[i] * height_list[i] * width_list[i]);
            }
            cudaMalloc((void**)&(out_list[10]), sizeof(float) * bsz * out_channels[9]);
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
            for (int i = 0; i < 11; ++i) {
                cudaFree(out_list[i]);
            }
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
    int batch_size_setting = batch_size;
    const int channels = 3;
    const int height = 224;
    const int width = 224;
    const int num_classes = 1000;
    ResNet18 model(batch_size);

    std::chrono::duration<double> model_cuda_init_time{0};
    auto model_cuda_init_start = std::chrono::high_resolution_clock::now();
    model.cuda();
    auto model_cuda_init_end = std::chrono::high_resolution_clock::now();
    model_cuda_init_time = model_cuda_init_end - model_cuda_init_start;

    std::chrono::duration<double> model_load_time{0};
    auto model_load_start = std::chrono::high_resolution_clock::now();
    model.load_from_statedict(state_dict);
    auto model_load_end = std::chrono::high_resolution_clock::now();
    model_load_time = model_load_end - model_load_start;

    std::chrono::duration<double> data_loader_init_time{0};
    auto data_loader_init_start = std::chrono::high_resolution_clock::now();
    DataLoader dt = DataLoader("/home/group14/CS433FinalProject/task2/select_file_list.txt", batch_size);
    // DataLoader dt = DataLoader("/home/group14/CS433FinalProject/task1/error_file_list.txt", batch_size);
    auto data_loader_init_end = std::chrono::high_resolution_clock::now();
    data_loader_init_time = data_loader_init_end - data_loader_init_start;

    printf("DataLoader initialized\n");

    std::ofstream file("/home/group14/CS433FinalProject/task2/target/output/predictions.txt");
    // std::ofstream file("error_list_predictions.txt");

    std::vector<std::string> file_list;
    float * batched_images = (float*)malloc(sizeof(float) * batch_size * channels * height * width);

    float * predictions;
    float * cuda_images;
    int * predicted_classes;
    std::chrono::duration<double> img_cuda_malloc_time{0};
    auto img_cuda_malloc_start = std::chrono::high_resolution_clock::now();
    cudaMalloc((void**)&cuda_images, sizeof(float) * batch_size * channels * height * width);
    auto img_cuda_malloc_end = std::chrono::high_resolution_clock::now();
    img_cuda_malloc_time = img_cuda_malloc_end - img_cuda_malloc_start;

    std::chrono::duration<double> output_malloc_time{0};
    auto output_malloc_start = std::chrono::high_resolution_clock::now();
    cudaMalloc((void**)&predicted_classes, sizeof(int) * batch_size);
    cudaMalloc((void**)&predictions, sizeof(float) * batch_size * num_classes);
    CUDA_POST_MALLOC_CHECK;
    int * host_predicted_classes = new int[batch_size];
    memset(host_predicted_classes, 0, sizeof(int) * batch_size);
    auto output_malloc_end = std::chrono::high_resolution_clock::now();
    output_malloc_time = output_malloc_end - output_malloc_start;

    std::chrono::duration<double> forward_time{0};
    std::chrono::duration<double> argmax_time{0};
    std::chrono::duration<double> cuda_copy_img_time{0};
    std::chrono::duration<double> cuda_copy_output_time{0};
    std::chrono::duration<double> save_output_to_file_time{0};
    while (dt.load_batch_data(batched_images, file_list, &batch_size) != -1) {
        // step 1: copy the images of the whole batch to cuda
        auto cuda_copy_img_start = std::chrono::high_resolution_clock::now();
        cudaMemcpy(cuda_images, batched_images, sizeof(float) * batch_size * channels * height * width, cudaMemcpyHostToDevice);
        auto cuda_copy_img_end = std::chrono::high_resolution_clock::now();
        cuda_copy_img_time += cuda_copy_img_end - cuda_copy_img_start;

        // step 2: forward
        auto forward_start = std::chrono::high_resolution_clock::now();
        model.forward(cuda_images, predictions, height, width);
        auto forward_end = std::chrono::high_resolution_clock::now();
        forward_time += forward_end - forward_start;

        // step 3: argmax
        auto argmax_start = std::chrono::high_resolution_clock::now();
        argmax(predictions, predicted_classes, batch_size, num_classes);
        auto argmax_end = std::chrono::high_resolution_clock::now();
        argmax_time += argmax_end - argmax_start;
        
        // step 4: copy the output to host
        auto cuda_copy_output_start = std::chrono::high_resolution_clock::now();
        cudaMemcpy(host_predicted_classes, predicted_classes, sizeof(int) * batch_size, cudaMemcpyDeviceToHost);
        auto cuda_copy_output_end = std::chrono::high_resolution_clock::now();
        cuda_copy_output_time += cuda_copy_output_end - cuda_copy_output_start;

        // step 5: save output to file
        auto save_output_to_file_start = std::chrono::high_resolution_clock::now(); 
        for (int i = 0; i < batch_size; ++i)
            file << file_list[i] << " " << std::to_string(host_predicted_classes[i]) << std::endl;
        auto save_output_to_file_end = std::chrono::high_resolution_clock::now();
        save_output_to_file_time += save_output_to_file_end - save_output_to_file_start;

        // printf("batch %d finished\n", ++batch_num);
    }

    auto total_time = model_cuda_init_time + model_load_time + data_loader_init_time + img_cuda_malloc_time + output_malloc_time + cuda_copy_img_time + forward_time + argmax_time + cuda_copy_output_time + save_output_to_file_time;
    printf("-- batch size: %d --\n", batch_size_setting);
    printf("model cuda init time: %fs, %f%%\n", model_cuda_init_time.count(), model_cuda_init_time.count() / total_time.count() * 100);
    printf("model load time: %fs, %f%%\n", model_load_time.count(), model_load_time.count() / total_time.count() * 100);
    printf("data loader init time: %fs, %f%%\n", data_loader_init_time.count(), data_loader_init_time.count() / total_time.count() * 100);
    printf("img cuda malloc time: %fs, %f%%\n", img_cuda_malloc_time.count(), img_cuda_malloc_time.count() / total_time.count() * 100);
    printf("output malloc time: %fs, %f%%\n", output_malloc_time.count(), output_malloc_time.count() / total_time.count() * 100);
    printf("cuda copy img time: %fs, %f%%\n", cuda_copy_img_time.count(), cuda_copy_img_time.count() / total_time.count() * 100);
    printf("forward time: %fs, %f%%\n", forward_time.count(), forward_time.count() / total_time.count() * 100);
    printf("argmax time: %fs, %f%%\n", argmax_time.count(), argmax_time.count() / total_time.count() * 100);
    printf("cuda copy output time: %fs, %f%%\n", cuda_copy_output_time.count(), cuda_copy_output_time.count() / total_time.count() * 100);
    printf("save output to file time: %fs, %f%%\n", save_output_to_file_time.count(), save_output_to_file_time.count() / total_time.count() * 100);
    printf("total time: %fs\n", total_time.count());

    // printf("--- forward time profile ---\n");
    // printf("conv1 time: %fs\n", conv1_time.count());
    // printf("relu1 time: %fs\n", relu1_time.count());
    // printf("maxpool time: %fs\n", max_pool_time.count());
    // printf("block time: %fs\n", block_time.count());
    // printf("mean pool time: %fs\n", mean_pool_time.count());
    // printf("linear time: %fs\n", linear_time.count());

    // printf("--- conv time profile ---\n");
    // printf("float2half time: %fs\n", float2half_time.count());
    // printf("im2col time: %fs\n", im2col_time.count());
    // printf("wmma_rbmm time: %fs\n", wmma_rbmm_time.count());
    // printf("batch_mat_vec_add time: %fs\n", batch_mat_vec_add_time.count());

    // printf("--- rbmm time profile ---\n");
    // printf("padding time: %fs\n", padding_time.count());
    // printf("wmma_rbmm time: %fs\n", rbmm_time.count());
    // printf("unpadding time: %fs\n", unpadding_time.count());

    cudaFree(cuda_images);
    cudaFree(predictions);
    cudaFree(predicted_classes);
    delete [] host_predicted_classes;
    free(batched_images);
    file.close();

    return 0;
}