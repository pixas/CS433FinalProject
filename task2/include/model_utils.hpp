#ifndef _GROUP14_MODEL_UTILS_HPP_
#define _GROUP14_MODEL_UTILS_HPP_


/**
 * @brief computes relu activation function for data_input
 * @param input input tensor
 * @param output output tensor
 * @param batch_size batch size for tensor
 * @param height    matrix height
 * @param width     matrix width
 * 
 */
template<typename T>
void relu(T *input, T *output, int batch_size, int height, int width, int input_dim);


/**
 * @brief adaptive 2d mean pooling layer. Because ResNet 18 only has one 1x1 mean pooling layer, this function ONLY supports 1x1 mean pooling!
 * @param input input tensor
 * @param output output tensor
 * @param batch_size batch size
 * @param channels input channel, not changed 
 * @param height input tensor's height
 * @param width input tensor's width
 */
 template<typename T>
 void adaptive_mean_pool(T *input, T *output, int batch_size, int channels, int height, int width);

/**
 * @brief adaptive 2d pooling layer
 * @param input input tensor
 * @param data_output output tensor
 * @param batch_size batch size 
 * @param channels input channel, not changed 
 * @param height input tensor's height
 * @param width input tensor's width
 * @param kernel_h kernel height
 * @param kernel_w kernel width
 * @param padding padding number for invalid position values
 * @param stride stride for sliding window
 */
template<typename T>
void max_pool_2d(T *input, T *output, int batch_size, int channels, int height, int width, int kernel_h, int kernel_w, int padding, int stride);

template<typename T>
void argmax(T * input, int * output, int batch_size, int num_classes);

#endif 