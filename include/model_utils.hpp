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
void relu(float *input, float * output, int batch_size, int height, int width, int input_dim);


/**
 * @brief adaptive 2d mean pooling layer. Because ResNet 18 only has one 1x1 mean pooling layer, this function ONLY supports 1x1 mean pooling!
 * @param input input tensor
 * @param output output tensor
 * @param batch_size batch size
 * @param channels input channel, not changed 
 * @param height input tensor's height
 * @param width input tensor's width
 */
void adaptive_mean_pool(float * input, float * output, int batch_size, int channels, int height, int width);

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
void max_pool_2d(float *input, float *output, int batch_size, int channels, int height, int width, int kernel_h, int kernel_w, int padding, int stride);


/**
 * @brief 2d batch normalization
 * @param input input tensor
 * @param output output tensor
 * @param batch_size batch size 
 * @param channels input channel, not changed 
 * @param height input tensor's height
 * @param width input tensor's width
 * @param running_mean the pre-computed mean value per batch
 * @param running_var the pre-computed variance value per batch
 * @param weight the learnable weight term
 * @param bias the learnable bias term
 */
void batch_norm_2d(float * input, float * output, int batch_size, int channels, int height, int width, float* running_mean, float * running_var, float * weight, float * bias);


#endif 