#ifndef _GROUP_14_BATCH_ADD_HPP_
#define _GROUP_14_BATCH_ADD_HPP_

/**
 * @brief batch addition
 * @param mat_a       input matrix    shape (batch, channel, height, width), allocated on GPU
 * @param mat_b       input matrix    shape (batch, channel, height, width), allocated on GPU
 * @param mat_out     output matrix   shape (batch, channel, height, width), allocated on GPU
 * @param batch       batch size
 * @param channel     channel size
 * @param height      height size
 * @param width       width size
 */
template<typename T>
void batch_add(T *mat_a, T *mat_b, T *mat_out, int batch, int channel, int height, int width);

#endif