#ifndef _GROUP14_MAT_VEC_ADD_HPP_
#define _GROUP14_MAT_VEC_ADD_HPP_

/**
 * @brief matrix vector addition
 * @param mat       input matrix    shape (height, width), allocated on GPU
 * @param vec       vector to add   shape (height, 1), allocated on GPU
 * @param res_mat   result matrix   shape (height, width), allocated on GPU
 * @param height    matrix height
 * @param width     matrix width
 */
template<typename T>
void mat_vec_add(T *mat, T *vec, T *res_mat, int height, int width);

/**
 * @brief matrix vector addition for batch
 * @param mat           input matrix    shape (batch, height, width), allocated on GPU
 * @param vec           vector to add   shape (height, 1), allocated on GPU
 * @param res_mat       result matrix   shape (batch, height, width), allocated on GPU
 * @param height        matrix height
 * @param width         matrix width
 * @param batch_size    batch size
 */
template<typename T>
void batch_mat_vec_add(T *mat, T *vec, T *res_mat, int height, int width, int batch_size);

#endif