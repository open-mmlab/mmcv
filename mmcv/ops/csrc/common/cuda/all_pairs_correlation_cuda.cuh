// Copyright (c) OpenMMLab. All rights reserved.
#ifndef ALLPAIRS_CORRELATION_CUDA
#define ALLPAIRS_CORRELATION_CUDA

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>
#include <vector>
#include <iostream>

using namespace torch;

#define TensorAcc4R PackedTensorAccessor32<scalar_t,4,RestrictPtrTraits>
#define TensorAcc5R PackedTensorAccessor32<scalar_t,5,RestrictPtrTraits>

#define THREADS_FORWARD 32
#define THREADS_BACKWARD 16


template <typename scalar_t>
__global__ void all_pairs_correlation_forward_cuda_kernel(const TensorAcc4R rInput1,
                                                const TensorAcc4R rInput2,
                                                TensorAcc5R output)

{
  const int iH = rInput2.size(1);
  const int iW = rInput2.size(2);
  const int C = rInput2.size(3);

  const int n = blockIdx.x;
  const int h = blockIdx.y;
  const int w = blockIdx.z;
  const int thread = threadIdx.x;

  __shared__ scalar_t prod_sum[THREADS_FORWARD];


  for (int i = 0 ; i < iH; ++i){
    for (int j= 0; j< iW; ++j){
      prod_sum[thread] = 0;
      for (int c=thread; c<C; c += THREADS_FORWARD){
        scalar_t v1 = rInput1[n][h][w][c];
        scalar_t v2 = rInput2[n][i][j][c];
        prod_sum[thread] += v1 * v2;
      }
      // accumulate
    __syncthreads();
    if (thread == 0) {
      scalar_t reduce_sum = 0;
      for (int index = 0; index < THREADS_FORWARD; ++index) {
        reduce_sum += prod_sum[index];
      }
      output[n][h][w][i][j] = reduce_sum;
      }
    }
  }
}
template <typename scalar_t>
__global__ void  all_pairs_correlation_backward_cuda_kernel_input1(const TensorAcc5R grad_output,
    const TensorAcc4R input2, TensorAcc4R grad_input1, int batch)
{
    const int iH = input2.size(2);
    const int iW = input2.size(3);

    // const int H = grad_output.size(3);
    // const int W = grad_output.size(4);

    const int n = batch;
    const int c = blockIdx.x;
    const int h = blockIdx.y;
    const int w = blockIdx.z;
    const int h2_off = threadIdx.x;
    const int w2_off = threadIdx.y;

    __shared__ scalar_t prod_sum[THREADS_BACKWARD][THREADS_BACKWARD];
    prod_sum[h2_off][w2_off] = 0;

    for (int h2 = h2_off; h2 < iH; h2 += THREADS_BACKWARD) {
      for (int w2 = w2_off; w2 < iW; w2 += THREADS_BACKWARD){
        scalar_t val = input2[n][c][h2][w2];
        prod_sum[h2_off][w2_off] += grad_output[n][h][w][h2][w2] * val;
      }
    }
    __syncthreads();

    if (h2_off == 0 && w2_off == 0){
      scalar_t reduce_sum =0;
      for (int h_off = 0; h_off < THREADS_BACKWARD; ++h_off){
        for (int w_off = 0; w_off < THREADS_BACKWARD; ++w_off){
          reduce_sum += prod_sum[h_off][w_off];

        }
      }
      grad_input1[n][c][h][w] = reduce_sum;
    }
}

template <typename scalar_t>
__global__ void  all_pairs_correlation_backward_cuda_kernel_input2(const TensorAcc5R grad_output,
    const TensorAcc4R input1, TensorAcc4R grad_input2,int batch)
{
    const int iH = input1.size(2);
    const int iW = input1.size(3);

    // const int H = grad_output.size(3);
    // const int W = grad_output.size(4);

    const int n = batch;
    const int c = blockIdx.x;
    const int h = blockIdx.y;
    const int w = blockIdx.z;
    const int h1_off = threadIdx.x;
    const int w1_off = threadIdx.y;

    __shared__ scalar_t prod_sum[THREADS_BACKWARD][THREADS_BACKWARD];
    prod_sum[h1_off][w1_off] = 0;

    for (int h1 = h1_off; h1 < iH; h1 += THREADS_BACKWARD) {
      for (int w1 = w1_off; w1 < iW; w1 += THREADS_BACKWARD){
        scalar_t val = input1[n][c][h1][w1];
        prod_sum[h1_off][w1_off] += grad_output[n][h1][w1][h][w] * val;
      }
    }
    __syncthreads();
    if (h1_off == 0 && w1_off == 0){
      scalar_t reduce_sum =0;
      for (int h_off = 0; h_off < THREADS_BACKWARD; ++h_off){
        for (int w_off = 0; w_off < THREADS_BACKWARD; ++w_off){
          reduce_sum += prod_sum[h_off][w_off];
        }
      }
      grad_input2[n][c][h][w] = reduce_sum;
    }
}
#endif
