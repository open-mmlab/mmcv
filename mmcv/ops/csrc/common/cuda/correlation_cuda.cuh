// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/ClementPinard/Pytorch-Correlation-extension/blob/master/Correlation_Module/correlation_cuda_kernel.cu
// Original licence: Under MIT License

#ifndef CORRELATION_CUDA
#define CORRELATION_CUDA

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include <iostream>
#include <vector>

using namespace torch;

#define TensorAcc4R PackedTensorAccessor32<scalar_t, 4, RestrictPtrTraits>
#define TensorAcc5R PackedTensorAccessor32<scalar_t, 5, RestrictPtrTraits>
#define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < H && y >= 0 && y < W)

#define THREADS_FORWARD 32
#define THREADS_BACKWARD 16

template <typename scalar_t>
__global__ void correlation_forward_cuda_kernel(
    const TensorAcc4R rInput1, const TensorAcc4R rInput2, TensorAcc5R output,
    int kH, int kW, int patchH, int patchW, int padH, int padW, int dilationH,
    int dilationW, int dilation_patchH, int dilation_patchW, int dH, int dW) {
  const int iH = rInput1.size(1);
  const int iW = rInput1.size(2);
  const int C = rInput1.size(3);

  const int n = blockIdx.x;
  const int h = blockIdx.y;
  const int w = blockIdx.z;
  const int thread = threadIdx.x;

  const int start_i = -padH + h * dH;
  const int start_j = -padW + w * dW;

  const int patchRadH = dilation_patchH * (patchH - 1) / 2;
  const int patchRadW = dilation_patchW * (patchW - 1) / 2;

  __shared__ scalar_t prod_sum[THREADS_FORWARD];

  for (int ph = 0; ph < patchH; ++ph) {
    int ph_dilated = ph * dilation_patchH - patchRadH;
    for (int pw = 0; pw < patchW; ++pw) {
      int pw_dilated = pw * dilation_patchW - patchRadW;
      prod_sum[thread] = 0;
      for (int i = 0; i < kH; ++i) {
        int i1 = start_i + i * dilationH;
        int i2 = i1 + ph_dilated;
        if
          WITHIN_BOUNDS(i1, i2, iH, iH) {
            for (int j = 0; j < kW; ++j) {
              int j1 = start_j + j * dilationW;
              int j2 = j1 + pw_dilated;
              if
                WITHIN_BOUNDS(j1, j2, iW, iW) {
                  for (int c = thread; c < C; c += THREADS_FORWARD) {
                    scalar_t v1 = rInput1[n][i1][j1][c];
                    scalar_t v2 = rInput2[n][i2][j2][c];
                    prod_sum[thread] += v1 * v2;
                  }
                }
            }
          }
      }
      // accumulate
      __syncthreads();
      if (thread == 0) {
        scalar_t reduce_sum = 0;
        for (int index = 0; index < THREADS_FORWARD; ++index) {
          reduce_sum += prod_sum[index];
        }
        output[n][ph][pw][h][w] = reduce_sum;
      }
    }
  }
}

template <typename scalar_t>
__global__ void correlation_backward_cuda_kernel_input1(
    const TensorAcc5R grad_output, const TensorAcc4R input2,
    TensorAcc4R grad_input1, const int kH, const int kW, const int patchH,
    const int patchW, const int padH, const int padW, const int dilationH,
    const int dilationW, const int dilation_patchH, const int dilation_patchW,
    const int dH, const int dW, const int batch) {
  const int iH = input2.size(2);
  const int iW = input2.size(3);

  const int H = grad_output.size(3);
  const int W = grad_output.size(4);

  const int patchRadH = (patchH - 1) / 2;
  const int patchRadW = (patchW - 1) / 2;

  const int n = batch;
  const int c = blockIdx.x;
  const int h = blockIdx.y;
  const int w = blockIdx.z;
  const int ph_off = threadIdx.x;
  const int pw_off = threadIdx.y;

  const int h_2 = h + padH;
  const int w_2 = w + padW;
  const int min_h = h_2 - kH * dilationH;
  const int min_w = w_2 - kW * dilationW;

  __shared__ scalar_t prod_sum[THREADS_BACKWARD][THREADS_BACKWARD];
  prod_sum[ph_off][pw_off] = 0;

  for (int ph = ph_off; ph < patchH; ph += THREADS_BACKWARD) {
    int i1 = h + dilation_patchH * (ph - patchRadH);
    for (int pw = pw_off; pw < patchW; pw += THREADS_BACKWARD) {
      int j1 = w + dilation_patchW * (pw - patchRadW);
      if (WITHIN_BOUNDS(i1, j1, iH, iW)) {
        scalar_t val = input2[n][c][i1][j1];
        for (int h_3 = h_2; h_3 > min_h; h_3 -= dilationH) {
          int i2 = (h_3) / dH;
          if (i2 * dH != h_3) continue;
          for (int w_3 = w_2; w_3 > min_w; w_3 -= dilationW) {
            int j2 = (w_3) / dW;
            if (j2 * dW != w_3) continue;
            if
              WITHIN_BOUNDS(i2, j2, H, W) {
                prod_sum[ph_off][pw_off] +=
                    grad_output[n][ph][pw][i2][j2] * val;
              }
          }
        }
      }
    }
  }

  __syncthreads();

  if (ph_off == 0 && pw_off == 0) {
    scalar_t reduce_sum = 0;
    for (int ph = 0; ph < THREADS_BACKWARD; ++ph) {
      for (int pw = 0; pw < THREADS_BACKWARD; ++pw) {
        reduce_sum += prod_sum[ph][pw];
      }
    }
    grad_input1[n][c][h][w] = reduce_sum;
  }
}

template <typename scalar_t>
__global__ void correlation_backward_cuda_kernel_input2(
    const TensorAcc5R grad_output, const TensorAcc4R input1,
    TensorAcc4R grad_input2, int kH, int kW, int patchH, int patchW, int padH,
    int padW, int dilationH, int dilationW, int dilation_patchH,
    int dilation_patchW, int dH, int dW, int batch) {
  const int iH = input1.size(2);
  const int iW = input1.size(3);

  const int patchRadH = (patchH - 1) / 2;
  const int patchRadW = (patchW - 1) / 2;

  const int H = grad_output.size(3);
  const int W = grad_output.size(4);

  const int dilatedKH = kH * dilationH;
  const int dilatedKW = kW * dilationW;

  const int n = batch;
  const int c = blockIdx.x;
  const int h = blockIdx.y;
  const int w = blockIdx.z;
  const int ph_off = threadIdx.x;
  const int pw_off = threadIdx.y;

  __shared__ scalar_t prod_sum[THREADS_BACKWARD][THREADS_BACKWARD];
  prod_sum[ph_off][pw_off] = 0;

  for (int ph = ph_off; ph < patchH; ph += THREADS_BACKWARD) {
    int i1 = h - dilation_patchH * (ph - patchRadH);
    for (int pw = pw_off; pw < patchW; pw += THREADS_BACKWARD) {
      int j1 = w - dilation_patchW * (pw - patchRadW);
      if
        WITHIN_BOUNDS(i1, j1, iH, iW) {
          scalar_t val = input1[n][c][i1][j1];

          const int h_2 = i1 + padH;
          const int w_2 = j1 + padW;
          const int min_h = h_2 - dilatedKH;
          const int min_w = w_2 - dilatedKW;

          for (int h_3 = h_2; h_3 > min_h; h_3 -= dilationH) {
            int i2 = (h_3) / dH;
            if (i2 * dH != h_3) continue;
            for (int w_3 = w_2; w_3 > min_w; w_3 -= dilationW) {
              int j2 = (w_3) / dW;
              if (j2 * dW != w_3) continue;
              if
                WITHIN_BOUNDS(i2, j2, H, W) {
                  prod_sum[ph_off][pw_off] +=
                      grad_output[n][ph][pw][i2][j2] * val;
                }
            }
          }
        }
    }
  }

  __syncthreads();

  if (ph_off == 0 && pw_off == 0) {
    scalar_t reduce_sum = 0;
    for (int ph = 0; ph < THREADS_BACKWARD; ++ph) {
      for (int pw = 0; pw < THREADS_BACKWARD; ++pw) {
        reduce_sum += prod_sum[ph][pw];
      }
    }
    grad_input2[n][c][h][w] = reduce_sum;
  }
}
#endif
