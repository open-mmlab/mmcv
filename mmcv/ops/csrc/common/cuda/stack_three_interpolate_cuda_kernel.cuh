// Copyright (c) OpenMMLab. All rights reserved
#ifndef STACK_THREE_INTERPOLATE_CUDA_KERNEL_CUH
#define STACK_THREE_INTERPOLATE_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename T>
__global__ void stack_three_interpolate_forward_cuda_kernel(int N, int channels,
                                                            const T *features,
                                                            const int *idx,
                                                            const T *weight,
                                                            T *out) {
  int c_idx = blockIdx.y;
  if (c_idx >= channels) return;
  CUDA_1D_KERNEL_LOOP(pt_idx, N) {
    const T *cur_weight = weight;
    const int *cur_idx = idx;
    T *cur_out = out;
    cur_weight += pt_idx * 3;
    cur_idx += pt_idx * 3;
    cur_out += pt_idx * channels + c_idx;

    cur_out[0] = cur_weight[0] * features[cur_idx[0] * channels + c_idx] +
                 cur_weight[1] * features[cur_idx[1] * channels + c_idx] +
                 cur_weight[2] * features[cur_idx[2] * channels + c_idx];
  }
}

template <typename T>
__global__ void stack_three_interpolate_backward_cuda_kernel(
    int N, int channels, const T *grad_out, const int *idx, const T *weight,
    T *grad_features) {
  // grad_out_tensor: (N1 + N2 ..., C)
  // idx_tensor: [N1 + N2 ..., 3]
  // weight_tensor: [N1 + N2 ..., 3]
  // Return:
  // grad_features_tensor: (M1 + M2 ..., C)

  int c_idx = blockIdx.y;
  if (c_idx >= channels) return;
  CUDA_1D_KERNEL_LOOP(pt_idx, N) {
    const T *cur_grad_out = grad_out;
    const T *cur_weight = weight;
    const int *cur_idx = idx;
    cur_grad_out += pt_idx * channels + c_idx;
    cur_weight += pt_idx * 3;
    cur_idx += pt_idx * 3;

    atomicAdd(grad_features + cur_idx[0] * channels + c_idx,
              cur_grad_out[0] * cur_weight[0]);
    atomicAdd(grad_features + cur_idx[1] * channels + c_idx,
              cur_grad_out[0] * cur_weight[1]);
    atomicAdd(grad_features + cur_idx[2] * channels + c_idx,
              cur_grad_out[0] * cur_weight[2]);
  }
}

#endif  // STACK_THREE_INTERPOLATE_CUDA_KERNEL_CUH
