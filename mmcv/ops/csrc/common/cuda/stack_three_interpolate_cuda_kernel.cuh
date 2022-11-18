// Copyright (c) OpenMMLab. All rights reserved
#ifndef STACK_THREE_INTERPOLATE_CUDA_KERNEL_CUH
#define STACK_THREE_INTERPOLATE_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

__global__ void stack_three_interpolate_forward_cuda_kernel(
int N, int channels, const float *features,
    const int *idx, const float *weight, float *out) {
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= N || c_idx >= channels) return;

    weight += pt_idx * 3;
    idx += pt_idx * 3;
    out += pt_idx * channels + c_idx;

    out[0] = weight[0] * features[idx[0] * channels + c_idx] +
        weight[1] * features[idx[1] * channels + c_idx] +
        weight[2] * features[idx[2] * channels + c_idx];
}

__global__ void stack_three_interpolate_backward_cuda_kernel(
int N, int channels, const float *grad_out,
    const int *idx, const float *weight, float *grad_features) {
    // grad_out_tensor: (N1 + N2 ..., C)
    // idx_tensor: [N1 + N2 ..., 3]
    // weight_tensor: [N1 + N2 ..., 3]
    // Return:
    // grad_features_tensor: (M1 + M2 ..., C)

    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= N || c_idx >= channels) return;

    grad_out += pt_idx * channels + c_idx;
    weight += pt_idx * 3;
    idx += pt_idx * 3;

    // printf("pt_idx=%d, c_idx=%d, idx=(%d, %d, %d), grad_out=%f\n", pt_idx, c_idx, idx[0], idx[1], idx[2], grad_out[0]);

    atomicAdd(grad_features + idx[0] * channels + c_idx, grad_out[0] * weight[0]);
    atomicAdd(grad_features + idx[1] * channels + c_idx, grad_out[0] * weight[1]);
    atomicAdd(grad_features + idx[2] * channels + c_idx, grad_out[0] * weight[2]);
}

#endif  // STACK_THREE_INTERPOLATE_CUDA_KERNEL_CUH
