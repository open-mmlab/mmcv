// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/group_points_gpu.cu
#ifndef STACK_GROUP_POINTS_CUDA_KERNEL_CUH
#define STACK_GROUP_POINTS_CUDA_KERNEL_CUH
#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename T>
__global__ void stack_group_points_forward_cuda_kernel(int b, int c, int m, int nsample,
    const T *features, const int *features_batch_cnt, const int *idx, const int *idx_batch_cnt, T *out) {
    // :param features: (N1 + N2 ..., C) tensor of features to group
    // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indices of features to group with
    // :param idx: (M1 + M2 ..., nsample) tensor containing the indices of features to group with
    // :param idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indices of features to group with
    // :return:
    //     output: (M1 + M2, C, nsample) tensor
    const T *cur_features = features;
    const int *cur_idx = idx;
    CUDA_1D_KERNEL_LOOP(index, m * c * nsample){
        int sample_idx = index % nsample;
        int c_idx = (index / nsample) % c;
        int pt_idx = (index / nsample / c);

        if (c_idx >= c || sample_idx >= nsample) break;
        int bs_idx = 0, pt_cnt = idx_batch_cnt[0];
        for (int pt_cnt = 0; bs_idx < b; bs_idx++){
            pt_cnt += idx_batch_cnt[bs_idx];
            if (pt_idx < pt_cnt) break;
        }

        int features_batch_start_idx = 0;
        for (int k = 0; k < bs_idx; k++) features_batch_start_idx += features_batch_cnt[k];
        cur_features += features_batch_start_idx * c;

        cur_idx += pt_idx * nsample + sample_idx;
        int in_idx = cur_idx[0] * c + c_idx;
        int out_idx = pt_idx * c * nsample + c_idx * nsample + sample_idx;

        out[out_idx] = cur_features[in_idx];
    }
}

template <typename T>
__global__ void stack_group_points_backward_cuda_kernel(int b, int c, int m, int n, int nsample,
    const T *grad_out, const int *idx, const int *idx_batch_cnt, const int *features_batch_cnt, T *grad_features) {
    // :param grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward
    // :param idx: (M1 + M2 ..., nsample) tensor containing the indices of features to group with
    // :param idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indices of features to group with
    // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indices of features to group with
    // :return:
    //     grad_features: (N1 + N2 ..., C) gradient of the features
    const T *cur_grad_out = grad_out;
    const int *cur_idx = idx;
    CUDA_1D_KERNEL_LOOP(index, m * c * nsample){
        int sample_idx = index % nsample;
        int c_idx = (index / nsample) % c;
        int pt_idx = (index / nsample / c);

        if (c_idx >= c || sample_idx >= nsample) break;

        int bs_idx = 0, pt_cnt = idx_batch_cnt[0];
        for (int k = 1; k < b; k++){
            if (pt_idx < pt_cnt) break;
            pt_cnt += idx_batch_cnt[k];
            bs_idx = k;
        }

        int features_batch_start_idx = 0;
        for (int k = 0; k < bs_idx; k++) features_batch_start_idx += features_batch_cnt[k];

        cur_grad_out += pt_idx * c * nsample + c_idx * nsample + sample_idx;
        cur_idx += pt_idx * nsample + sample_idx;
        grad_features += (features_batch_start_idx + cur_idx[0]) * c + c_idx;

        atomicAdd(grad_features, cur_grad_out[0]);
    }
}

#endif // GROUP_POINTS_CUDA_KERNEL_CUH
