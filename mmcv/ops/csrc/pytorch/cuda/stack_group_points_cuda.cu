// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/group_points_gpu.cu
#include <stdio.h>
#include <stdlib.h>

#include "pytorch_cuda_helper.hpp"
#include "stack_group_points_cuda_kernel.cuh"

void StackGroupPointsForwardCUDAKernelLauncher(
    int b, int c, int m, int nsample, const Tensor features_tensor,
    const Tensor features_batch_cnt_tensor, const Tensor idx_tensor,
    const Tensor idx_batch_cnt_tensor, Tensor out_tensor) {
  // points: (B, C, N)
  // idx: (B, npoints, nsample)
  // output:
  //      out: (B, C, npoints, nsample)

  at::cuda::CUDAGuard device_guard(features_tensor.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(DIVUP(m * c * nsample, THREADS_PER_BLOCK));
  dim3 threads(THREADS_PER_BLOCK);

  //   const float *features_ptr = features_tensor.data_ptr<scalar_t>();
  //   const int *idx_ptr = idx_tensor.data_ptr<int>();
  //   const int *features_batch_cnt_ptr =
  //   features_batch_cnt_tensor.data_ptr<int>(); const int *idx_batch_cnt_ptr =
  //   idx_batch_cnt_tensor.data_ptr<int>(); float *out_ptr =
  //   out_tensor.data_ptr<scalar_t>();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features_tensor.scalar_type(), "stack_group_points_forward_cuda_kernel",
      [&] {
        stack_group_points_forward_cuda_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                b, c, m, nsample, features_tensor.data_ptr<scalar_t>(),
                features_batch_cnt_tensor.data_ptr<int>(),
                idx_tensor.data_ptr<int>(),
                idx_batch_cnt_tensor.data_ptr<int>(),
                out_tensor.data_ptr<scalar_t>());
      });

  AT_CUDA_CHECK(cudaGetLastError());
}

void StackGroupPointsBackwardCUDAKernelLauncher(
    int b, int c, int m, int n, int nsample, const Tensor grad_out_tensor,
    const Tensor idx_tensor, const Tensor idx_batch_cnt_tensor,
    const Tensor features_batch_cnt_tensor, Tensor grad_features_tensor) {
  // grad_out: (B, C, npoints, nsample)
  // idx: (B, npoints, nsample)
  // output:
  //      grad_points: (B, C, N)

  at::cuda::CUDAGuard device_guard(grad_features_tensor.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(DIVUP(m * c * nsample, THREADS_PER_BLOCK));
  dim3 threads(THREADS_PER_BLOCK);

  //   const float *grad_out_ptr = grad_out_tensor.data_ptr<float>();
  //   const int *idx_ptr = idx_tensor.data_ptr<int>();
  //   const int *idx_batch_cnt_ptr = idx_batch_cnt_tensor.data_ptr<int>();
  //   const int *features_batch_cnt_ptr =
  //   features_batch_cnt_tensor.data_ptr<int>(); float *grad_features_ptr =
  //   grad_features_tensor.data_ptr<float>();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_features_tensor.scalar_type(),
      "stack_group_points_backward_cuda_kernel", [&] {
        stack_group_points_backward_cuda_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                b, c, m, n, nsample, grad_out_tensor.data_ptr<scalar_t>(),
                idx_tensor.data_ptr<int>(),
                idx_batch_cnt_tensor.data_ptr<int>(),
                features_batch_cnt_tensor.data_ptr<int>(),
                grad_features_tensor.data_ptr<scalar_t>());
      });

  AT_CUDA_CHECK(cudaGetLastError());
}
