// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/group_points_gpu.cu
#include <stdio.h>
#include <stdlib.h>

#include "pytorch_musa_helper.hpp"
#include "stack_group_points_musa_kernel.muh"

void StackGroupPointsForwardMUSAKernelLauncher(
    int b, int c, int m, int nsample, const Tensor features_tensor,
    const Tensor features_batch_cnt_tensor, const Tensor idx_tensor,
    const Tensor idx_batch_cnt_tensor, Tensor out_tensor) {
  // points: (B, C, N)
  // idx: (B, npoints, nsample)
  // output:
  //      out: (B, C, npoints, nsample)
  c10::musa::MUSAGuard device_guard(features_tensor.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();

  dim3 blocks(DIVUP(m * c * nsample, THREADS_PER_BLOCK));
  dim3 threads(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features_tensor.scalar_type(), "stack_group_points_forward_musa_kernel",
      [&] {
        stack_group_points_forward_musa_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                b, c, m, nsample, features_tensor.data_ptr<scalar_t>(),
                features_batch_cnt_tensor.data_ptr<int>(),
                idx_tensor.data_ptr<int>(),
                idx_batch_cnt_tensor.data_ptr<int>(),
                out_tensor.data_ptr<scalar_t>());
      });

  AT_MUSA_CHECK(musaGetLastError());
}

void StackGroupPointsBackwardMUSAKernelLauncher(
    int b, int c, int m, int n, int nsample, const Tensor grad_out_tensor,
    const Tensor idx_tensor, const Tensor idx_batch_cnt_tensor,
    const Tensor features_batch_cnt_tensor, Tensor grad_features_tensor) {
  c10::musa::MUSAGuard device_guard(grad_features_tensor.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();

  dim3 blocks(DIVUP(m * c * nsample, THREADS_PER_BLOCK));
  dim3 threads(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_features_tensor.scalar_type(),
      "stack_group_points_backward_musa_kernel", [&] {
        stack_group_points_backward_musa_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                b, c, m, n, nsample, grad_out_tensor.data_ptr<scalar_t>(),
                idx_tensor.data_ptr<int>(),
                idx_batch_cnt_tensor.data_ptr<int>(),
                features_batch_cnt_tensor.data_ptr<int>(),
                grad_features_tensor.data_ptr<scalar_t>());
      });

  AT_MUSA_CHECK(musaGetLastError());
}
