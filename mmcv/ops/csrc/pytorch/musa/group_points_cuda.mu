// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/group_points_gpu.cu
#include <stdio.h>
#include <stdlib.h>

#include "group_points_cuda_kernel.muh"
#include "pytorch_cuda_helper.hpp"

void GroupPointsForwardCUDAKernelLauncher(int b, int c, int n, int npoints,
                                          int nsample, const Tensor points,
                                          const Tensor idx, Tensor out) {
  // points: (B, C, N)
  // idx: (B, npoints, nsample)
  // output:
  //      out: (B, C, npoints, nsample)

  at::musa::MUSAGuard device_guard(points.device());
  musaStream_t stream = at::musa::getCurrentMUSAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(GET_BLOCKS(npoints * nsample, THREADS_PER_BLOCK), c, b);
  dim3 threads(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES(
      points.scalar_type(), "group_points_forward_cuda_kernel", [&] {
        group_points_forward_cuda_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                b, c, n, npoints, nsample, points.data_ptr<scalar_t>(),
                idx.data_ptr<int>(), out.data_ptr<scalar_t>());
      });

  AT_MUSA_CHECK(musaGetLastError());
}

void GroupPointsBackwardCUDAKernelLauncher(int b, int c, int n, int npoints,
                                           int nsample, const Tensor grad_out,
                                           const Tensor idx,
                                           Tensor grad_points) {
  // grad_out: (B, C, npoints, nsample)
  // idx: (B, npoints, nsample)
  // output:
  //      grad_points: (B, C, N)

  at::musa::MUSAGuard device_guard(grad_out.device());
  musaStream_t stream = at::musa::getCurrentMUSAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(GET_BLOCKS(npoints * nsample, THREADS_PER_BLOCK), c, b);
  dim3 threads(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES(
      grad_out.scalar_type(), "group_points_backward_cuda_kernel", [&] {
        group_points_backward_cuda_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                b, c, n, npoints, nsample, grad_out.data_ptr<scalar_t>(),
                idx.data_ptr<int>(), grad_points.data_ptr<scalar_t>());
      });

  AT_MUSA_CHECK(musaGetLastError());
}
