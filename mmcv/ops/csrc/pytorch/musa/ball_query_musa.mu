// Copyright (c) OpenMMLab. All rights reserved
// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/ball_query_gpu.cu

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ball_query_musa_kernel.muh"
#include "pytorch_musa_helper.hpp"

void BallQueryForwardMUSAKernelLauncher(int b, int n, int m, float min_radius,
                                        float max_radius, int nsample,
                                        const Tensor new_xyz, const Tensor xyz,
                                        Tensor idx) {
  // new_xyz: (B, M, 3)
  // xyz: (B, N, 3)
  // output:
  //      idx: (B, M, nsample)

  at::musa::MUSAGuard device_guard(new_xyz.device());
  musaStream_t stream = at::musa::getCurrentMUSAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(GET_BLOCKS(m, THREADS_PER_BLOCK), b);
  dim3 threads(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES(
      new_xyz.scalar_type(), "ball_query_forward_musa_kernel", [&] {
        ball_query_forward_musa_kernel<scalar_t>
            <<<blocks, threads, 0, stream>>>(
                b, n, m, min_radius, max_radius, nsample,
                new_xyz.data_ptr<scalar_t>(), xyz.data_ptr<scalar_t>(),
                idx.data_ptr<int>());
      });

  AT_MUSA_CHECK(musaGetLastError());
}
