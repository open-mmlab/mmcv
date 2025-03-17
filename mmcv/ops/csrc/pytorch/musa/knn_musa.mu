// Copyright (c) OpenMMLab. All rights reserved
// Modified from
// https://github.com/CVMI-Lab/PAConv/tree/main/scene_seg/lib/pointops/src/knnquery_heap

#include <cmath>
#include <cstdio>

#include "knn_musa_kernel.muh"
#include "pytorch_musa_helper.hpp"

void KNNForwardMUSAKernelLauncher(int b, int n, int m, int nsample,
                                  const Tensor xyz, const Tensor new_xyz,
                                  Tensor idx, Tensor dist2) {
  // param new_xyz: (B, m, 3)
  // param xyz: (B, n, 3)
  // param idx: (B, m, nsample)

  c10::musa::MUSAGuard device_guard(new_xyz.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(GET_BLOCKS(m, THREADS_PER_BLOCK), b);
  dim3 threads(THREADS_PER_BLOCK);

  AT_DISPATCH_FLOATING_TYPES(
      new_xyz.scalar_type(), "knn_forward_musa_kernel", [&] {
        knn_forward_musa_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            b, n, m, nsample, xyz.data_ptr<scalar_t>(),
            new_xyz.data_ptr<scalar_t>(), idx.data_ptr<int>(),
            dist2.data_ptr<scalar_t>());
      });

  AT_MUSA_CHECK(musaGetLastError());
}
