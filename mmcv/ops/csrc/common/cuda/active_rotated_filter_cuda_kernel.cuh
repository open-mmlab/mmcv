// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/csuhan/s2anet/blob/master/mmdet/ops/orn/src/cuda/ActiveRotatingFilter_cuda.cu
#ifndef ACTIVE_ROTATED_FILTER_CUDA_KERNEL_CUH
#define ACTIVE_ROTATED_FILTER_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename scalar_t>
__global__ void ARF_forward_cuda_kernel(
    const int nthreads, const scalar_t* weight_data, const int* indices_data,
    const int nInputPlane, const int nOutputPlane, const int num_orientations,
    const int num_rotations, const int nEntry, scalar_t* output_data) {
  CUDA_1D_KERNEL_LOOP(n, nthreads) {
    int l = n % nEntry;
    int j = (n / nEntry) % nInputPlane;
    int i = n / nEntry / nInputPlane;
    int k;
    scalar_t val = *(weight_data + n);
    for (k = 0; k < num_rotations; k++) {
      int index = (int)(*(indices_data + l * num_rotations + k)) - 1;
      scalar_t *target = output_data + i * (num_rotations * nInputPlane * nEntry) +
                         k * (nInputPlane * nEntry) + j * (nEntry) + index;
      *target = val;
    }
  }
}

template <typename scalar_t>
__global__ void ARF_backward_cuda_kernel(
    const int nthreads, const scalar_t* gradWeight_data,
    const int* indices_data, const int nInputPlane, const int nOutputPlane,
    const int num_orientations, const int num_rotations, const int nEntry,
    scalar_t* weight_data) {
  CUDA_1D_KERNEL_LOOP(n, nthreads) {
    int l = n % nEntry;
    int j = (n / nEntry) % nInputPlane;
    int i = n / nEntry / nInputPlane;
    int k;
    scalar_t *val = weight_data + n;
    *val = 0;
    for (k = 0; k < num_rotations; k++) {
      int index = (int)(*(indices_data + l * num_rotations + k)) - 1;
      scalar_t target =
          *(gradWeight_data + i * (num_rotations * nInputPlane * nEntry) +
            k * (nInputPlane * nEntry) + j * (nEntry) + index);
      *val = *val + target;
    }
  }
}
#endif  // ACTIVE_ROTATED_FILTER_CUDA_KERNEL_CUH
