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
    const long nthreads, const scalar_t* weight_data, const int* indices_data,
    const uint16 nInputPlane, const uint16 nOutputPlane,
    const uint8 nOrientation, const uint8 nRotation, const uint16 nEntry,
    scalar_t* output_data) {
  CUDA_1D_KERNEL_LOOP(n, nthreads) {
    int l = n % nEntry;
    int j = (n / nEntry) % nInputPlane;
    int i = n / nEntry / nInputPlane;
    int k;
    scalar_t val = *(weight_data + n);
    for (k = 0; k < nRotation; k++) {
      uint16 index = (uint16)(*(indices_data + l * nRotation + k)) - 1;
      scalar_t* target = output_data + i * (nRotation * nInputPlane * nEntry) +
                         k * (nInputPlane * nEntry) + j * (nEntry) + index;
      *target = val;
    }
  }
}

template <typename scalar_t>
__global__ void ARF_backward_cuda_kernel(
    const int nthreads, const scalar_t* gradWeight_data,
    const int* indices_data, const int nInputPlane, const int nOutputPlane,
    const int nOrientation, const int nRotation, const int nEntry,
    scalar_t* weight_data) {
  CUDA_1D_KERNEL_LOOP(n, nthreads) {
    int l = n % nEntry;
    int j = (n / nEntry) % nInputPlane;
    int i = n / nEntry / nInputPlane;
    int k;
    scalar_t* val = weight_data + n;
    *val = 0;
    for (k = 0; k < nRotation; k++) {
      int index = (uint16)(*(indices_data + l * nRotation + k)) - 1;
      scalar_t target =
          *(gradWeight_data + i * (nRotation * nInputPlane * nEntry) +
            k * (nInputPlane * nEntry) + j * (nEntry) + index);
      *val = *val + target;
    }
  }
}
#endif  // ACTIVE_ROTATED_FILTER_CUDA_KERNEL_CUH
