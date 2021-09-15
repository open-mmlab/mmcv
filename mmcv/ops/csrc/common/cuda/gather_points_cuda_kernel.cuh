// Copyright (c) OpenMMLab. All rights reserved
#ifndef GATHER_POINTS_CUDA_KERNEL_CUH
#define GATHER_POINTS_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

__global__ void gather_points_kernel(int b, int c, int n, int m,
                                     const float *__restrict__ points,
                                     const int *__restrict__ idx,
                                     float *__restrict__ out) {
  // points: (B, C, N)
  // idx: (B, M)
  // output:
  //      out: (B, C, M)

  int bs_idx = blockIdx.z;
  int c_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= b || c_idx >= c || pt_idx >= m) return;

  out += bs_idx * c * m + c_idx * m + pt_idx;
  idx += bs_idx * m + pt_idx;
  points += bs_idx * c * n + c_idx * n;
  out[0] = points[idx[0]];
}

__global__ void gather_points_grad_kernel(int b, int c, int n, int m,
                                          const float *__restrict__ grad_out,
                                          const int *__restrict__ idx,
                                          float *__restrict__ grad_points) {
  // grad_out: (B, C, M)
  // idx: (B, M)
  // output:
  //      grad_points: (B, C, N)

  int bs_idx = blockIdx.z;
  int c_idx = blockIdx.y;
  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (bs_idx >= b || c_idx >= c || pt_idx >= m) return;

  grad_out += bs_idx * c * m + c_idx * m + pt_idx;
  idx += bs_idx * m + pt_idx;
  grad_points += bs_idx * c * n + c_idx * n;

  atomicAdd(grad_points + idx[0], grad_out[0]);
}

#endif  // GATHER_POINTS_CUDA_KERNEL_CUH
