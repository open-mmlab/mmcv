// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/chrdiller/pyTorchChamferDistance/blob/master/chamfer_distance/chamfer_distance.cu
#ifndef CHAMFER_DISTANCE_CUDA_KERNEL_CUH
#define CHAMFER_DISTANCE_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

template <typename scalar_t>
__global__ void chamfer_distance_forward_cuda_kernel(int b, int n,
                                                     const scalar_t* xyz, int m,
                                                     const scalar_t* xyz2,
                                                     scalar_t* result,
                                                     int* result_i) {
  __shared__ scalar_t buf[THREADS_PER_BLOCK * 2];
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int k2 = 0; k2 < m; k2 += THREADS_PER_BLOCK) {
      int end_k = min(m, k2 + THREADS_PER_BLOCK) - k2;
      for (int j = threadIdx.x; j < end_k * 2; j += blockDim.x) {
        buf[j] = xyz2[(i * m + k2) * 2 + j];
      }
      __syncthreads();
      for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n;
           j += blockDim.x * gridDim.y) {
        scalar_t x1 = xyz[(i * n + j) * 2 + 0];
        scalar_t y1 = xyz[(i * n + j) * 2 + 1];
        int best_i = 0;
        scalar_t best = 0;
        int end_ka = end_k - (end_k & 2);
        if (end_ka == THREADS_PER_BLOCK) {
          for (int k = 0; k < THREADS_PER_BLOCK; k += 4) {
            {
              scalar_t x2 = buf[k * 2 + 0] - x1;
              scalar_t y2 = buf[k * 2 + 1] - y1;
              scalar_t d = x2 * x2 + y2 * y2;
              if (k == 0 || d < best) {
                best = d;
                best_i = k + k2;
              }
            }
            {
              scalar_t x2 = buf[k * 2 + 2] - x1;
              scalar_t y2 = buf[k * 2 + 3] - y1;
              scalar_t d = x2 * x2 + y2 * y2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 1;
              }
            }
            {
              scalar_t x2 = buf[k * 2 + 4] - x1;
              scalar_t y2 = buf[k * 2 + 5] - y1;
              scalar_t d = x2 * x2 + y2 * y2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 2;
              }
            }
            {
              scalar_t x2 = buf[k * 2 + 6] - x1;
              scalar_t y2 = buf[k * 2 + 7] - y1;
              scalar_t d = x2 * x2 + y2 * y2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 3;
              }
            }
          }
        } else {
          for (int k = 0; k < end_ka; k += 4) {
            {
              scalar_t x2 = buf[k * 2 + 0] - x1;
              scalar_t y2 = buf[k * 2 + 1] - y1;
              scalar_t d = x2 * x2 + y2 * y2;
              if (k == 0 || d < best) {
                best = d;
                best_i = k + k2;
              }
            }
            {
              scalar_t x2 = buf[k * 2 + 2] - x1;
              scalar_t y2 = buf[k * 2 + 3] - y1;
              scalar_t d = x2 * x2 + y2 * y2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 1;
              }
            }
            {
              scalar_t x2 = buf[k * 2 + 4] - x1;
              scalar_t y2 = buf[k * 2 + 5] - y1;
              scalar_t d = x2 * x2 + y2 * y2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 2;
              }
            }
            {
              scalar_t x2 = buf[k * 2 + 6] - x1;
              scalar_t y2 = buf[k * 2 + 7] - y1;
              scalar_t d = x2 * x2 + y2 * y2;
              if (d < best) {
                best = d;
                best_i = k + k2 + 3;
              }
            }
          }
        }
        for (int k = end_ka; k < end_k; k++) {
          scalar_t x2 = buf[k * 2 + 0] - x1;
          scalar_t y2 = buf[k * 2 + 1] - y1;
          scalar_t d = x2 * x2 + y2 * y2;
          if (k == 0 || d < best) {
            best = d;
            best_i = k + k2;
          }
        }
        if (k2 == 0 || result[(i * n + j)] > best) {
          result[(i * n + j)] = best;
          result_i[(i * n + j)] = best_i;
        }
      }
      __syncthreads();
    }
  }
}

template <typename scalar_t>
__global__ void chamfer_distance_backward_cuda_kernel(
    int b, int n, const scalar_t* xyz1, int m, const scalar_t* xyz2,
    const scalar_t* grad_dist1, const int* idx1, scalar_t* grad_xyz1,
    scalar_t* grad_xyz2) {
  for (int i = blockIdx.x; i < b; i += gridDim.x) {
    for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n;
         j += blockDim.x * gridDim.y) {
      scalar_t x1 = xyz1[(i * n + j) * 2 + 0];
      scalar_t y1 = xyz1[(i * n + j) * 2 + 1];
      int j2 = idx1[i * n + j];
      scalar_t x2 = xyz2[(i * m + j2) * 2 + 0];
      scalar_t y2 = xyz2[(i * m + j2) * 2 + 1];
      scalar_t g = grad_dist1[i * n + j] * 2;
      atomicAdd(&(grad_xyz1[(i * n + j) * 2 + 0]), g * (x1 - x2));
      atomicAdd(&(grad_xyz1[(i * n + j) * 2 + 1]), g * (y1 - y2));
      atomicAdd(&(grad_xyz2[(i * m + j2) * 2 + 0]), -(g * (x1 - x2)));
      atomicAdd(&(grad_xyz2[(i * m + j2) * 2 + 1]), -(g * (y1 - y2)));
    }
  }
}
#endif  // CHAMFER_DISTANCE_CUDA_KERNEL_CUH
