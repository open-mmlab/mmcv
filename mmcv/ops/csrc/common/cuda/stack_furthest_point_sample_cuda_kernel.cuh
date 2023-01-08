// Copyright (c) OpenMMLab. All rights reserved.
#ifndef STACK_FURTHEST_POINT_SAMPLE_CUDA_KERNEL_CUH
#define STACK_FURTHEST_POINT_SAMPLE_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif
#define TOTAL_THREADS 1024

__device__ void update(float *__restrict__ dists, int *__restrict__ dists_i,
                       int idx1, int idx2) {
  const float v1 = dists[idx1], v2 = dists[idx2];
  const int i1 = dists_i[idx1], i2 = dists_i[idx2];
  dists[idx1] = max(v1, v2);
  dists_i[idx1] = v2 > v1 ? i2 : i1;
}

template <unsigned int block_size>
__global__ void stack_farthest_point_sample_cuda_kernel(
    int batch_size, int N, const float *dataset, float *temp,
    int *xyz_batch_cnt, int *idxs, int *num_sampled_points) {
  // """
  // Args:
  //     ctx:
  //     dataset: (N1 + N2 + ..., 3) where N > npoint
  //     temp: (N1 + N2 + ...) where N > npoint
  //     xyz_batch_cnt: [N1, N2, ...]
  //     num_sampled_points: [M1, M2, ...] int, number of features in the
  //     sampled set

  // Returns:
  //     idxs: (npoint.sum()) tensor containing the set,
  //     npoint: (M1, M2, ...)
  // """

  __shared__ float dists[block_size];
  __shared__ int dists_i[block_size];

  int bs_idx = blockIdx.x;

  int xyz_batch_start_idx = 0, idxs_start_idx = 0;
  for (int k = 0; k < bs_idx; k++) {
    xyz_batch_start_idx += xyz_batch_cnt[k];
    idxs_start_idx += num_sampled_points[k];
  }

  dataset += xyz_batch_start_idx * 3;
  temp += xyz_batch_start_idx;
  idxs += idxs_start_idx;

  int n = xyz_batch_cnt[bs_idx];
  int m = num_sampled_points[bs_idx];

  int tid = threadIdx.x;
  const int stride = block_size;

  int old = 0;
  if (threadIdx.x == 0) idxs[0] = xyz_batch_start_idx;

  __syncthreads();
  for (int j = 1; j < m; j++) {
    int besti = 0;
    float best = -1;
    float x1 = dataset[old * 3 + 0];
    float y1 = dataset[old * 3 + 1];
    float z1 = dataset[old * 3 + 2];
    for (int k = tid; k < n; k += stride) {
      float x2, y2, z2;
      x2 = dataset[k * 3 + 0];
      y2 = dataset[k * 3 + 1];
      z2 = dataset[k * 3 + 2];
      // float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
      // if (mag <= 1e-3)
      // continue;

      float d =
          (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
      float d2 = min(d, temp[k]);
      temp[k] = d2;
      besti = d2 > best ? k : besti;
      best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 1024) {
      if (tid < 512) {
        update(dists, dists_i, tid, tid + 512);
      }
      __syncthreads();
    }

    if (block_size >= 512) {
      if (tid < 256) {
        update(dists, dists_i, tid, tid + 256);
      }
      __syncthreads();
    }
    if (block_size >= 256) {
      if (tid < 128) {
        update(dists, dists_i, tid, tid + 128);
      }
      __syncthreads();
    }
    if (block_size >= 128) {
      if (tid < 64) {
        update(dists, dists_i, tid, tid + 64);
      }
      __syncthreads();
    }
    if (block_size >= 64) {
      if (tid < 32) {
        update(dists, dists_i, tid, tid + 32);
      }
      __syncthreads();
    }
    if (block_size >= 32) {
      if (tid < 16) {
        update(dists, dists_i, tid, tid + 16);
      }
      __syncthreads();
    }
    if (block_size >= 16) {
      if (tid < 8) {
        update(dists, dists_i, tid, tid + 8);
      }
      __syncthreads();
    }
    if (block_size >= 8) {
      if (tid < 4) {
        update(dists, dists_i, tid, tid + 4);
      }
      __syncthreads();
    }
    if (block_size >= 4) {
      if (tid < 2) {
        update(dists, dists_i, tid, tid + 2);
      }
      __syncthreads();
    }
    if (block_size >= 2) {
      if (tid < 1) {
        update(dists, dists_i, tid, tid + 1);
      }
      __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0) idxs[j] = old + xyz_batch_start_idx;
  }
}

#endif  // STACK_FURTHEST_POINT_SAMPLE_CUDA_KERNEL_CUH
