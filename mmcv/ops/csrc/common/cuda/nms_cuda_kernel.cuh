// Copyright (c) OpenMMLab. All rights reserved
#ifndef NMS_CUDA_KERNEL_CUH
#define NMS_CUDA_KERNEL_CUH

#include <float.h>
#ifdef MMCV_WITH_TRT
#include "common_cuda_helper.hpp"
#else  // MMCV_WITH_TRT
#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else  // MMCV_USE_PARROTS
#include "pytorch_cuda_helper.hpp"
#endif  // MMCV_USE_PARROTS
#endif  // MMCV_WITH_TRT

int const threadsPerBlock = sizeof(unsigned long long int) * 8;

__device__ inline bool devIoU(float const *const a, float const *const b,
                              const int offset, const float threshold) {
  float left = fmaxf(a[0], b[0]), right = fminf(a[2], b[2]);
  float top = fmaxf(a[1], b[1]), bottom = fminf(a[3], b[3]);
  float width = fmaxf(right - left + offset, 0.f),
        height = fmaxf(bottom - top + offset, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + offset) * (a[3] - a[1] + offset);
  float Sb = (b[2] - b[0] + offset) * (b[3] - b[1] + offset);
  return interS > threshold * (Sa + Sb - interS);
}

__global__ void nms_cuda(const int n_boxes, const float iou_threshold,
                         const int offset, const float *dev_boxes,
                         unsigned long long *dev_mask) {
  int blocks = (n_boxes + threadsPerBlock - 1) / threadsPerBlock;
  CUDA_2D_KERNEL_BLOCK_LOOP(col_start, blocks, row_start, blocks) {
    const int tid = threadIdx.x;

    if (row_start > col_start) return;

    const int row_size =
        fminf(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size =
        fminf(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

    __shared__ float block_boxes[threadsPerBlock * 4];
    if (tid < col_size) {
      block_boxes[tid * 4 + 0] =
          dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 0];
      block_boxes[tid * 4 + 1] =
          dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 1];
      block_boxes[tid * 4 + 2] =
          dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 2];
      block_boxes[tid * 4 + 3] =
          dev_boxes[(threadsPerBlock * col_start + tid) * 4 + 3];
    }
    __syncthreads();

    if (tid < row_size) {
      const int cur_box_idx = threadsPerBlock * row_start + tid;
      const float *cur_box = dev_boxes + cur_box_idx * 4;
      int i = 0;
      unsigned long long int t = 0;
      int start = 0;
      if (row_start == col_start) {
        start = tid + 1;
      }
      for (i = start; i < col_size; i++) {
        if (devIoU(cur_box, block_boxes + i * 4, offset, iou_threshold)) {
          t |= 1ULL << i;
        }
      }
      dev_mask[cur_box_idx * gridDim.y + col_start] = t;
    }
  }
}

__global__ void gather_keep_from_mask_parallize(
    bool *keep, const unsigned long long *dev_mask, const int n_boxes) {
  const int col_blocks = (n_boxes + threadsPerBlock - 1) / threadsPerBlock;
  const int tid = threadIdx.x;

  extern __shared__ unsigned long long remv[];
  __shared__ const unsigned long long *p[1];
  __shared__ bool finish[1];
  __shared__ int nblock_ptr[1];

  for (int i = tid; i < col_blocks; i += blockDim.x) {
    remv[i] = 0;
  }
  if (tid == 0) {
    finish[0] = false;
  }
  __syncthreads();

  int nblock = 0;
  int inblock = 0;
  int i = 0;
  bool do_sync = false;

  while (!finish[0]) {
    if (tid == 0) {
      do_sync = false;
      for (; nblock < col_blocks; ++nblock) {
#pragma unroll
        for (; inblock < threadsPerBlock; ++inblock) {
          if (i < n_boxes && !(remv[nblock] & (1ULL << inblock))) {
            keep[i] = true;
            p[0] = dev_mask + i * col_blocks;
            i += 1;
            inblock = (inblock + 1) % threadsPerBlock;
            nblock_ptr[0] = nblock;
            nblock += inblock == 0 ? 1 : 0;
            do_sync = true;
            break;
          } else {
            i += 1;
          }
        }
        if (do_sync) break;
        inblock = 0;
      }
      if (!do_sync) {
        finish[0] = true;
      }
    }
    __syncthreads();

    if (!finish[0]) {
      for (int j = tid; j < col_blocks; j += blockDim.x) {
        if (j >= nblock_ptr[0]) remv[j] |= p[0][j];
      }
      __syncthreads();
    }
  }
}

#endif  // NMS_CUDA_KERNEL_CUH
