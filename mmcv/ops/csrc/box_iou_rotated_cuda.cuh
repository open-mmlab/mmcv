// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_cuda.cu
#ifndef BOX_IOU_ROTATED_CUDA_CUH
#define BOX_IOU_ROTATED_CUDA_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif
#include "box_iou_rotated_utils.hpp"

// 2D block with 32 * 16 = 512 threads per block
const int BLOCK_DIM_X = 32;
const int BLOCK_DIM_Y = 16;

inline int divideUP(const int x, const int y) { return (((x) + (y)-1) / (y)); }

template <typename T>
__global__ void box_iou_rotated_cuda_kernel(const int n_boxes1,
                                            const int n_boxes2,
                                            const T* dev_boxes1,
                                            const T* dev_boxes2, T* dev_ious,
                                            const bool aligned) {

  if (aligned) {
    CUDA_1D_KERNEL_LOOP(index, n_boxes1) {
      int b1 = index;
      int b2 = index;

      int base1 = b1 * 5;

      const int row_start = blockIdx.x * blockDim.x;
      const int col_start = blockIdx.y * blockDim.y;

      const int row_size = min(n_boxes1 - row_start, blockDim.x);
      const int col_size = min(n_boxes2 - col_start, blockDim.y);

      __shared__ float block_boxes1[BLOCK_DIM_X * 5];
      __shared__ float block_boxes2[BLOCK_DIM_Y * 5];

      if (threadIdx.x < row_size && threadIdx.y == 0) {
        block_boxes1[base1 + 0] = dev_boxes1[base1 + 0];
        block_boxes1[base1 + 1] = dev_boxes1[base1 + 1];
        block_boxes1[base1 + 2] = dev_boxes1[base1 + 2];
        block_boxes1[base1 + 3] = dev_boxes1[base1 + 3];
        block_boxes1[base1 + 4] = dev_boxes1[base1 + 4];
      }

      int base2 = b2 * 5;

      if (threadIdx.x < col_size && threadIdx.y == 0) {
        block_boxes2[base2 + 0] = dev_boxes2[base2 + 0];
        block_boxes2[base2 + 1] = dev_boxes2[base2 + 1];
        block_boxes2[base2 + 2] = dev_boxes2[base2 + 2];
        block_boxes2[base2 + 3] = dev_boxes2[base2 + 3];
        block_boxes2[base2 + 4] = dev_boxes2[base2 + 4];
      }

      __syncthreads();

      if (threadIdx.x < row_size && threadIdx.y < col_size) {
        int offset =
            (row_start + threadIdx.x) * n_boxes2 + col_start + threadIdx.y;
        dev_ious[offset] = single_box_iou_rotated<T>(
            block_boxes1 + threadIdx.x * 5, block_boxes2 + threadIdx.y * 5);
      }

    }
  } else {
    CUDA_1D_KERNEL_LOOP(index, n_boxes1 * n_boxes2) {
      int b1 = index / n_boxes2;
      int b2 = index % n_boxes2;

      int base1 = b1 * 5;

      const int row_start = blockIdx.x * blockDim.x;
      const int col_start = blockIdx.y * blockDim.y;

      const int row_size = min(n_boxes1 - row_start, blockDim.x);
      const int col_size = min(n_boxes2 - col_start, blockDim.y);

      __shared__ float block_boxes1[BLOCK_DIM_X * 5];
      __shared__ float block_boxes2[BLOCK_DIM_Y * 5];

      if (threadIdx.x < row_size && threadIdx.y == 0) {
        block_boxes1[base1 + 0] = dev_boxes1[base1 + 0];
        block_boxes1[base1 + 1] = dev_boxes1[base1 + 1];
        block_boxes1[base1 + 2] = dev_boxes1[base1 + 2];
        block_boxes1[base1 + 3] = dev_boxes1[base1 + 3];
        block_boxes1[base1 + 4] = dev_boxes1[base1 + 4];
      }

      int base2 = b2 * 5;

      if (threadIdx.x < col_size && threadIdx.y == 0) {
        block_boxes2[base2 + 0] = dev_boxes2[base2 + 0];
        block_boxes2[base2 + 1] = dev_boxes2[base2 + 1];
        block_boxes2[base2 + 2] = dev_boxes2[base2 + 2];
        block_boxes2[base2 + 3] = dev_boxes2[base2 + 3];
        block_boxes2[base2 + 4] = dev_boxes2[base2 + 4];
      }

      __syncthreads();

      if (threadIdx.x < row_size && threadIdx.y < col_size) {
        int offset =
            (row_start + threadIdx.x) * n_boxes2 + col_start + threadIdx.y;
        dev_ious[offset] = single_box_iou_rotated<T>(
            block_boxes1 + threadIdx.x * 5, block_boxes2 + threadIdx.y * 5);
      }
    }
  }
}

#endif
