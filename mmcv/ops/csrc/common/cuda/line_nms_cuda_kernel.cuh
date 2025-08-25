// Modified from
// https://github.com/Turoad/CLRNet by VBTI

// The functions below originates from Fast R-CNN
// See https://github.com/rbgirshick/py-faster-rcnn
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License
// Written by Shaoqing Ren

#ifndef LINE_NMS_CUDA_KERNEL_CUH
#define LINE_NMS_CUDA_KERNEL_CUH

#ifdef MMCV_USE_PARROTS
#include "parrots_cuda_helper.hpp"
#else
#include "pytorch_cuda_helper.hpp"
#endif

// Hard-coded maximum. Increase if needed.
#define MAX_COL_BLOCKS 1000
#define N_OFFSETS \
  72  // if you use more than 73 offsets you will have to adjust this value
#define N_STRIPS (N_OFFSETS - 1)
#define PROP_SIZE (5 + N_OFFSETS)
#define DATASET_OFFSET 0

#ifndef DIVUP2
#define DIVUP2(m, n) ((m) / (n) + ((m) % (n) > 0))
#endif

constexpr int64_t threadsPerBlock = sizeof(unsigned long long) * 8;

template <typename scalar_t>
__device__ inline bool devIoU(scalar_t const *const a, scalar_t const *const b,
                              const scalar_t threshold) {
  const int start_a = static_cast<int>(a[2] * N_STRIPS - DATASET_OFFSET +
                                       0.5f);  // 0.5 rounding trick
  const int start_b = static_cast<int>(b[2] * N_STRIPS - DATASET_OFFSET + 0.5f);
  const int start = max(start_a, start_b);
  const int end_a =
      start_a + static_cast<int>(a[4]) - 1 +
      static_cast<int>(
          0.5f - ((a[4] - 1) < 0));  //  - (x<0) trick to adjust for negative
                                     //  numbers (in case length is 0)
  const int end_b = start_b + static_cast<int>(b[4]) - 1 +
                    static_cast<int>(0.5f - ((b[4] - 1) < 0));
  const int end = min(min(end_a, end_b), N_OFFSETS - 1);

  if (end < start || start < 0 || end >= N_OFFSETS) return false;

  scalar_t dist = 0;
  for (int i = 5 + start; i <= 5 + end && i < PROP_SIZE; ++i) {
    dist += abs(a[i] - b[i]);
  }
  return dist < (threshold * (end - start + 1));
}

template <typename scalar_t>
__global__ void line_nms_cuda_forward_cuda_kernel(
    const int64_t n_boxes, const scalar_t nms_overlap_thresh,
    const scalar_t *dev_boxes, const int64_t *idx, int64_t *dev_mask) {
  const int64_t row_start = blockIdx.y;
  const int64_t col_start = blockIdx.x;

  if (row_start > col_start) return;

  const int64_t row_size =
      min((n_boxes - row_start * threadsPerBlock), threadsPerBlock);
  const int64_t col_size =
      min((n_boxes - col_start * threadsPerBlock), threadsPerBlock);

  if (row_size <= 0 || col_size <= 0) return;

  __shared__ scalar_t shared_boxes[threadsPerBlock * PROP_SIZE];

  if (threadIdx.x < col_size) {
    const int64_t box_idx = idx[threadsPerBlock * col_start + threadIdx.x];
    if (box_idx >= 0 && box_idx < n_boxes) {  // Add bounds check
      for (int i = 0; i < PROP_SIZE; ++i) {
        shared_boxes[threadIdx.x * PROP_SIZE + i] =
            dev_boxes[box_idx * PROP_SIZE + i];
      }
    }
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const scalar_t *cur_box = dev_boxes + cur_box_idx * PROP_SIZE;
    unsigned long long t = 0;
    int start = (row_start == col_start) ? threadIdx.x + 1 : 0;

    for (int i = start; i < col_size; i++) {
      if (devIoU<scalar_t>(cur_box, shared_boxes + i * PROP_SIZE,
                           nms_overlap_thresh)) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP2(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
    // }
    // }
  }
}

__global__ void nms_collect(const int64_t boxes_num, const int64_t col_blocks,
                            int64_t top_k, const int64_t *idx,
                            const int64_t *mask, int64_t *keep,
                            int64_t *parent_object_index,
                            int64_t *num_to_keep) {
  // Add safety check for col_blocks
  if (col_blocks > MAX_COL_BLOCKS) {
    *num_to_keep = 0;
    return;
  }
  int64_t num_to_keep_ = 0;

  // mark the bboxes which have been removed.
  extern __shared__ unsigned long long removed[];

  // initialize removed.
  const int tid = threadIdx.x;
  for (int i = tid; i < col_blocks; i += blockDim.x) {
    removed[i] = 0;
  }
  __syncthreads();

  for (int64_t i = 0; i < boxes_num; ++i) {
    parent_object_index[i] = 0;
  }

  for (int64_t i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(removed[nblock] & (1ULL << inblock))) {
      int64_t idxi = idx[i];
      keep[num_to_keep_] = idxi;
      const int64_t *p = mask + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        removed[j] |= p[j];
      }
      for (int j = i; j < boxes_num; j++) {
        int nblockj = j / threadsPerBlock;
        int inblockj = j % threadsPerBlock;
        if (p[nblockj] & (1ULL << inblockj))
          parent_object_index[idx[j]] = num_to_keep_ + 1;
      }
      parent_object_index[idx[i]] = num_to_keep_ + 1;

      num_to_keep_++;

      if (num_to_keep_ == top_k) break;
    }
  }

  // Initialize the rest of the keep array to avoid uninitialized values.
  for (int64_t i = num_to_keep_; i < boxes_num; ++i) keep[i] = 0;

  *num_to_keep = min(top_k, num_to_keep_);
}

#endif  // LINE_NMS_CUDA_KERNEL_CUH
