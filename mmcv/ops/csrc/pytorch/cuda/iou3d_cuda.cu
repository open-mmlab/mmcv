// Modified from
// https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/ops/iou3d_nms/src/iou3d_nms_kernel.cu

/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/

#include <stdio.h>

#include "iou3d_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

void IoU3DBoxesOverlapBevForwardCUDAKernelLauncher(const int num_a,
                                                   const float *boxes_a,
                                                   const int num_b,
                                                   const float *boxes_b,
                                                   float *ans_overlap) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 blocks(
      DIVUP(num_b, THREADS_PER_BLOCK),
      DIVUP(num_a, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

  iou3d_boxes_overlap_bev_forward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      num_a, boxes_a, num_b, boxes_b, ans_overlap);

  AT_CUDA_CHECK(cudaGetLastError());
}

void IoU3DBoxesIoUBevForwardCUDAKernelLauncher(const int num_a,
                                               const float *boxes_a,
                                               const int num_b,
                                               const float *boxes_b,
                                               float *ans_iou) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 blocks(
      DIVUP(num_b, THREADS_PER_BLOCK),
      DIVUP(num_a, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

  iou3d_boxes_iou_bev_forward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      num_a, boxes_a, num_b, boxes_b, ans_iou);

  AT_CUDA_CHECK(cudaGetLastError());
}

void IoU3DNMSForwardCUDAKernelLauncher(const float *boxes,
                                       unsigned long long *mask, int boxes_num,
                                       float nms_overlap_thresh) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 blocks(DIVUP(boxes_num, THREADS_PER_BLOCK_NMS),
              DIVUP(boxes_num, THREADS_PER_BLOCK_NMS));
  dim3 threads(THREADS_PER_BLOCK_NMS);

  nms_forward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      boxes_num, nms_overlap_thresh, boxes, mask);

  AT_CUDA_CHECK(cudaGetLastError());
}

void IoU3DNMSNormalForwardCUDAKernelLauncher(const float *boxes,
                                             unsigned long long *mask,
                                             int boxes_num,
                                             float nms_overlap_thresh) {
  dim3 blocks(DIVUP(boxes_num, THREADS_PER_BLOCK_NMS),
              DIVUP(boxes_num, THREADS_PER_BLOCK_NMS));
  dim3 threads(THREADS_PER_BLOCK_NMS);

  nms_normal_forward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      boxes_num, nms_overlap_thresh, boxes, mask);

  AT_CUDA_CHECK(cudaGetLastError());
}
