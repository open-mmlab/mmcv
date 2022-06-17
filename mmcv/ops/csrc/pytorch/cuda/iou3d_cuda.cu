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
                                                   const Tensor boxes_a,
                                                   const int num_b,
                                                   const Tensor boxes_b,
                                                   Tensor ans_overlap) {
  at::cuda::CUDAGuard device_guard(boxes_a.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(GET_BLOCKS(num_b, THREADS_PER_BLOCK_IOU3D),
              GET_BLOCKS(num_a, THREADS_PER_BLOCK_IOU3D));
  dim3 threads(THREADS_PER_BLOCK_IOU3D, THREADS_PER_BLOCK_IOU3D);

  iou3d_boxes_overlap_bev_forward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      num_a, boxes_a.data_ptr<float>(), num_b, boxes_b.data_ptr<float>(),
      ans_overlap.data_ptr<float>());

  AT_CUDA_CHECK(cudaGetLastError());
}

void IoU3DNMS3DForwardCUDAKernelLauncher(const Tensor boxes,
                                         unsigned long long *mask,
                                         int boxes_num,
                                         float nms_overlap_thresh) {
  at::cuda::CUDAGuard device_guard(boxes.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 blocks(GET_BLOCKS(boxes_num, THREADS_PER_BLOCK_NMS),
              GET_BLOCKS(boxes_num, THREADS_PER_BLOCK_NMS));
  dim3 threads(THREADS_PER_BLOCK_NMS);

  iou3d_nms3d_forward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      boxes_num, nms_overlap_thresh, boxes.data_ptr<float>(), mask);

  AT_CUDA_CHECK(cudaGetLastError());
}

void IoU3DNMS3DNormalForwardCUDAKernelLauncher(const Tensor boxes,
                                               unsigned long long *mask,
                                               int boxes_num,
                                               float nms_overlap_thresh) {
  at::cuda::CUDAGuard device_guard(boxes.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 blocks(GET_BLOCKS(boxes_num, THREADS_PER_BLOCK_NMS),
              GET_BLOCKS(boxes_num, THREADS_PER_BLOCK_NMS));
  dim3 threads(THREADS_PER_BLOCK_NMS);

  iou3d_nms3d_normal_forward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      boxes_num, nms_overlap_thresh, boxes.data_ptr<float>(), mask);

  AT_CUDA_CHECK(cudaGetLastError());
}
