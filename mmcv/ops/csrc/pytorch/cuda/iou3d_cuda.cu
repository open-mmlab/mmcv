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
#include "pytorch_device_registry.hpp"

void IoU3DBoxesOverlapBevForwardCUDAKernelLauncher(const int num_a,
                                                   const Tensor boxes_a,
                                                   const int num_b,
                                                   const Tensor boxes_b,
                                                   Tensor ans_overlap) {
  at::cuda::CUDAGuard device_guard(boxes_a.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(DIVUP(num_b, THREADS_PER_BLOCK_IOU3D),
              DIVUP(num_a, THREADS_PER_BLOCK_IOU3D));
  dim3 threads(THREADS_PER_BLOCK_IOU3D, THREADS_PER_BLOCK_IOU3D);

  iou3d_boxes_overlap_bev_forward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      num_a, boxes_a.data_ptr<float>(), num_b, boxes_b.data_ptr<float>(),
      ans_overlap.data_ptr<float>());

  AT_CUDA_CHECK(cudaGetLastError());
}

void IoU3DBoxesIoUBevForwardCUDAKernelLauncher(const int num_a,
                                               const Tensor boxes_a,
                                               const int num_b,
                                               const Tensor boxes_b,
                                               Tensor ans_iou) {
  at::cuda::CUDAGuard device_guard(boxes_a.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // blockIdx.x(col), blockIdx.y(row)
  dim3 blocks(DIVUP(num_b, THREADS_PER_BLOCK_IOU3D),
              DIVUP(num_a, THREADS_PER_BLOCK_IOU3D));
  dim3 threads(THREADS_PER_BLOCK_IOU3D, THREADS_PER_BLOCK_IOU3D);

  iou3d_boxes_iou_bev_forward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      num_a, boxes_a.data_ptr<float>(), num_b, boxes_b.data_ptr<float>(),
      ans_iou.data_ptr<float>());

  AT_CUDA_CHECK(cudaGetLastError());
}

void IoU3DNMSForwardCUDAKernelLauncher(const Tensor boxes,
                                       unsigned long long *mask, int boxes_num,
                                       float nms_overlap_thresh) {
  at::cuda::CUDAGuard device_guard(boxes.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 blocks(DIVUP(boxes_num, THREADS_PER_BLOCK_NMS),
              DIVUP(boxes_num, THREADS_PER_BLOCK_NMS));
  dim3 threads(THREADS_PER_BLOCK_NMS);

  nms_forward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      boxes_num, nms_overlap_thresh, boxes.data_ptr<float>(), mask);

  AT_CUDA_CHECK(cudaGetLastError());
}

void IoU3DNMSNormalForwardCUDAKernelLauncher(const Tensor boxes,
                                             unsigned long long *mask,
                                             int boxes_num,
                                             float nms_overlap_thresh) {
  at::cuda::CUDAGuard device_guard(boxes.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 blocks(DIVUP(boxes_num, THREADS_PER_BLOCK_NMS),
              DIVUP(boxes_num, THREADS_PER_BLOCK_NMS));
  dim3 threads(THREADS_PER_BLOCK_NMS);

  nms_normal_forward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      boxes_num, nms_overlap_thresh, boxes.data_ptr<float>(), mask);

  AT_CUDA_CHECK(cudaGetLastError());
}

void iou3d_boxes_overlap_bev_forward_cuda(const int num_a, const Tensor boxes_a,
                                          const int num_b, const Tensor boxes_b,
                                          Tensor ans_overlap) {
  IoU3DBoxesOverlapBevForwardCUDAKernelLauncher(num_a, boxes_a, num_b, boxes_b,
                                                ans_overlap);
};

void iou3d_boxes_iou_bev_forward_cuda(const int num_a, const Tensor boxes_a,
                                      const int num_b, const Tensor boxes_b,
                                      Tensor ans_iou) {
  IoU3DBoxesIoUBevForwardCUDAKernelLauncher(num_a, boxes_a, num_b, boxes_b,
                                            ans_iou);
};

void iou3d_nms_forward_cuda(const Tensor boxes, unsigned long long *mask,
                            int boxes_num, float nms_overlap_thresh) {
  IoU3DNMSForwardCUDAKernelLauncher(boxes, mask, boxes_num, nms_overlap_thresh);
};

void iou3d_nms_normal_forward_cuda(const Tensor boxes, unsigned long long *mask,
                                   int boxes_num, float nms_overlap_thresh) {
  IoU3DNMSNormalForwardCUDAKernelLauncher(boxes, mask, boxes_num,
                                          nms_overlap_thresh);
};

void iou3d_boxes_overlap_bev_forward_impl(const int num_a, const Tensor boxes_a,
                                          const int num_b, const Tensor boxes_b,
                                          Tensor ans_overlap);

void iou3d_boxes_iou_bev_forward_impl(const int num_a, const Tensor boxes_a,
                                      const int num_b, const Tensor boxes_b,
                                      Tensor ans_iou);

void iou3d_nms_forward_impl(const Tensor boxes, unsigned long long *mask,
                            int boxes_num, float nms_overlap_thresh);

void iou3d_nms_normal_forward_impl(const Tensor boxes, unsigned long long *mask,
                                   int boxes_num, float nms_overlap_thresh);

REGISTER_DEVICE_IMPL(iou3d_boxes_overlap_bev_forward_impl, CUDA,
                     iou3d_boxes_overlap_bev_forward_cuda);
REGISTER_DEVICE_IMPL(iou3d_boxes_iou_bev_forward_impl, CUDA,
                     iou3d_boxes_iou_bev_forward_cuda);
REGISTER_DEVICE_IMPL(iou3d_nms_forward_impl, CUDA, iou3d_nms_forward_cuda);
REGISTER_DEVICE_IMPL(iou3d_nms_normal_forward_impl, CUDA,
                     iou3d_nms_normal_forward_cuda);
