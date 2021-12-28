// Copyright (c) OpenMMLab. All rights reserved
// modified from
// https://github.com/SDL-GuoZonghao/BeyondBoundingBox/blob/main/mmdet/ops/iou/src/convex_iou_kernel.cu
#include "convex_iou_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

void ConvexIoUCUDAKernelLauncher(const Tensor pointsets, const Tensor polygons,
                                 Tensor ious) {
  int output_size = ious.numel();
  int num_pointsets = pointsets.size(0);
  int num_polygons = polygons.size(0);

  at::cuda::CUDAGuard device_guard(pointsets.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  convex_iou_cuda_kernel<<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0,
                           stream>>>(
      num_pointsets, num_polygons, pointsets.data_ptr<float>(),
      polygons.data_ptr<float>(), ious.data_ptr<float>());

  AT_CUDA_CHECK(cudaGetLastError());
}

void ConvexGIoUCUDAKernelLauncher(const Tensor pointsets, const Tensor polygons,
                                  Tensor output) {
  int output_size = output.numel();
  int num_pointsets = pointsets.size(0);
  int num_polygons = polygons.size(0);

  at::cuda::CUDAGuard device_guard(pointsets.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int optimal_block_num =
      (output_size + THREADS_PER_BLOCK / 8 - 1) / (THREADS_PER_BLOCK / 8);
  convex_giou_cuda_kernel<<<min(optimal_block_num, 4096), THREADS_PER_BLOCK / 8,
                            0, stream>>>(
      num_pointsets, num_polygons, pointsets.data_ptr<float>(),
      polygons.data_ptr<float>(), output.data_ptr<float>());

  AT_CUDA_CHECK(cudaGetLastError());
}
