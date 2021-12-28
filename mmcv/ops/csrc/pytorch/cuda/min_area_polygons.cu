// Copyright (c) OpenMMLab. All rights reserved
// modified from
// https://github.com/SDL-GuoZonghao/BeyondBoundingBox/blob/main/mmdet/ops/minareabbox/src/minareabbox_kernel.cu
#include "min_area_polygons_cuda.cuh"
#include "pytorch_cuda_helper.hpp"

void MinAreaPolygonsCUDAKernelLauncher(const Tensor pointsets,
                                       Tensor polygons) {
  int num_pointsets = pointsets.size(0);
  const int output_size = polygons.numel();
  at::cuda::CUDAGuard device_guard(pointsets.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  min_area_polygons_cuda_kernel<<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0,
                                  stream>>>(
      num_pointsets, pointsets.data_ptr<float>(), polygons.data_ptr<float>());
  AT_CUDA_CHECK(cudaGetLastError());
}
