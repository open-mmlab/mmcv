// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/
#include "min_area_polygons_cuda.cuh"
#include "pytorch_cuda_helper.hpp"

void MinAreaPolygonsCUDAKernelLauncher(const Tensor pointsets,
                                       Tensor polygons) {
  int num_pointsets = pointsets.size(0);
  at::cuda::CUDAGuard device_guard(pointsets.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  min_area_polygons_cuda_kernel<scalar_t>
      <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
          num_pointsets, pointsets.data_ptr<scalar_t>(),
          polygons.data_ptr<scalar_t>());
  AT_CUDA_CHECK(cudaGetLastError());
}
