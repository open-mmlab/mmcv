// Copyright (c) OpenMMLab. All rights reserved
#include "convex_iou_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

void ConvexIoUCUDAKernelLauncher(const Tensor pointsets, const Tensor polygons,
                    Tensor ious) {
  int output_size = ious.numel();
  int num_pointsets = pointsets.size(0);
  int num_polygons = polygons.size(0);

  at::cuda::CUDAGuard device_guard(pointsets.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      pointsets.scalar_type(), "convex_iou_cuda_kernel", ([&] {
        convex_iou_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                num_pointsets, num_polygons,
                pointsets.data_ptr<scalar_t>(), polygons.data_ptr<scalar_t>(),
                ious.data_ptr<scalar_t>());
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}
