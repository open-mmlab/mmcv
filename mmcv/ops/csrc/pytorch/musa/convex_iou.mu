// Copyright (c) OpenMMLab. All rights reserved
// modified from
// https://github.com/SDL-GuoZonghao/BeyondBoundingBox/blob/main/mmdet/ops/iou/src/convex_iou_kernel.cu
#include "convex_iou_musa_kernel.muh"
#include "pytorch_musa_helper.hpp"

void ConvexIoUMUSAKernelLauncher(const Tensor pointsets, const Tensor polygons,
                                 Tensor ious) {
  int output_size = ious.numel();
  int num_pointsets = pointsets.size(0);
  int num_polygons = polygons.size(0);

  at::musa::MUSAGuard device_guard(pointsets.device());
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  AT_DISPATCH_FLOATING_TYPES(
      pointsets.scalar_type(), "convex_iou_musa_kernel", ([&] {
        convex_iou_musa_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK / 2, 0, stream>>>(
                num_pointsets, num_polygons, pointsets.data_ptr<scalar_t>(),
                polygons.data_ptr<scalar_t>(), ious.data_ptr<scalar_t>());
      }));
  AT_MUSA_CHECK(musaGetLastError());
}

void ConvexGIoUMUSAKernelLauncher(const Tensor pointsets, const Tensor polygons,
                                  Tensor output) {
  int output_size = output.numel();
  int num_pointsets = pointsets.size(0);
  int num_polygons = polygons.size(0);

  at::musa::MUSAGuard device_guard(pointsets.device());
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  AT_DISPATCH_FLOATING_TYPES(
      pointsets.scalar_type(), "convex_giou_musa_kernel", ([&] {
        convex_giou_musa_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK / 2, 0, stream>>>(
                num_pointsets, num_polygons, pointsets.data_ptr<scalar_t>(),
                polygons.data_ptr<scalar_t>(), output.data_ptr<scalar_t>());
      }));
  AT_MUSA_CHECK(musaGetLastError());
}
