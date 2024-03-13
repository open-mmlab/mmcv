// Copyright (c) OpenMMLab. All rights reserved
// modified from
// https://github.com/SDL-GuoZonghao/BeyondBoundingBox/blob/main/mmdet/ops/minareabbox/src/minareabbox_kernel.cu
#include "min_area_polygons_musa.muh"
#include "pytorch_musa_helper.hpp"

void MinAreaPolygonsMUSAKernelLauncher(const Tensor pointsets,
                                       Tensor polygons) {
  int num_pointsets = pointsets.size(0);
  const int output_size = polygons.numel();
  c10::musa::MUSAGuard device_guard(pointsets.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();
  AT_DISPATCH_FLOATING_TYPES(
      pointsets.scalar_type(), "min_area_polygons_musa_kernel", ([&] {
        min_area_polygons_musa_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                num_pointsets, pointsets.data_ptr<scalar_t>(),
                polygons.data_ptr<scalar_t>());
      }));
  AT_MUSA_CHECK(musaGetLastError());
}
