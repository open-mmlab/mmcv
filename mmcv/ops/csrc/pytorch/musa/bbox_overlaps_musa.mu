// Copyright (c) OpenMMLab. All rights reserved
#include "bbox_overlaps_musa_kernel.muh"
#include "pytorch_musa_helper.hpp"


template <>
__global__ void bbox_overlaps_musa_kernel<at::Half>(
    const at::Half* bbox1, const at::Half* bbox2, at::Half* ious,
    const int num_bbox1, const int num_bbox2, const int mode,
    const bool aligned, const int offset) {
  bbox_overlaps_musa_kernel_half(reinterpret_cast<const __half*>(bbox1),
                                 reinterpret_cast<const __half*>(bbox2),
                                 reinterpret_cast<__half*>(ious), num_bbox1,
                                 num_bbox2, mode, aligned, offset);
}


void BBoxOverlapsMUSAKernelLauncher(const Tensor bboxes1, const Tensor bboxes2,
                                    Tensor ious, const int mode,
                                    const bool aligned, const int offset) {
  int output_size = ious.numel();
  int num_bbox1 = bboxes1.size(0);
  int num_bbox2 = bboxes2.size(0);

  c10::musa::MUSAGuard device_guard(bboxes1.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();
  AT_DISPATCH_FLOATING_TYPES(
      bboxes1.scalar_type(), "bbox_overlaps_musa_kernel", ([&] {
        bbox_overlaps_musa_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
                bboxes1.data_ptr<scalar_t>(), bboxes2.data_ptr<scalar_t>(),
                ious.data_ptr<scalar_t>(), num_bbox1, num_bbox2, mode, aligned,
                offset);
      }));
  AT_MUSA_CHECK(musaGetLastError());
}
