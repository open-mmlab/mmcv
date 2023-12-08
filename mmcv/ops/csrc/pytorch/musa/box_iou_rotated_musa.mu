// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_musa.cu
#include "box_iou_rotated_musa.muh"
#include "pytorch_musa_helper.hpp"

void box_iou_rotated_musa(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                          const int mode_flag, const bool aligned) {
  using scalar_t = float;
  AT_ASSERTM(boxes1.is_privateuseone(), "boxes1 must be a MUSA tensor");
  AT_ASSERTM(boxes2.is_privateuseone(), "boxes2 must be a MUSA tensor");

  int output_size = ious.numel();
  int num_boxes1 = boxes1.size(0);
  int num_boxes2 = boxes2.size(0);

  at::musa::MUSAGuard device_guard(boxes1.device());
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  box_iou_rotated_musa_kernel<scalar_t>
      <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
          num_boxes1, num_boxes2, boxes1.data_ptr<scalar_t>(),
          boxes2.data_ptr<scalar_t>(), (scalar_t*)ious.data_ptr<scalar_t>(),
          mode_flag, aligned);
  AT_MUSA_CHECK(musaGetLastError());
}
