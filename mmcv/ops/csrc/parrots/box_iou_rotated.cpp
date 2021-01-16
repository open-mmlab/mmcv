// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated.h
#include "pytorch_cpp_helper.hpp"

void box_iou_rotated_cpu(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                         const int mode_flag, const bool aligned);

#ifdef MMCV_WITH_CUDA
void box_iou_rotated_cuda(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                          const int mode_flag, const bool aligned);
#endif

// Interface for Python
// inline is needed to prevent multiple function definitions when this header is
// included by different cpps
void box_iou_rotated(const Tensor boxes1, const Tensor boxes2, Tensor ious,
                     const int mode_flag, const bool aligned) {
  assert(boxes1.device().is_cuda() == boxes2.device().is_cuda());
  if (boxes1.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    box_iou_rotated_cuda(boxes1, boxes2, ious, mode_flag, aligned);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  } else {
    box_iou_rotated_cpu(boxes1, boxes2, ious, mode_flag, aligned);
  }
}
