// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated.h
#include "pytorch_cpp_helper.hpp"

Tensor box_iou_rotated_cpu(const Tensor boxes1, const Tensor boxes2);

#ifdef MMCV_WITH_CUDA
Tensor box_iou_rotated_cuda(const Tensor boxes1, const Tensor boxes2);
#endif

// Interface for Python
// inline is needed to prevent multiple function definitions when this header is
// included by different cpps
Tensor box_iou_rotated(const Tensor boxes1, const Tensor boxes2) {
  assert(boxes1.device().is_cuda() == boxes2.device().is_cuda());
  if (boxes1.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    return box_iou_rotated_cuda(boxes1, boxes2);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  return box_iou_rotated_cpu(boxes1, boxes2);
}
