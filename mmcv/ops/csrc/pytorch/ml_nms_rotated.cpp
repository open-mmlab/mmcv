// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include "pytorch_cpp_helper.hpp"

Tensor nms_rotated_cpu(
    const Tensor dets,
    const Tensor scores,
    const Tensor labels,
    const float iou_threshold);

#ifdef MMCV_WITH_CUDA
Tensor nms_rotated_cuda(
    const Tensor dets,
    const Tensor scores,
    const Tensor labels,
    const float iou_threshold);
#endif

// Interface for Python
// inline is needed to prevent multiple function definitions when this header is
// included by different cpps
Tensor ml_nms_rotated(
    const Tensor dets,
    const Tensor scores,
    const Tensor labels,
    const float iou_threshold) {
  assert(dets.device().is_cuda() == scores.device().is_cuda());
  if (dets.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    return nms_rotated_cuda(dets, scores, labels, iou_threshold);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  return nms_rotated_cpu(dets, scores, labels, iou_threshold);
}
