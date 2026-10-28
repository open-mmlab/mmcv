// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include "pytorch_cpp_helper.hpp"

Tensor nms_quadri_cpu(const Tensor dets, const Tensor scores,
                      const float iou_threshold);

#ifdef MMCV_WITH_CUDA
Tensor nms_quadri_cuda(const Tensor dets, const Tensor scores,
                       const Tensor order, const Tensor dets_sorted,
                       const float iou_threshold, const int multi_label);
#endif
#ifdef MMCV_WITH_MUSA
Tensor nms_quadri_musa(const Tensor dets, const Tensor scores,
                       const Tensor order, const Tensor dets_sorted,
                       const float iou_threshold, const int multi_label);
#endif

// Interface for Python
// inline is needed to prevent multiple function definitions when this header is
// included by different cpps
Tensor nms_quadri(const Tensor dets, const Tensor scores, const Tensor order,
                  const Tensor dets_sorted, const float iou_threshold,
                  const int multi_label) {
  TORCH_CHECK(dets.device() == scores.device(),
              "dets and scores must be on the same device");
  if (dets.is_cuda()) {
#ifdef MMCV_WITH_CUDA
    return nms_quadri_cuda(dets, scores, order, dets_sorted, iou_threshold,
                          multi_label);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  }
#ifdef MMCV_WITH_MUSA
  if (dets.is_privateuseone()) {
    return nms_quadri_musa(dets, scores, order, dets_sorted, iou_threshold,
                          multi_label);
  }
#endif
  TORCH_CHECK(dets.is_cpu(), "nms_quadri does not support this device");
  return nms_quadri_cpu(dets, scores, iou_threshold);
}
