// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/nms_rotated/nms_rotated.h
#include "pytorch_cpp_helper.hpp"

Tensor nms_rotated_cpu(const Tensor dets, const Tensor scores,
                       const float iou_threshold);

#ifdef MMCV_WITH_CUDA
Tensor nms_rotated_cuda(const Tensor dets, const Tensor scores,
                        const Tensor order, const Tensor dets_sorted,
                        const float iou_threshold, const int multi_label);
#endif

#ifdef MMCV_WITH_NPU
Tensor nms_rotated_npu(const Tensor dets, const Tensor scores,
                       const Tensor labels, const float iou_threshold);
#endif

#ifdef MMCV_WITH_MLU
Tensor nms_rotated_mlu(const Tensor dets, const Tensor scores,
                       const float iou_threshold);
#endif

// Interface for Python
// inline is needed to prevent multiple function definitions when this header is
// included by different cpps
Tensor nms_rotated(const Tensor dets, const Tensor scores, const Tensor order,
                   const Tensor dets_sorted, const Tensor labels,
                   const float iou_threshold, const int multi_label) {
  assert(dets.device().is_cuda() == scores.device().is_cuda());
  if (dets.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    return nms_rotated_cuda(dets, scores, order, dets_sorted.contiguous(),
                            iou_threshold, multi_label);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
#ifdef MMCV_WITH_XLA
  } else if (dets.device().type() == at::kXLA) {
    return nms_rotated_npu(dets, scores, labels, iou_threshold);
#endif
#ifdef MMCV_WITH_KPRIVATE
  } else if (dets.device().type() == at::kPrivateUse1) {
    return nms_rotated_npu(dets, scores, labels, iou_threshold);
#endif
#ifdef MMCV_WITH_MLU
  } else if (dets.device().type() == at::kMLU) {
    return nms_rotated_mlu(dets, scores, iou_threshold);
#endif
  }

  return nms_rotated_cpu(dets.contiguous(), scores.contiguous(), iou_threshold);
}
