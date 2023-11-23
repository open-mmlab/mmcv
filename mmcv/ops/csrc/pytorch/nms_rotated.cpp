// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/nms_rotated/nms_rotated.h
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include "csrc_dipu/base/basedef.h"
#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiScalar;
using dipu::diopi_helper::toDiopiTensorHandle;
#endif

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

#ifdef MMCV_WITH_DIOPI
Tensor nms_rotated_diopi(const Tensor dets, const Tensor scores,
                         const Tensor order, const Tensor dets_sorted,
                         const Tensor labels, const float iou_threshold,
                         const int multi_label) {
  auto dets_p = toDiopiTensorHandle(dets);
  diopiDevice_t device;
  diopiGetTensorDevice(dets_p, &device);
  if (device == diopi_host) {
    return nms_rotated_cpu(dets.contiguous(), scores.contiguous(),
                           iou_threshold);
  }
  diopiContext ctx(dipu::getCurrentDIPUStream().rawstream());
  diopiContextHandle_t ch = &ctx;
  Tensor out;
  auto out_p = toDiopiTensorHandle(out);
  diopiTensorHandle_t *out_handle = &out_p;
  auto scores_p = toDiopiTensorHandle(scores);
  auto order_p = toDiopiTensorHandle(order);
  auto dets_sorted_p = toDiopiTensorHandle(dets_sorted);
  auto labels_p = toDiopiTensorHandle(labels);
  bool is_mock_cuda = dets.device().type() == dipu::DIPU_DEVICE_TYPE;
  if (is_mock_cuda &&
      reinterpret_cast<void *>(diopiNmsRotatedMmcv) != nullptr) {
    auto ret = diopiNmsRotatedMmcv(ch, out_handle, dets_p, scores_p, order_p,
                                   dets_sorted_p, labels_p, iou_threshold,
                                   static_cast<bool>(multi_label));
    if (ret == diopiSuccess) {
      auto tensorhandle = reinterpret_cast<Tensor *>(*out_handle);
      return *tensorhandle;
    }
  }
  LOG(WARNING) << "Fallback to cpu: mmcv ext op nms_rotated";
  auto dets_cpu = dets.cpu();
  auto scores_cpu = scores.cpu();
  return nms_rotated_cpu(dets_cpu, scores_cpu, iou_threshold);
}
#endif

// Interface for Python
// inline is needed to prevent multiple function definitions when this header is
// included by different cpps
Tensor nms_rotated(const Tensor dets, const Tensor scores, const Tensor order,
                   const Tensor dets_sorted, const Tensor labels,
                   const float iou_threshold, const int multi_label) {
#ifdef MMCV_WITH_DIOPI
  return nms_rotated_diopi(dets, scores, order, dets_sorted, labels,
                           iou_threshold, multi_label);
#endif
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
