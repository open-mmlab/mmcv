// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include "csrc_dipu/diopirt/diopirt_impl.h"
#include "csrc_dipu/base/base_def.h"

using dipu::diopi_helper::toDiopiScalar;
using dipu::diopi_helper::toDiopiTensorHandle;
#endif

Tensor nms_impl(Tensor boxes, Tensor scores, float iou_threshold, int offset) {
  return DISPATCH_DEVICE_IMPL(nms_impl, boxes, scores, iou_threshold, offset);
}

Tensor softnms_impl(Tensor boxes, Tensor scores, Tensor dets,
                    float iou_threshold, float sigma, float min_score,
                    int method, int offset) {
  return DISPATCH_DEVICE_IMPL(softnms_impl, boxes, scores, dets, iou_threshold,
                              sigma, min_score, method, offset);
}

std::vector<std::vector<int> > nms_match_impl(Tensor dets,
                                              float iou_threshold) {
  return DISPATCH_DEVICE_IMPL(nms_match_impl, dets, iou_threshold);
}

#ifdef MMCV_WITH_DIOPI
Tensor nms_diopi(Tensor boxes, Tensor scores, float iou_threshold, int offset) {
  auto boxes_p = toDiopiTensorHandle(boxes);
  diopiDevice_t device;
  diopiGetTensorDevice(boxes_p, &device);
  if (device == diopi_host) {
    return nms_impl(boxes, scores, iou_threshold, offset);
  }
  diopiContext ctx(dipu::getCurrentDIPUStream().rawstream());
  diopiContextHandle_t ch = &ctx;
  Tensor out;
  auto outp = toDiopiTensorHandle(out);
  diopiTensorHandle_t* outhandle = &outp;
  auto scores_p = toDiopiTensorHandle(scores);
  bool is_mock_cuda = boxes.device().type() == dipu::DIPU_DEVICE_TYPE;
  if (is_mock_cuda && reinterpret_cast<void*>(diopiNmsMmcv) != nullptr) {
    auto ret =
        diopiNmsMmcv(ch, outhandle, boxes_p, scores_p, iou_threshold, offset);
    if (ret == diopiSuccess) {
      auto tensorhandle = reinterpret_cast<Tensor*>(*outhandle);
      return *tensorhandle;
    }
  }
  LOG(WARNING) << "Fallback to cpu: mmcv ext op nms";
  auto boxes_cpu = boxes.cpu();
  auto scores_cpu = scores.cpu();
  return nms_impl(boxes_cpu, scores_cpu, iou_threshold, offset);
}
#endif

Tensor nms(Tensor boxes, Tensor scores, float iou_threshold, int offset) {
#ifdef MMCV_WITH_DIOPI
  return nms_diopi(boxes, scores, iou_threshold, offset);
#else
  return nms_impl(boxes, scores, iou_threshold, offset);
#endif
}

Tensor softnms(Tensor boxes, Tensor scores, Tensor dets, float iou_threshold,
               float sigma, float min_score, int method, int offset) {
  return softnms_impl(boxes, scores, dets, iou_threshold, sigma, min_score,
                      method, offset);
}

std::vector<std::vector<int> > nms_match(Tensor dets, float iou_threshold) {
  return nms_match_impl(dets, iou_threshold);
}
