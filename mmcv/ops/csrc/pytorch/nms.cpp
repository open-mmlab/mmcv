// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include "diopi.hpp"
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

Tensor nms(Tensor boxes, Tensor scores, float iou_threshold, int offset) {
#ifdef MMCV_WITH_DIOPI
  auto boxes_p = toDiopiTensorHandle(boxes);
  diopiDevice_t device;
  diopiGetTensorDevice(boxes_p, &device);
  diopiDtype_t dtype;
  diopiGetTensorDtype(boxes_p, &dtype);
  if (device == diopi_host || dtype == diopi_dtype_float16) {
    return nms_impl(boxes, scores, iou_threshold, offset);
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  Tensor out;
  auto outp = toDiopiTensorHandle(out);
  diopiTensorHandle_t* outhandle = &outp;
  auto scores_p = toDiopiTensorHandle(scores);
  if (&diopiNmsMmcv) {
    diopiNmsMmcv(ch, outhandle, boxes_p, scores_p, iou_threshold, offset);
    auto tensorhandle = reinterpret_cast<Tensor*>(*outhandle);
    return *tensorhandle;
  } else {
    return nms_impl(boxes, scores, iou_threshold, offset);
  }
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
