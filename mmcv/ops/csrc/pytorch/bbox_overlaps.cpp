// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include "diopi.hpp"
#endif

void bbox_overlaps_impl(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                        const int mode, const bool aligned, const int offset) {
  DISPATCH_DEVICE_IMPL(bbox_overlaps_impl, bboxes1, bboxes2, ious, mode,
                       aligned, offset);
}

void bbox_overlaps(const Tensor bboxes1, const Tensor bboxes2, Tensor ious,
                   const int mode, const bool aligned, const int offset) {
#ifdef MMCV_WITH_DIOPI
  auto bboxes1_p = toDiopiTensorHandle(bboxes1);
  diopiDevice_t device;
  diopiGetTensorDevice(bboxes1_p, &device);
  if (device == diopi_host) {
    bbox_overlaps_impl(bboxes1, bboxes2, ious, mode, aligned, offset);
    return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto bboxes2_p = toDiopiTensorHandle(bboxes2);
  auto ious_p = toDiopiTensorHandle(ious);
  if (&diopiBboxOverlaps) {
    diopiBboxOverlaps(ch, bboxes1_p, bboxes2_p, ious_p, mode, aligned, offset);
  } else {
    bbox_overlaps_impl(bboxes1, bboxes2, ious, mode, aligned, offset);
  }
#else
  bbox_overlaps_impl(bboxes1, bboxes2, ious, mode, aligned, offset);
#endif
}
