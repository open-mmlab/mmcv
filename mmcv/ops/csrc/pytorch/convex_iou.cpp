// Copyright (c) OpenMMLab. All rights reserved
// modified from
// https://github.com/SDL-GuoZonghao/BeyondBoundingBox/tree/main/mmdet/ops/iou/src
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include "diopi.hpp"
#endif

void convex_iou_impl(const Tensor pointsets, const Tensor polygons,
                     Tensor ious) {
  DISPATCH_DEVICE_IMPL(convex_iou_impl, pointsets, polygons, ious);
}

void convex_iou(const Tensor pointsets, const Tensor polygons, Tensor ious) {
#ifdef MMCV_WITH_DIOPI
  auto pointsets_p = toDiopiTensorHandle(pointsets);
  diopiDevice_t device;
  diopiGetTensorDevice(pointsets_p, &device);
  diopiDtype_t dtype;
  diopiGetTensorDtype(pointsets_p, &dtype);
  if (device == diopi_host || dtype == diopi_dtype_float16) {
    convex_iou_impl(pointsets, polygons, ious);
    return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto polygons_p = toDiopiTensorHandle(polygons);
  auto ious_p = toDiopiTensorHandle(ious);
  if (&diopiConvexIouMmcv) {
    diopiConvexIouMmcv(ch, ious_p, pointsets_p, polygons_p);
  } else {
    convex_iou_impl(pointsets, polygons, ious);
  }
#else
  convex_iou_impl(pointsets, polygons, ious);
#endif
}

void convex_giou_impl(const Tensor pointsets, const Tensor polygons,
                      Tensor output) {
  DISPATCH_DEVICE_IMPL(convex_giou_impl, pointsets, polygons, output);
}

void convex_giou(const Tensor pointsets, const Tensor polygons, Tensor output) {
#ifdef MMCV_WITH_DIOPI
  auto pointsets_p = toDiopiTensorHandle(pointsets);
  diopiDevice_t device;
  diopiGetTensorDevice(pointsets_p, &device);
  diopiDtype_t dtype;
  diopiGetTensorDtype(pointsets_p, &dtype);
  if (device == diopi_host || dtype == diopi_dtype_float16) {
    convex_giou_impl(pointsets, polygons, output);
    return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto polygons_p = toDiopiTensorHandle(polygons);
  auto output_p = toDiopiTensorHandle(output);
  if (&diopiConvexGiouMmcv) {
    diopiConvexGiouMmcv(ch, output_p, pointsets_p, polygons_p);
  } else {
    convex_giou_impl(pointsets, polygons, output);
  }
#else
  convex_giou_impl(pointsets, polygons, output);
#endif
}
