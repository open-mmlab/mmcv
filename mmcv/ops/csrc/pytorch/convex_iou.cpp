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
  auto pointsets_p = reinterpret_cast<diopiConstTensorHandle_t>(&pointsets);
  diopiDevice_t device;
  diopiGetTensorDevice(pointsets_p, &device);
  if (device == diopi_host) {
      convex_iou_impl(pointsets, polygons, ious);
      return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto polygons_p = reinterpret_cast<diopiConstTensorHandle_t>(&polygons);
  auto ious_p = reinterpret_cast<diopiTensorHandle_t>(&ious);
  if (&diopiConvexIou) {
   diopiConvexIou(ch, pointsets_p, polygons_p, ious_p);
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
  auto pointsets_p = reinterpret_cast<diopiConstTensorHandle_t>(&pointsets);
  diopiDevice_t device;
  diopiGetTensorDevice(pointsets_p, &device);
  if (device == diopi_host) {
      convex_giou_impl(pointsets, polygons, output);
      return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto polygons_p = reinterpret_cast<diopiConstTensorHandle_t>(&polygons);
  auto output_p = reinterpret_cast<diopiTensorHandle_t>(&output);
  if (&diopiConvexGiou) {
   diopiConvexGiou(ch, pointsets_p, polygons_p, output_p);
  } else {
   convex_giou_impl(pointsets, polygons, output);
  }
#else
  convex_giou_impl(pointsets, polygons, output);
#endif
}
