// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>
#include "diopi.hpp"
#endif

void min_area_polygons_impl(const Tensor pointsets, Tensor polygons) {
  DISPATCH_DEVICE_IMPL(min_area_polygons_impl, pointsets, polygons);
}

void min_area_polygons(const Tensor pointsets, Tensor polygons) {
#ifdef MMCV_WITH_DIOPI
  auto pointsets_p = reinterpret_cast<diopiConstTensorHandle_t>(&pointsets);
  diopiDevice_t device;
  diopiGetTensorDevice(pointsets_p, &device);
  if (device == diopi_host) {
      min_area_polygons_impl(pointsets, polygons);
      return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto polygons_p = reinterpret_cast<diopiTensorHandle_t>(&polygons);
  if (&diopiMinAreaPolygons) {
   diopiMinAreaPolygons(ch, pointsets_p, polygons_p);
  } else {
   min_area_polygons_impl(pointsets, polygons);
  }
#else
  min_area_polygons_impl(pointsets, polygons);
#endif
}
