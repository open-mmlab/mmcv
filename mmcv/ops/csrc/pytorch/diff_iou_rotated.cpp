// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include "diopi.hpp"
#endif

Tensor diff_iou_rotated_sort_vertices_forward_impl(Tensor vertices, Tensor mask,
                                                   Tensor num_valid) {
  return DISPATCH_DEVICE_IMPL(diff_iou_rotated_sort_vertices_forward_impl,
                              vertices, mask, num_valid);
}

Tensor diff_iou_rotated_sort_vertices_forward(Tensor vertices, Tensor mask,
                                              Tensor num_valid) {
#ifdef MMCV_WITH_DIOPI
  auto vertices_p = toDiopiTensorHandle(vertices);
  diopiDevice_t device;
  diopiGetTensorDevice(vertices_p, &device);
  diopiDtype_t dtype;
  diopiGetTensorDtype(vertices_p, &dtype);
  if (device == diopi_host || dtype == diopi_dtype_float16) {
    return diff_iou_rotated_sort_vertices_forward_impl(vertices, mask,
                                                       num_valid);
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  Tensor out;
  auto outp = toDiopiTensorHandle(out);
  diopiTensorHandle_t* outhandle = &outp;
  auto mask_p = toDiopiTensorHandle(mask);
  auto num_valid_p = toDiopiTensorHandle(num_valid);
  if (&diopiDiffIouRotatedSortVertices) {
    diopiDiffIouRotatedSortVertices(ch, outhandle, vertices_p, mask_p,
                                    num_valid_p);
    auto tensorhandle = reinterpret_cast<Tensor*>(*outhandle);
    return *tensorhandle;
  } else {
    return diff_iou_rotated_sort_vertices_forward_impl(vertices, mask,
                                                       num_valid);
  }
#else
  return diff_iou_rotated_sort_vertices_forward_impl(vertices, mask, num_valid);
#endif
}
