// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/chrdiller/pyTorchChamferDistance/blob/master/chamfer_distance/chamfer_distance.cpp

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include "diopi.hpp"
#endif

void chamfer_distance_forward_impl(const Tensor xyz1, const Tensor xyz2,
                                   const Tensor dist1, const Tensor dist2,
                                   const Tensor idx1, const Tensor idx2) {
  DISPATCH_DEVICE_IMPL(chamfer_distance_forward_impl, xyz1, xyz2, dist1, dist2,
                       idx1, idx2);
}

void chamfer_distance_backward_impl(const Tensor xyz1, const Tensor xyz2,
                                    Tensor idx1, Tensor idx2, Tensor graddist1,
                                    Tensor graddist2, Tensor gradxyz1,
                                    Tensor gradxyz2) {
  DISPATCH_DEVICE_IMPL(chamfer_distance_backward_impl, xyz1, xyz2, idx1, idx2,
                       graddist1, graddist2, gradxyz1, gradxyz2);
}

void chamfer_distance_forward(const Tensor xyz1, const Tensor xyz2,
                              const Tensor dist1, const Tensor dist2,
                              const Tensor idx1, const Tensor idx2) {
#ifdef MMCV_WITH_DIOPI
  auto xyz1_p = toDiopiTensorHandle(xyz1);
  diopiDevice_t device;
  diopiGetTensorDevice(xyz1_p, &device);
  diopiDtype_t dtype;
  diopiGetTensorDtype(xyz1_p, &dtype);
  if (device == diopi_host || dtype == diopi_dtype_float16) {
    chamfer_distance_forward_impl(xyz1, xyz2, dist1, dist2, idx1, idx2);
    return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto xyz2_p = toDiopiTensorHandle(xyz2);
  auto dist1_p = toDiopiTensorHandleWithConstCase(dist1);
  auto dist2_p = toDiopiTensorHandleWithConstCase(dist2);
  auto idx1_p = toDiopiTensorHandleWithConstCase(idx1);
  auto idx2_p = toDiopiTensorHandleWithConstCase(idx2);
  if (&diopiChamferDistance) {
    diopiChamferDistance(ch, xyz1_p, xyz2_p, dist1_p, dist2_p, idx1_p, idx2_p);
  } else {
    chamfer_distance_forward_impl(xyz1, xyz2, dist1, dist2, idx1, idx2);
  }
#else
  chamfer_distance_forward_impl(xyz1, xyz2, dist1, dist2, idx1, idx2);
#endif
}

void chamfer_distance_backward(const Tensor xyz1, const Tensor xyz2,
                               Tensor idx1, Tensor idx2, Tensor graddist1,
                               Tensor graddist2, Tensor gradxyz1,
                               Tensor gradxyz2) {
#ifdef MMCV_WITH_DIOPI
  auto xyz1_p = toDiopiTensorHandle(xyz1);
  diopiDevice_t device;
  diopiGetTensorDevice(xyz1_p, &device);
  diopiDtype_t dtype;
  diopiGetTensorDtype(xyz1_p, &dtype);
  if (device == diopi_host || dtype == diopi_dtype_float16) {
    chamfer_distance_backward_impl(xyz1, xyz2, idx1, idx2, graddist1, graddist2,
                                   gradxyz1, gradxyz2);
    return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto xyz2_p = toDiopiTensorHandle(xyz2);
  auto idx1_p = toDiopiTensorHandle(idx1);
  auto idx2_p = toDiopiTensorHandle(idx2);
  auto graddist1_p = toDiopiTensorHandle(graddist1);
  auto graddist2_p = toDiopiTensorHandle(graddist2);
  auto gradxyz1_p = toDiopiTensorHandle(gradxyz1);
  auto gradxyz2_p = toDiopiTensorHandle(gradxyz2);
  if (&diopiChamferDistanceBackward) {
    diopiChamferDistanceBackward(ch, xyz1_p, xyz2_p, idx1_p, idx2_p,
                                 graddist1_p, graddist2_p, gradxyz1_p,
                                 gradxyz2_p);
  } else {
    chamfer_distance_backward_impl(xyz1, xyz2, idx1, idx2, graddist1, graddist2,
                                   gradxyz1, gradxyz2);
  }
#else
  chamfer_distance_backward_impl(xyz1, xyz2, idx1, idx2, graddist1, graddist2,
                                 gradxyz1, gradxyz2);
#endif
}
