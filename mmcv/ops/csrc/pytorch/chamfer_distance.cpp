// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/chrdiller/pyTorchChamferDistance/blob/master/chamfer_distance/chamfer_distance.cpp

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include "diopi.hpp"

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
  // chamfer_distance_forward_impl(xyz1, xyz2, dist1, dist2, idx1, idx2);
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  diopiTensorHandle_t* test = nullptr;
  auto xyz1_p = reinterpret_cast<diopiTensorHandle_t>(const_cast<Tensor*>(&xyz1));
  auto xyz2_p = reinterpret_cast<diopiTensorHandle_t>(const_cast<Tensor*>(&xyz2));
  auto dist1_p = reinterpret_cast<diopiTensorHandle_t>(const_cast<Tensor*>(&dist1));
  auto dist2_p = reinterpret_cast<diopiTensorHandle_t>(const_cast<Tensor*>(&dist2));
  auto idx1_p = reinterpret_cast<diopiTensorHandle_t>(const_cast<Tensor*>(&idx1));
  auto idx2_p = reinterpret_cast<diopiTensorHandle_t>(const_cast<Tensor*>(&idx2));
  diopiChamferDistance(ch, xyz1_p, xyz2_p, dist1_p, dist2_p, idx1_p, idx2_p);
}

void chamfer_distance_backward(const Tensor xyz1, const Tensor xyz2,
                               Tensor idx1, Tensor idx2, Tensor graddist1,
                               Tensor graddist2, Tensor gradxyz1,
                               Tensor gradxyz2) {
  // chamfer_distance_backward_impl(xyz1, xyz2, idx1, idx2, graddist1, graddist2,
  //                                gradxyz1, gradxyz2);
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto xyz1_p = reinterpret_cast<diopiTensorHandle_t>(const_cast<Tensor*>(&xyz1));
  auto xyz2_p = reinterpret_cast<diopiTensorHandle_t>(const_cast<Tensor*>(&xyz2));
  auto idx1_p = reinterpret_cast<diopiTensorHandle_t>(&idx1);
  auto idx2_p = reinterpret_cast<diopiTensorHandle_t>(&idx2);
  auto graddist1_p = reinterpret_cast<diopiTensorHandle_t>(&graddist1);
  auto graddist2_p = reinterpret_cast<diopiTensorHandle_t>(&graddist2);
  auto gradxyz1_p = reinterpret_cast<diopiTensorHandle_t>(&gradxyz1);
  auto gradxyz2_p = reinterpret_cast<diopiTensorHandle_t>(&gradxyz2);
  diopiChamferDistanceBackward(ch, xyz1_p, xyz2_p, idx1_p, idx2_p,graddist1_p,
                               graddist2_p, gradxyz1_p, gradxyz2_p);
}
