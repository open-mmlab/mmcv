// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/SJTU-Thinklab-Det/r3det-on-mmdetection/blob/master/mmdet/ops/fr/src/feature_refine_cuda.cpp

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void feature_refine_forward_impl(const Tensor features,
                                 const Tensor best_bboxes,
                                 const float spatial_scale, const int points,
                                 Tensor output) {
  DISPATCH_DEVICE_IMPL(feature_refine_forward_impl, features, best_bboxes,
                       spatial_scale, points, output);
}

void feature_refine_backward_impl(const Tensor top_grad,
                                  const Tensor best_bboxes,
                                  const float spatial_scale, const int points,
                                  Tensor bottom_grad) {
  DISPATCH_DEVICE_IMPL(feature_refine_backward_impl, top_grad, best_bboxes,
                       spatial_scale, points, bottom_grad);
}

void feature_refine_forward(const Tensor features, const Tensor best_bboxes,
                            Tensor output, const float spatial_scale,
                            const int points) {
  feature_refine_forward_impl(features, best_bboxes, spatial_scale, points,
                              output);
}

void feature_refine_backward(const Tensor top_grad, const Tensor best_bboxes,
                             Tensor bottom_grad, const float spatial_scale,
                             const int points) {
  feature_refine_backward_impl(top_grad, best_bboxes, spatial_scale, points,
                               bottom_grad);
}
