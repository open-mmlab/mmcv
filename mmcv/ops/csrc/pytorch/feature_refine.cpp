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
                            const float spatial_scale, const int points,
                            Tensor output) {
  if (features.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(features);
    CHECK_CUDA_INPUT(best_bboxes);
    CHECK_CUDA_INPUT(output);

    return feature_refine_forward_impl(features, best_bboxes, spatial_scale,
                                       points, output);
#else
    AT_ERROR("feature_refine is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("feature_refine is not implemented on CPU");
  }
}

void feature_refine_backward(const Tensor top_grad, const Tensor best_bboxes,
                             const float spatial_scale, const int points,
                             Tensor bottom_grad) {
  if (top_grad.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(top_grad);
    CHECK_CUDA_INPUT(best_bboxes);
    CHECK_CUDA_INPUT(bottom_grad);

    return feature_refine_backward_impl(top_grad, best_bboxes, spatial_scale,
                                        points, bottom_grad);
#else
    AT_ERROR("feature_refine is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("feature_refine is not implemented on CPU");
  }
}
