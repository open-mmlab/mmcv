// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/SJTU-Thinklab-Det/r3det-on-mmdetection/blob/master/mmdet/ops/fr/src/feature_refine_cuda.cpp

#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
int FRForwardLauncher(const Tensor features, const Tensor best_bboxes,
                      const float spatial_scale, const int points,
                      Tensor output);
int feature_refine_forward_cuda(const Tensor features, const Tensor best_bboxes,
                                const float spatial_scale, const int points,
                                Tensor output) {
  int FRForwardLauncher(const Tensor features, const Tensor best_bboxes,
                        const float spatial_scale, const int points,
                        Tensor output);
};

int FRBackwardLauncher(const Tensor top_grad, const Tensor best_bboxes,
                       const float spatial_scale, const int points,
                       Tensor bottom_grad);
int feature_refine_backward_cuda(const Tensor top_grad,
                                 const Tensor best_bboxes,
                                 const float spatial_scale, const int points,
                                 Tensor bottom_grad) {
  int FRBackwardLauncher(const Tensor top_grad, const Tensor best_bboxes,
                         const float spatial_scale, const int points,
                         Tensor bottom_grad);
};
#endif

int feature_refine_forward(const Tensor features, const Tensor best_bboxes,
                           const float spatial_scale, const int points,
                           Tensor output) {
  if (features.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(features);
    CHECK_CUDA_INPUT(best_bboxes);
    CHECK_CUDA_INPUT(output);

    return feature_refine_forward_cuda(features, best_bboxes, spatial_scale,
                                       points, output);
#else
    AT_ERROR("feature_refine is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("feature_refine is not implemented on CPU");
  }
}

int feature_refine_backward(const Tensor top_grad, const Tensor best_bboxes,
                            const float spatial_scale, const int points,
                            Tensor bottom_grad) {
  if (top_grad.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(top_grad);
    CHECK_CUDA_INPUT(best_bboxes);
    CHECK_CUDA_INPUT(bottom_grad);

    return feature_refine_backward_cuda(top_grad, best_bboxes, spatial_scale,
                                        points, bottom_grad);
#else
    AT_ERROR("feature_refine is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("feature_refine is not implemented on CPU");
  }
}
