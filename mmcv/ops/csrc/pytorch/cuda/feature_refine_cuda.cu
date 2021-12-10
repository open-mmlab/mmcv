// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/SJTU-Thinklab-Det/r3det-on-mmdetection/blob/master/mmdet/ops/fr/src/feature_refine_kernel.cu
#include "feature_refine_cuda_kernel.cuh"
#include "pytorch_cuda_helper.hpp"

int FRForwardLauncher(const torch::Tensor features,
                      const torch::Tensor best_bboxes,  // of shape (n, h, w, 5)
                      const float spatial_scale, const int points,
                      torch::Tensor output) {
  const int output_size = features.numel();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.scalar_type(), "FRForwardLaucherFun", ([&] {
        const scalar_t* bottom_data = features.data_ptr<scalar_t>();
        const scalar_t* bboxes_data = best_bboxes.data_ptr<scalar_t>();
        scalar_t* top_data = output.data_ptr<scalar_t>();

        feature_refine_forward_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, points, bottom_data, bboxes_data,
                scalar_t(spatial_scale), features.size(1), features.size(2),
                features.size(3), top_data);
      }));
  return 1;
}

int FRBackwardLauncher(const torch::Tensor top_grad,
                       const torch::Tensor best_bboxes,
                       const float spatial_scale, const int points,
                       torch::Tensor bottom_grad) {
  const int output_size = top_grad.numel();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.scalar_type(), "FRBackwardLaucherFun", ([&] {
        const scalar_t* top_diff = top_grad.data_ptr<scalar_t>();
        const scalar_t* bboxes_data = best_bboxes.data_ptr<scalar_t>();
        scalar_t* bottom_diff = bottom_grad.data_ptr<scalar_t>();

        feature_refine_backward_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, points, top_diff, bboxes_data,
                scalar_t(spatial_scale), top_grad.size(1), top_grad.size(2),
                top_grad.size(3), bottom_diff);
      }));
  return 1;
}
