// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void bezier_align_forward_impl(Tensor input, Tensor rois, Tensor output,
                               int aligned_height, int aligned_width,
                               float spatial_scale, int sampling_ratio,
                               bool aligned) {
  DISPATCH_DEVICE_IMPL(bezier_align_forward_impl, input, rois, output,
                       aligned_height, aligned_width, spatial_scale,
                       sampling_ratio, aligned);
}

void bezier_align_backward_impl(Tensor grad_output, Tensor rois,
                                Tensor grad_input, int aligned_height,
                                int aligned_width, float spatial_scale,
                                int sampling_ratio, bool aligned) {
  DISPATCH_DEVICE_IMPL(bezier_align_backward_impl, grad_output, rois,
                       grad_input, aligned_height, aligned_width, spatial_scale,
                       sampling_ratio, aligned);
}

void bezier_align_forward(Tensor input, Tensor rois, Tensor output,
                          int aligned_height, int aligned_width,
                          float spatial_scale, int sampling_ratio,
                          bool aligned) {
  bezier_align_forward_impl(input, rois, output, aligned_height, aligned_width,
                            spatial_scale, sampling_ratio, aligned);
}

void bezier_align_backward(Tensor grad_output, Tensor rois, Tensor grad_input,
                           int aligned_height, int aligned_width,
                           float spatial_scale, int sampling_ratio,
                           bool aligned) {
  bezier_align_backward_impl(grad_output, rois, grad_input, aligned_height,
                             aligned_width, spatial_scale, sampling_ratio,
                             aligned);
}
