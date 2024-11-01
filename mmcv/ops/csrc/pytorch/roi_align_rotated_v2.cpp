// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void roi_align_rotated_v2_forward_impl(Tensor input, Tensor rois, Tensor output,
                                    double spatial_scale, int sampling_ratio,
                                    int pooled_height, int pooled_width,
                                    bool aligned, bool clockwise) {
  DISPATCH_DEVICE_IMPL(roi_align_rotated_v2_forward_impl, input, rois, output,
                       spatial_scale, sampling_ratio, pooled_height, pooled_width,
                       aligned, clockwise);
}


void roi_align_rotated_v2_forward(Tensor input, Tensor rois, Tensor output,
                               double spatial_scale, int sampling_ratio,
                               int pooled_height, int pooled_width,
                               bool aligned, bool clockwise) {
  roi_align_rotated_v2_forward_impl(input, rois, output, spatial_scale, sampling_ratio,
                                    pooled_height, pooled_width, aligned, clockwise);
}


void roi_align_rotated_v2_backward_impl(Tensor input, Tensor rois, Tensor grad_output, Tensor grad_input,
                                    int pooled_height, int pooled_width, double spatial_scale,
                                    int sampling_ratio, bool aligned, bool clockwise) {
  DISPATCH_DEVICE_IMPL(roi_align_rotated_v2_backward_impl, input, rois, grad_output, grad_input,
                    pooled_height, pooled_width, spatial_scale, sampling_ratio, aligned, clockwise);
}


void roi_align_rotated_v2_backward(Tensor input, Tensor rois, Tensor grad_output, Tensor grad_input,
                                    int pooled_height, int pooled_width, double spatial_scale,
                                    int sampling_ratio, bool aligned, bool clockwise) {
  roi_align_rotated_v2_backward_impl(input, rois, grad_output, grad_input,
                                    pooled_height, pooled_width, spatial_scale, sampling_ratio, aligned, clockwise);
}
