// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include "diopi.hpp"
#endif

void deform_roi_pool_forward_impl(Tensor input, Tensor rois, Tensor offset,
                                  Tensor output, int pooled_height,
                                  int pooled_width, float spatial_scale,
                                  int sampling_ratio, float gamma) {
  DISPATCH_DEVICE_IMPL(deform_roi_pool_forward_impl, input, rois, offset,
                       output, pooled_height, pooled_width, spatial_scale,
                       sampling_ratio, gamma);
}

void deform_roi_pool_backward_impl(Tensor grad_output, Tensor input,
                                   Tensor rois, Tensor offset,
                                   Tensor grad_input, Tensor grad_offset,
                                   int pooled_height, int pooled_width,
                                   float spatial_scale, int sampling_ratio,
                                   float gamma) {
  DISPATCH_DEVICE_IMPL(deform_roi_pool_backward_impl, grad_output, input, rois,
                       offset, grad_input, grad_offset, pooled_height,
                       pooled_width, spatial_scale, sampling_ratio, gamma);
}

void deform_roi_pool_forward(Tensor input, Tensor rois, Tensor offset,
                             Tensor output, int pooled_height, int pooled_width,
                             float spatial_scale, int sampling_ratio,
                             float gamma) {
#ifdef MMCV_WITH_DIOPI
  auto input_p = reinterpret_cast<diopiTensorHandle_t>(&input);
  diopiDevice_t device;
  diopiGetTensorDevice(input_p, &device);
  if (device == diopi_host) {
      deform_roi_pool_forward_impl(input, rois, offset, output, pooled_height,
                               pooled_width, spatial_scale, sampling_ratio,
                               gamma);
      return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto rois_p = reinterpret_cast<diopiTensorHandle_t>(&rois);
  auto offset_p = reinterpret_cast<diopiTensorHandle_t>(&offset);
  auto output_p = reinterpret_cast<diopiTensorHandle_t>(&output);
  diopiDeformRoiPool(ch, input_p, rois_p, offset_p, output_p, pooled_height,
                               pooled_width, spatial_scale, sampling_ratio,
                               gamma);
#else
  deform_roi_pool_forward_impl(input, rois, offset, output, pooled_height,
                               pooled_width, spatial_scale, sampling_ratio,
                               gamma);
#endif
}

void deform_roi_pool_backward(Tensor grad_output, Tensor input, Tensor rois,
                              Tensor offset, Tensor grad_input,
                              Tensor grad_offset, int pooled_height,
                              int pooled_width, float spatial_scale,
                              int sampling_ratio, float gamma) {
#ifdef MMCV_WITH_DIOPI
  auto grad_output_p = reinterpret_cast<diopiTensorHandle_t>(&grad_output);
  diopiDevice_t device;
  diopiGetTensorDevice(grad_output_p, &device);
  if (device == diopi_host) {
      deform_roi_pool_backward_impl(grad_output, input, rois, offset, grad_input,
                                grad_offset, pooled_height, pooled_width,
                                spatial_scale, sampling_ratio, gamma);
      return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto input_p = reinterpret_cast<diopiTensorHandle_t>(&input);
  auto rois_p = reinterpret_cast<diopiTensorHandle_t>(&rois);
  auto offset_p = reinterpret_cast<diopiTensorHandle_t>(&offset);
  auto grad_input_p = reinterpret_cast<diopiTensorHandle_t>(&grad_input);
  auto grad_offset_p = reinterpret_cast<diopiTensorHandle_t>(&grad_offset);
  diopiDeformRoiPoolBackward(ch, grad_output_p, input_p, rois_p, offset_p, grad_input_p,
                                grad_offset_p, pooled_height, pooled_width,
                                spatial_scale, sampling_ratio, gamma);
#else
  deform_roi_pool_backward_impl(grad_output, input, rois, offset, grad_input,
                                grad_offset, pooled_height, pooled_width,
                                spatial_scale, sampling_ratio, gamma);
#endif
}
