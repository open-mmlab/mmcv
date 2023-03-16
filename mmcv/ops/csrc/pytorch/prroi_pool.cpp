// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>
#include "diopi.hpp"
#endif

void prroi_pool_forward_impl(Tensor input, Tensor rois, Tensor output,
                             int pooled_height, int pooled_width,
                             float spatial_scale) {
  DISPATCH_DEVICE_IMPL(prroi_pool_forward_impl, input, rois, output,
                       pooled_height, pooled_width, spatial_scale);
}

void prroi_pool_backward_impl(Tensor grad_output, Tensor rois,
                              Tensor grad_input, int pooled_height,
                              int pooled_width, float spatial_scale) {
  DISPATCH_DEVICE_IMPL(prroi_pool_backward_impl, grad_output, rois, grad_input,
                       pooled_height, pooled_width, spatial_scale);
}

void prroi_pool_coor_backward_impl(Tensor output, Tensor grad_output,
                                   Tensor input, Tensor rois, Tensor grad_rois,
                                   int pooled_height, int pooled_width,
                                   float spatial_scale) {
  DISPATCH_DEVICE_IMPL(prroi_pool_coor_backward_impl, output, grad_output,
                       input, rois, grad_rois, pooled_height, pooled_width,
                       spatial_scale);
}

void prroi_pool_forward(Tensor input, Tensor rois, Tensor output,
                        int pooled_height, int pooled_width,
                        float spatial_scale) {
#ifdef MMCV_WITH_DIOPI
  auto input_p = reinterpret_cast<diopiTensorHandle_t>(&input);
  diopiDevice_t device;
  diopiGetTensorDevice(input_p, &device);
  if (device == diopi_host) {
      prroi_pool_forward_impl(input, rois, output, pooled_height, pooled_width,
                          spatial_scale);
      return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto rois_p = reinterpret_cast<diopiTensorHandle_t>(&rois);
  auto output_p = reinterpret_cast<diopiTensorHandle_t>(&output);
  // diopiPrroiPool(ch, input_p, rois_p, output_p, pooled_height, pooled_width,
  //               spatial_scale);
#else
  prroi_pool_forward_impl(input, rois, output, pooled_height, pooled_width,
                          spatial_scale);
#endif
}

void prroi_pool_backward(Tensor grad_output, Tensor rois, Tensor grad_input,
                         int pooled_height, int pooled_width,
                         float spatial_scale) {
#ifdef MMCV_WITH_DIOPI
  auto grad_output_p = reinterpret_cast<diopiTensorHandle_t>(&grad_output);
  diopiDevice_t device;
  diopiGetTensorDevice(grad_output_p, &device);
  if (device == diopi_host) {
      prroi_pool_backward_impl(grad_output, rois, grad_input, pooled_height,
                           pooled_width, spatial_scale);
      return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto rois_p = reinterpret_cast<diopiTensorHandle_t>(&rois);
  auto grad_input_p = reinterpret_cast<diopiTensorHandle_t>(&grad_input);
  // diopiPrroiPoolbackward(ch, grad_output_p, rois_p, grad_input_p, pooled_height,
  //                       pooled_width, spatial_scale);
#else
  prroi_pool_backward_impl(grad_output, rois, grad_input, pooled_height,
                           pooled_width, spatial_scale);
#endif
}

void prroi_pool_coor_backward(Tensor output, Tensor grad_output, Tensor input,
                              Tensor rois, Tensor grad_rois, int pooled_height,
                              int pooled_width, float spatial_scale) {
#ifdef MMCV_WITH_DIOPI
  auto grad_output_p = reinterpret_cast<diopiTensorHandle_t>(&grad_output);
  diopiDevice_t device;
  diopiGetTensorDevice(grad_output_p, &device);
  if (device == diopi_host) {
      prroi_pool_coor_backward_impl(output, grad_output, input, rois, grad_rois,
                                pooled_height, pooled_width, spatial_scale);
      return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto output_p = reinterpret_cast<diopiTensorHandle_t>(&output);
  auto input_p = reinterpret_cast<diopiTensorHandle_t>(&input);
  auto rois_p = reinterpret_cast<diopiTensorHandle_t>(&rois);
  auto grad_rois_p = reinterpret_cast<diopiTensorHandle_t>(&grad_rois);
  // diopiPrroiPoolCoorBackward(ch, output_p, grad_output_p, input_p, rois_p,
  //    grad_rois_p, pooled_height, pooled_width, spatial_scale);
#else
  prroi_pool_coor_backward_impl(output, grad_output, input, rois, grad_rois,
                                pooled_height, pooled_width, spatial_scale);
#endif
}
