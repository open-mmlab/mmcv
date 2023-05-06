// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include "diopi.hpp"
#endif


void roi_align_forward_impl(Tensor input, Tensor rois, Tensor output,
                            Tensor argmax_y, Tensor argmax_x,
                            int aligned_height, int aligned_width,
                            float spatial_scale, int sampling_ratio,
                            int pool_mode, bool aligned) {
  DISPATCH_DEVICE_IMPL(roi_align_forward_impl, input, rois, output, argmax_y,
                       argmax_x, aligned_height, aligned_width, spatial_scale,
                       sampling_ratio, pool_mode, aligned);
}

void roi_align_backward_impl(Tensor grad_output, Tensor rois, Tensor argmax_y,
                             Tensor argmax_x, Tensor grad_input,
                             int aligned_height, int aligned_width,
                             float spatial_scale, int sampling_ratio,
                             int pool_mode, bool aligned) {
  DISPATCH_DEVICE_IMPL(roi_align_backward_impl, grad_output, rois, argmax_y,
                       argmax_x, grad_input, aligned_height, aligned_width,
                       spatial_scale, sampling_ratio, pool_mode, aligned);
}

void roi_align_forward(Tensor input, Tensor rois, Tensor output,
                       Tensor argmax_y, Tensor argmax_x, int aligned_height,
                       int aligned_width, float spatial_scale,
                       int sampling_ratio, int pool_mode, bool aligned) {
#ifdef MMCV_WITH_DIOPI
  auto input_p = toDiopiTensorHandle(input);
  diopiDtype_t dtype;
  diopiDevice_t device;
  diopiGetTensorDtype(input_p, &dtype);
  diopiGetTensorDevice(input_p, &device);
  if (device == diopi_host|| dtype == diopi_dtype_float16) {
    roi_align_forward_impl(input, rois, output, argmax_y, argmax_x,
                         aligned_height, aligned_width, spatial_scale,
                         sampling_ratio, pool_mode, aligned);
    return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto rois_p = toDiopiTensorHandle(rois);
  auto out_p = toDiopiTensorHandle(output);
  auto argmax_y_p = toDiopiTensorHandle(argmax_y);
  auto argmax_x_p = toDiopiTensorHandle(argmax_x);
  if (&diopiRoiAlignMmcv) {
    diopiRoiAlignMmcv(ch, out_p, argmax_y_p, argmax_x_p, input_p, rois_p,
                  aligned_height, aligned_width,
                  sampling_ratio, pool_mode, spatial_scale, aligned);
  } else {
    roi_align_forward_impl(input, rois, output, argmax_y, argmax_x,
                            aligned_height, aligned_width, spatial_scale,
                            sampling_ratio, pool_mode, aligned);
  }
#else
  roi_align_forward_impl(input, rois, output, argmax_y, argmax_x,
                         aligned_height, aligned_width, spatial_scale,
                         sampling_ratio, pool_mode, aligned);
#endif
}

void roi_align_backward(Tensor grad_output, Tensor rois, Tensor argmax_y,
                        Tensor argmax_x, Tensor grad_input, int aligned_height,
                        int aligned_width, float spatial_scale,
                        int sampling_ratio, int pool_mode, bool aligned) {
#ifdef MMCV_WITH_DIOPI
  auto grad_output_ = toDiopiTensorHandle(grad_output);

  diopiDevice_t device;
  diopiDtype_t dtype;
  diopiGetTensorDtype(grad_output_, &dtype);
  diopiGetTensorDevice(grad_output_, &device);
  if (device == diopi_host|| dtype == diopi_dtype_float16) {
    roi_align_backward_impl(grad_output, rois, argmax_y, argmax_x, grad_input,
                          aligned_height, aligned_width, spatial_scale,
                          sampling_ratio, pool_mode, aligned);
    return;
  }
  auto rois_ = toDiopiTensorHandle(rois);
  auto argmax_y_ = toDiopiTensorHandle(argmax_y);
  auto argmax_x_ = toDiopiTensorHandle(argmax_x);
  auto grad_input_ = toDiopiTensorHandle(grad_input);
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  if(&diopiRoiAlignBackwardMmcv){
    diopiRoiAlignBackwardMmcv(ch, grad_input_, grad_output_, rois_, argmax_y_, argmax_x_,
                           aligned_height, aligned_width,
                          spatial_scale, pool_mode, sampling_ratio, aligned);
  } else {
    roi_align_backward_impl(grad_output, rois, argmax_y, argmax_x, grad_input,
                          aligned_height, aligned_width, spatial_scale,
                          sampling_ratio, pool_mode, aligned);
  }
#else
  roi_align_backward_impl(grad_output, rois, argmax_y, argmax_x, grad_input,
                          aligned_height, aligned_width, spatial_scale,
                          sampling_ratio, pool_mode, aligned);
#endif

}
