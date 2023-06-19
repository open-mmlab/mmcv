// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include "csrc_dipu/diopirt/diopirt_impl.h"

using dipu::diopi_helper::toDiopiScalar;
using dipu::diopi_helper::toDiopiTensorHandle;
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

#ifdef MMCV_WITH_DIOPI
void roi_align_forward_diopi(Tensor input, Tensor rois, Tensor output,
                             Tensor argmax_y, Tensor argmax_x,
                             int aligned_height, int aligned_width,
                             float spatial_scale, int sampling_ratio,
                             int pool_mode, bool aligned) {
  auto input_p = toDiopiTensorHandle(input);
  diopiDevice_t device;
  diopiGetTensorDevice(input_p, &device);
  if (device == diopi_host) {
    roi_align_forward_impl(input, rois, output, argmax_y, argmax_x,
                           aligned_height, aligned_width, spatial_scale,
                           sampling_ratio, pool_mode, aligned);
    return;
  }
  diopiContext ctx(dipu::getCurrentDIPUStream().rawstream());
  diopiContextHandle_t ch = &ctx;
  auto rois_p = toDiopiTensorHandle(rois);
  auto out_p = toDiopiTensorHandle(output);
  auto argmax_y_p = toDiopiTensorHandle(argmax_y);
  auto argmax_x_p = toDiopiTensorHandle(argmax_x);
  bool is_mock_cuda = input.device().type() == c10::DeviceType::PrivateUse1;
  if (is_mock_cuda && reinterpret_cast<void*>(diopiRoiAlignMmcv) != nullptr) {
    auto ret = diopiRoiAlignMmcv(
        ch, out_p, argmax_y_p, argmax_x_p, input_p, rois_p, aligned_height,
        aligned_width, sampling_ratio, pool_mode, spatial_scale, aligned);
    if (ret == diopiSuccess) return;
  }
  LOG(WARNING) << "Fallback to cpu: mmcv ext op roi_align_forward";
  auto input_cpu = input.cpu();
  auto rois_cpu = rois.cpu();
  auto out_cpu = output.cpu();
  auto argmax_y_cpu = argmax_y.cpu();
  auto argmax_x_cpu = argmax_x.cpu();
  roi_align_forward_impl(input_cpu, rois_cpu, out_cpu, argmax_y_cpu,
                         argmax_x_cpu, aligned_height, aligned_width,
                         spatial_scale, sampling_ratio, pool_mode, aligned);
  output.copy_(out_cpu);
}

void roi_align_backward_diopi(Tensor grad_output, Tensor rois, Tensor argmax_y,
                              Tensor argmax_x, Tensor grad_input,
                              int aligned_height, int aligned_width,
                              float spatial_scale, int sampling_ratio,
                              int pool_mode, bool aligned) {
  auto grad_output_ = toDiopiTensorHandle(grad_output);
  diopiDevice_t device;
  diopiGetTensorDevice(grad_output_, &device);
  if (device == diopi_host) {
    roi_align_backward_impl(grad_output, rois, argmax_y, argmax_x, grad_input,
                            aligned_height, aligned_width, spatial_scale,
                            sampling_ratio, pool_mode, aligned);
    return;
  }
  auto rois_ = toDiopiTensorHandle(rois);
  auto argmax_y_ = toDiopiTensorHandle(argmax_y);
  auto argmax_x_ = toDiopiTensorHandle(argmax_x);
  auto grad_input_ = toDiopiTensorHandle(grad_input);
  diopiContext ctx(dipu::getCurrentDIPUStream().rawstream());
  diopiContextHandle_t ch = &ctx;
  bool is_mock_cuda = grad_output.device().type() == c10::DeviceType::PrivateUse1;
  if (is_mock_cuda && reinterpret_cast<void*>(diopiRoiAlignBackwardMmcv) != nullptr) {
    auto ret = diopiRoiAlignBackwardMmcv(ch, grad_input_, grad_output_, rois_,
                                         argmax_y_, argmax_x_, aligned_height,
                                         aligned_width, sampling_ratio,
                                         pool_mode, spatial_scale, aligned);
    if (ret == diopiSuccess) return;
  }
  LOG(WARNING) << "Fallback to cpu: mmcv ext op roi_align_backward";
  auto grad_output_cpu = grad_output.cpu();
  auto rois_cpu = rois.cpu();
  auto argmax_y_cpu = argmax_y.cpu();
  auto argmax_x_cpu = argmax_x.cpu();
  auto grad_input_cpu = grad_input.cpu();
  roi_align_backward_impl(grad_output_cpu, rois_cpu, argmax_y_cpu, argmax_x_cpu,
                          grad_input_cpu, aligned_height, aligned_width,
                          spatial_scale, sampling_ratio, pool_mode, aligned);
  grad_input.copy_(grad_input_cpu);
}
#endif

void roi_align_forward(Tensor input, Tensor rois, Tensor output,
                       Tensor argmax_y, Tensor argmax_x, int aligned_height,
                       int aligned_width, float spatial_scale,
                       int sampling_ratio, int pool_mode, bool aligned) {
#ifdef MMCV_WITH_DIOPI
  roi_align_forward_diopi(input, rois, output, argmax_y, argmax_x,
                          aligned_height, aligned_width, spatial_scale,
                          sampling_ratio, pool_mode, aligned);
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
  roi_align_backward_diopi(grad_output, rois, argmax_y, argmax_x, grad_input,
                           aligned_height, aligned_width, spatial_scale,
                           sampling_ratio, pool_mode, aligned);
#else
  roi_align_backward_impl(grad_output, rois, argmax_y, argmax_x, grad_input,
                          aligned_height, aligned_width, spatial_scale,
                          sampling_ratio, pool_mode, aligned);
#endif
}
