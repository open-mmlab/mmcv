// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include "diopi.hpp"
#endif

void border_align_forward_impl(const Tensor &input, const Tensor &boxes,
                               Tensor output, Tensor argmax_idx,
                               const int pool_size) {
  DISPATCH_DEVICE_IMPL(border_align_forward_impl, input, boxes, output,
                       argmax_idx, pool_size);
}

void border_align_backward_impl(const Tensor &grad_output, const Tensor &boxes,
                                const Tensor &argmax_idx, Tensor grad_input,
                                const int pool_size) {
  DISPATCH_DEVICE_IMPL(border_align_backward_impl, grad_output, boxes,
                       argmax_idx, grad_input, pool_size);
}

void border_align_forward(const Tensor &input, const Tensor &boxes,
                          Tensor output, Tensor argmax_idx,
                          const int pool_size) {
#ifdef MMCV_WITH_DIOPI
  auto input_p = reinterpret_cast<diopiConstTensorHandle_t>(&input);
  diopiDevice_t device;
  diopiGetTensorDevice(input_p, &device);
  if (device == diopi_host) {
      border_align_forward_impl(input, boxes, output, argmax_idx, pool_size);
      return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto boxes_p = reinterpret_cast<diopiConstTensorHandle_t>(&boxes);
  auto output_p = reinterpret_cast<diopiTensorHandle_t>(&output);
  auto argmax_idx_p = reinterpret_cast<diopiTensorHandle_t>(&argmax_idx);
  diopiBorderAlign(ch, input_p, boxes_p, output_p, argmax_idx_p, pool_size);
#else
  border_align_forward_impl(input, boxes, output, argmax_idx, pool_size);
#endif
}

void border_align_backward(const Tensor &grad_output, const Tensor &boxes,
                           const Tensor &argmax_idx, Tensor grad_input,
                           const int pool_size) {
#ifdef MMCV_WITH_DIOPI
  auto grad_output_p = reinterpret_cast<diopiConstTensorHandle_t>(&grad_output);
  diopiDevice_t device;
  diopiGetTensorDevice(grad_output_p, &device);
  if (device == diopi_host) {
      border_align_backward_impl(grad_output, boxes, argmax_idx, grad_input,
                             pool_size);
      return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto boxes_p = reinterpret_cast<diopiConstTensorHandle_t>(&boxes);
  auto argmax_idx_p = reinterpret_cast<diopiConstTensorHandle_t>(&argmax_idx);
  auto grad_input_p = reinterpret_cast<diopiTensorHandle_t>(&grad_input);
  diopiBorderAlignBackward(ch, grad_output_p, boxes_p, argmax_idx_p, grad_input_p,
                             pool_size);
#else
  border_align_backward_impl(grad_output, boxes, argmax_idx, grad_input,
                             pool_size);
#endif
}
