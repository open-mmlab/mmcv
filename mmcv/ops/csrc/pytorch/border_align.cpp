// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

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
  auto input_p = toDiopiTensorHandle(input);
  diopiDevice_t device;
  diopiGetTensorDevice(input_p, &device);
  diopiDtype_t dtype;
  diopiGetTensorDtype(input_p, &dtype);
  if (device == diopi_host || dtype == diopi_dtype_float16) {
    border_align_forward_impl(input, boxes, output, argmax_idx, pool_size);
    return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto boxes_p = toDiopiTensorHandle(boxes);
  auto output_p = toDiopiTensorHandle(output);
  auto argmax_idx_p = toDiopiTensorHandle(argmax_idx);
  if (&diopiBorderAlignMmcv) {
    diopiBorderAlignMmcv(ch, output_p, argmax_idx_p, input_p, boxes_p ,pool_size);
  } else {
    border_align_forward_impl(input, boxes, output, argmax_idx, pool_size);
  }
#else
  border_align_forward_impl(input, boxes, output, argmax_idx, pool_size);
#endif
}

void border_align_backward(const Tensor &grad_output, const Tensor &boxes,
                           const Tensor &argmax_idx, Tensor grad_input,
                           const int pool_size) {
#ifdef MMCV_WITH_DIOPI
  auto grad_output_p = toDiopiTensorHandle(grad_output);
  diopiDevice_t device;
  diopiGetTensorDevice(grad_output_p, &device);
  diopiDtype_t dtype;
  diopiGetTensorDtype(grad_output_p, &dtype);
  if (device == diopi_host || dtype == diopi_dtype_float16) {
    border_align_backward_impl(grad_output, boxes, argmax_idx, grad_input,
                               pool_size);
    return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto boxes_p = toDiopiTensorHandle(boxes);
  auto argmax_idx_p = toDiopiTensorHandle(argmax_idx);
  auto grad_input_p = toDiopiTensorHandle(grad_input);
  if (&diopiBorderAlignBackwardMmcv) {
    diopiBorderAlignBackwardMmcv(ch,grad_input_p, grad_output_p, boxes_p, argmax_idx_p,
                             pool_size);
  } else {
    border_align_backward_impl(grad_output, boxes, argmax_idx, grad_input,
                               pool_size);
  }
#else
  border_align_backward_impl(grad_output, boxes, argmax_idx, grad_input,
                             pool_size);
#endif
}
