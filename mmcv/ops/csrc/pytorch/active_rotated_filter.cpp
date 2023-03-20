// Copyright (c) OpenMMLab. All rights reserved.
// Modified from
// https://github.com/csuhan/s2anet/blob/master/mmdet/ops/orn/src/ActiveRotatingFilter.h

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>
#include "diopi.hpp"
#endif

void active_rotated_filter_forward_impl(const Tensor input,
                                        const Tensor indices, Tensor output) {
  DISPATCH_DEVICE_IMPL(active_rotated_filter_forward_impl, input, indices,
                       output);
}

void active_rotated_filter_backward_impl(const Tensor grad_out,
                                         const Tensor indices, Tensor grad_in) {
  DISPATCH_DEVICE_IMPL(active_rotated_filter_backward_impl, grad_out, indices,
                       grad_in);
}

void active_rotated_filter_forward(const Tensor input, const Tensor indices,
                                   Tensor output) {
#ifdef MMCV_WITH_DIOPI
  auto input_p = toDiopiTensorHandle(&input);
  diopiDevice_t device;
  diopiGetTensorDevice(input_p, &device);
  if (device == diopi_host) {
      active_rotated_filter_forward_impl(input, indices, output);
      return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto indices_p = toDiopiTensorHandle(&indices);
  auto out_p = toDiopiTensorHandle(&output);
  if (&diopiActiveRotatedFilter) {
   diopiActiveRotatedFilter(ch, input_p, indices_p, out_p);
  } else {
   active_rotated_filter_forward_impl(input, indices, output);
  }
#else
  active_rotated_filter_forward_impl(input, indices, output);
#endif
}

void active_rotated_filter_backward(const Tensor grad_out, const Tensor indices,
                                    Tensor grad_in) {
#ifdef MMCV_WITH_DIOPI
  auto grad_out_p = toDiopiTensorHandle(&grad_out);
  diopiDevice_t device;
  diopiGetTensorDevice(grad_out_p, &device);
  if (device == diopi_host) {
      active_rotated_filter_backward_impl(grad_out, indices, grad_in);
      return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto indices_p = toDiopiTensorHandle(&indices);
  auto grad_in_p = toDiopiTensorHandle(&grad_in);
  if (&diopiActiveRotatedFilterBackward) {
   diopiActiveRotatedFilterBackward(ch, grad_out_p, indices_p, grad_in_p);
  } else {
   active_rotated_filter_backward_impl(grad_out, indices, grad_in);
  }
#else
  active_rotated_filter_backward_impl(grad_out, indices, grad_in);
#endif
}
