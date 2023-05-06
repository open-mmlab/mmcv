// Copyright (c) OpenMMLab. All rights reserved
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include "diopi.hpp"
#endif

void sigmoid_focal_loss_forward_impl(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha) {
  DISPATCH_DEVICE_IMPL(sigmoid_focal_loss_forward_impl, input, target, weight,
                       output, gamma, alpha);
}

void sigmoid_focal_loss_backward_impl(Tensor input, Tensor target,
                                      Tensor weight, Tensor grad_input,
                                      float gamma, float alpha) {
  DISPATCH_DEVICE_IMPL(sigmoid_focal_loss_backward_impl, input, target, weight,
                       grad_input, gamma, alpha);
}

void softmax_focal_loss_forward_impl(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha) {
  DISPATCH_DEVICE_IMPL(softmax_focal_loss_forward_impl, input, target, weight,
                       output, gamma, alpha);
}

void softmax_focal_loss_backward_impl(Tensor input, Tensor target,
                                      Tensor weight, Tensor buff,
                                      Tensor grad_input, float gamma,
                                      float alpha) {
  DISPATCH_DEVICE_IMPL(softmax_focal_loss_backward_impl, input, target, weight,
                       buff, grad_input, gamma, alpha);
}

void sigmoid_focal_loss_forward(Tensor input, Tensor target, Tensor weight,
                                Tensor output, float gamma, float alpha) {
#ifdef MMCV_WITH_DIOPI
  auto input_p = toDiopiTensorHandle(input);
  diopiDevice_t device;
  diopiGetTensorDevice(input_p, &device);
  diopiDtype_t dtype;
  diopiGetTensorDtype(input_p, &dtype);
  if (device == diopi_host|| dtype == diopi_dtype_float16) {
    sigmoid_focal_loss_forward_impl(input, target, weight, output, gamma, alpha);
    return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto target_p = toDiopiTensorHandle(target);
  auto weight_p = toDiopiTensorHandle(weight);
  auto output_p = toDiopiTensorHandle(output);
  if(&diopiSigmoidFocalLossMmcv){
    diopiSigmoidFocalLossMmcv(ch, input_p, target_p, weight_p, output_p, gamma, alpha);
  }else{
    sigmoid_focal_loss_forward_impl(input, target, weight, output, gamma, alpha);
  }
#else
  sigmoid_focal_loss_forward_impl(input, target, weight, output, gamma, alpha);
#endif
}

void sigmoid_focal_loss_backward(Tensor input, Tensor target, Tensor weight,
                                 Tensor grad_input, float gamma, float alpha) {
#ifdef MMCV_WITH_DIOPI
  auto input_p = toDiopiTensorHandle(input);
  diopiDevice_t device;
  diopiGetTensorDevice(input_p, &device);
  diopiDtype_t dtype;
  diopiGetTensorDtype(input_p, &dtype);
  if (device == diopi_host|| dtype == diopi_dtype_float16) {
    sigmoid_focal_loss_backward_impl(input, target, weight, grad_input, gamma,alpha);
    return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto target_p = toDiopiTensorHandle(target);
  auto weight_p = toDiopiTensorHandle(weight);
  auto grad_input_p = toDiopiTensorHandle(grad_input);
  if(&diopiSigmoidFocalLossBackwardMmcv){
    diopiSigmoidFocalLossBackwardMmcv(ch, input_p, target_p,weight_p,  grad_input_p, gamma, alpha);
  }else{
    sigmoid_focal_loss_backward_impl(input, target, weight, grad_input, gamma, alpha);
  }
#else
  sigmoid_focal_loss_backward_impl(input, target, weight, grad_input, gamma,
                                   alpha);
#endif
}

void softmax_focal_loss_forward(Tensor input, Tensor target, Tensor weight,
                                Tensor output, float gamma, float alpha) {
  softmax_focal_loss_forward_impl(input, target, weight, output, gamma, alpha);
}

void softmax_focal_loss_backward(Tensor input, Tensor target, Tensor weight,
                                 Tensor buff, Tensor grad_input, float gamma,
                                 float alpha) {
  softmax_focal_loss_backward_impl(input, target, weight, buff, grad_input,
                                   gamma, alpha);
}
