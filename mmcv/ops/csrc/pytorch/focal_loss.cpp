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
  if (device == diopi_host) {
    sigmoid_focal_loss_forward_impl(input, target, weight, output, gamma,
                                    alpha);
    return;
  }
  diopiContext ctx(dipu::getCurrentDIPUStream().rawstream());
  diopiContextHandle_t ch = &ctx;
  auto target_p = toDiopiTensorHandle(target);
  auto weight_p = toDiopiTensorHandle(weight);
  auto output_p = toDiopiTensorHandle(output);
  if (reinterpret_cast<void *>(diopiSigmoidFocalLossMmcv) != nullptr) {
    auto ret = diopiSigmoidFocalLossMmcv(ch, output_p, input_p, target_p,
                                         weight_p, gamma, alpha);
    if (ret == diopiSuccess) return;
  }
  LOG(WARNING)
      << "Fallback to cpu: mmcv ext op sigmoid_focal_loss_forward_impl";
  auto input_cpu = input.cpu();
  auto target_cpu = target.cpu();
  auto weight_cpu = weight.cpu();
  auto output_cpu = output.cpu();
  sigmoid_focal_loss_forward_impl(input_cpu, target_cpu, weight_cpu, output_cpu,
                                  gamma, alpha);
  output.copy_(output_cpu);
  return;
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
  if (device == diopi_host) {
    sigmoid_focal_loss_backward_impl(input, target, weight, grad_input, gamma,
                                     alpha);
    return;
  }
  diopiContext ctx(dipu::getCurrentDIPUStream().rawstream());
  diopiContextHandle_t ch = &ctx;
  auto target_p = toDiopiTensorHandle(target);
  auto weight_p = toDiopiTensorHandle(weight);
  auto grad_input_p = toDiopiTensorHandle(grad_input);
  if (reinterpret_cast<void *>(diopiSigmoidFocalLossBackwardMmcv) != nullptr) {
    auto ret = diopiSigmoidFocalLossBackwardMmcv(
        ch, grad_input_p, input_p, target_p, weight_p, gamma, alpha);
    if (ret == diopiSuccess) return;
  }
  LOG(WARNING)
      << "Fallback to cpu: mmcv ext op sigmoid_focal_loss_forward_impl";
  auto input_cpu = input.cpu();
  auto target_cpu = target.cpu();
  auto weight_cpu = weight.cpu();
  auto grad_input_cpu = grad_input.cpu();
  sigmoid_focal_loss_backward_impl(input_cpu, target_cpu, weight_cpu,
                                   grad_input_cpu, gamma, alpha);
  grad_input.copy_(grad_input_cpu);
  return;
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
