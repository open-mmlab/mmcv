// Copyright(c) OpenMMLab.All rights reserved.
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

Tensor nattenav_forward_impl(const Tensor attn, const Tensor value) {
  return DISPATCH_DEVICE_IMPL(nattenav_forward_impl, attn, value);
}

Tensor nattenav_forward(const Tensor attn, const Tensor value) {
  return nattenav_forward_impl(attn, value);
}

std::vector<Tensor> nattenav_backward_impl(const Tensor grad_output,
                                           const Tensor attn,
                                           const Tensor value) {
  return DISPATCH_DEVICE_IMPL(nattenav_backward_impl, grad_output, attn, value);
}

std::vector<Tensor> nattenav_backward(const Tensor grad_output,
                                      const Tensor attn, const Tensor value) {
  return nattenav_backward_impl(grad_output, attn, value);
}
