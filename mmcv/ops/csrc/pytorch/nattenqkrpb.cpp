// Copyright(c) OpenMMLab.All rights reserved.
#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

Tensor nattenqkrpb_forward_impl(const Tensor query, const Tensor key,
                                const Tensor rpb) {
  return DISPATCH_DEVICE_IMPL(nattenqkrpb_forward_impl, query, key, rpb);
}

Tensor nattenqkrpb_forward(const Tensor query, const Tensor key,
                           const Tensor rpb) {
  return nattenqkrpb_forward_impl(query, key, rpb);
}

std::vector<Tensor> nattenqkrpb_backward_impl(const Tensor grad_output,
                                              const Tensor query,
                                              const Tensor key) {
  return DISPATCH_DEVICE_IMPL(nattenqkrpb_backward_impl, grad_output, query,
                              key);
}

std::vector<Tensor> nattenqkrpb_backward(const Tensor grad_output,
                                         const Tensor query, const Tensor key) {
  return nattenqkrpb_backward_impl(grad_output, query, key);
}
