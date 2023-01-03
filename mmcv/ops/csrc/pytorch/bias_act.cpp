#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

torch::Tensor bias_act_op_impl(const torch::Tensor &input,
                               const torch::Tensor &bias,
                               const torch::Tensor &xref,
                               const torch::Tensor &yref,
                               const torch::Tensor &dy, int grad, int dim,
                               int act, float alpha, float gain, float clamp) {
  return DISPATCH_DEVICE_IMPL(bias_act_op_impl, input, bias, xref, yref, dy,
                              grad, dim, act, alpha, gain, clamp);
}

torch::Tensor bias_act(const torch::Tensor &input, const torch::Tensor &bias,
                       const torch::Tensor &xref, const torch::Tensor &yref,
                       const torch::Tensor &dy, int grad, int dim, int act,
                       float alpha, float gain, float clamp) {
  return bias_act_op_impl(input, bias, xref, yref, dy, grad, dim, act, alpha,
                          gain, clamp);
}
