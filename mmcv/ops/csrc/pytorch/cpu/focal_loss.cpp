#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

void sigmoid_focal_loss_forward_cpu(Tensor input, Tensor target,
                                               Tensor weight, Tensor output,
                                               const float gamma,
                                               const float alpha)
{
  std::cout << "please implement sigmoid_focal_loss_forward_cpu" << std::endl;
}

void sigmoid_focal_loss_backward_cpu(Tensor input, Tensor target,
                                      Tensor weight, Tensor grad_input,
                                      float gamma, float alpha)
{
  std::cout << "please implement sigmoid_focal_loss_backward_cpu" << std::endl;
}


void softmax_focal_loss_forward_cpu(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha)
{
  std::cout << "please implement  softmax_focal_loss_forward_cpu" << std::endl;
}

void softmax_focal_loss_backward_cpu(Tensor input, Tensor target,
                                      Tensor weight, Tensor buff,
                                      Tensor grad_input, float gamma,
                                      float alpha)
{
  std::cout << "please implement softmax_focal_loss_backward_cpu" << std::endl;
}


void sigmoid_focal_loss_forward_impl(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha);
void sigmoid_focal_loss_backward_impl(Tensor input, Tensor target,
                                      Tensor weight, Tensor grad_input,
                                      float gamma, float alpha);
void softmax_focal_loss_forward_impl(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha);
void softmax_focal_loss_backward_impl(Tensor input, Tensor target,
                                      Tensor weight, Tensor buff,
                                      Tensor grad_input, float gamma,
                                      float alpha);

REGISTER_DEVICE_IMPL(softmax_focal_loss_forward_impl, CPU,
                     softmax_focal_loss_forward_cpu);

REGISTER_DEVICE_IMPL(softmax_focal_loss_backward_impl, CPU,
                     softmax_focal_loss_backward_cpu);

REGISTER_DEVICE_IMPL(sigmoid_focal_loss_forward_impl, CPU,
                     sigmoid_focal_loss_forward_cpu);

REGISTER_DEVICE_IMPL(sigmoid_focal_loss_backward_impl, CPU,
                     sigmoid_focal_loss_backward_cpu);
