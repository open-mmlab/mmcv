#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;

void sigmoid_focal_loss_forward_npu(Tensor input, Tensor target, Tensor weight,
                                    Tensor output, float gamma, float alpha) {
  at::Tensor target_y = at::reshape(target, input.sizes());
  target_y =
      at_npu::native::NPUNativeFunctions::npu_dtype_cast(target_y, at::kInt);
  int64_t weight_size = weight.size(0);
  at::Tensor weight_y = at::ones_like(input);
  if (weight_size > 0) {
    weight_y = at_npu::native::NPUNativeFunctions::npu_broadcast(weight,
                                                                 input.sizes());
  }
  OpCommand cmd;
  cmd.Name("SigmoidFocalLoss")
      .Input(input)
      .Input(target_y)
      .Input(weight_y)
      .Output(output)
      .Attr("gamma", gamma)
      .Attr("alpha", alpha)
      .Attr("reduction", "none")
      .Run();
}

void sigmoid_focal_loss_forward_impl(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha);

void sigmoid_focal_loss_backward_npu(Tensor input, Tensor target, Tensor weight,
                                     Tensor grad_input, float gamma,
                                     float alpha) {
  at::Tensor target_y = at::reshape(target, input.sizes());
  target_y =
      at_npu::native::NPUNativeFunctions::npu_dtype_cast(target_y, at::kInt);
  at::Tensor grad_up = at::ones_like(input);
  int64_t weight_size = weight.size(0);
  at::Tensor weight_y = at::ones_like(input);
  if (weight_size > 0) {
    weight_y = at_npu::native::NPUNativeFunctions::npu_broadcast(weight,
                                                                 input.sizes());
  }
  OpCommand cmd;
  cmd.Name("SigmoidFocalLossGrad")
      .Input(input)
      .Input(target_y)
      .Input(grad_up)
      .Input(weight_y)
      .Output(grad_input)
      .Attr("gamma", gamma)
      .Attr("alpha", alpha)
      .Attr("reduction", "none")
      .Run();
}

void sigmoid_focal_loss_backward_impl(Tensor input, Tensor target,
                                      Tensor weight, Tensor grad_input,
                                      float gamma, float alpha);

void softmax_focal_loss_forward_npu(Tensor input, Tensor target, Tensor weight,
                                    Tensor output, float gamma, float alpha) {
  int64_t n_class = input.size(1);
  at::Tensor target_y =
      at_npu::native::NPUNativeFunctions::one_hot(target, n_class);
  target_y =
      at_npu::native::NPUNativeFunctions::npu_dtype_cast(target_y, at::kInt);
  int64_t weight_size = weight.size(0);
  at::Tensor weight_y = at::ones_like(input);
  if (weight_size > 0) {
    weight_y = at_npu::native::NPUNativeFunctions::npu_broadcast(weight,
                                                                 input.sizes());
  }
  OpCommand cmd;
  cmd.Name("SoftmaxFocalLoss")
      .Input(input)
      .Input(target_y)
      .Input(weight_y)
      .Output(output)
      .Attr("gamma", gamma)
      .Attr("alpha", alpha)
      .Attr("reduction", "none")
      .Run();
}

void softmax_focal_loss_forward_impl(Tensor input, Tensor target, Tensor weight,
                                     Tensor grad_input, float gamma,
                                     float alpha);

void softmax_focal_loss_backward_npu(Tensor input, Tensor target, Tensor weight,
                                     Tensor buff, Tensor grad_input,
                                     float gamma, float alpha) {
  int64_t n_class = input.size(1);
  at::Tensor target_y =
      at_npu::native::NPUNativeFunctions::one_hot(target, n_class);
  target_y =
      at_npu::native::NPUNativeFunctions::npu_dtype_cast(target_y, at::kInt);
  at::Tensor grad_up = at::ones_like(input);
  int64_t weight_size = weight.size(0);
  at::Tensor weight_y = at::ones_like(input);
  if (weight_size > 0) {
    weight_y = at_npu::native::NPUNativeFunctions::npu_broadcast(weight,
                                                                 input.sizes());
  }

  OpCommand cmd;
  cmd.Name("SoftmaxFocalLossGrad")
      .Input(input)
      .Input(target_y)
      .Input(grad_up)
      .Input(weight_y)
      .Output(grad_input)
      .Attr("gamma", gamma)
      .Attr("alpha", alpha)
      .Attr("reduction", "none")
      .Run();
}

void softmax_focal_loss_backward_impl(Tensor input, Tensor target,
                                      Tensor weight, Tensor buff,
                                      Tensor grad_input, float gamma,
                                      float alpha);

REGISTER_NPU_IMPL(sigmoid_focal_loss_forward_impl,
                  sigmoid_focal_loss_forward_npu);

REGISTER_NPU_IMPL(sigmoid_focal_loss_backward_impl,
                  sigmoid_focal_loss_backward_npu);

REGISTER_NPU_IMPL(softmax_focal_loss_forward_impl,
                  softmax_focal_loss_forward_npu);

REGISTER_NPU_IMPL(softmax_focal_loss_backward_impl,
                  softmax_focal_loss_backward_npu);
