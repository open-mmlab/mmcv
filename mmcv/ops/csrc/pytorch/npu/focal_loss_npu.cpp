#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;

<<<<<<< HEAD
=======

>>>>>>> 08f0a16 (clean code)
void sigmoid_focal_loss_forward_npu(Tensor input, Tensor target, Tensor weight,
                                    Tensor output, float gamma, float alpha) {
<<<<<<< HEAD
  int64_t n_class = input.size(1);
  at::Tensor target_y = at::ones_like(input);
  if (n_class == 1) {
    target_y = at::reshape(target, input.sizes());
    target_y = at::mul(target_y, -1.0);
    target_y = at::add(target_y, 1.0);
  } else {
    target_y = at_npu::native::NPUNativeFunctions::one_hot(target, n_class);
  }
  target_y =
      at_npu::native::NPUNativeFunctions::npu_dtype_cast(target_y, at::kInt);
  int64_t weight_size = weight.size(0);
  at::Tensor weight_y = at::ones_like(input);
  if (weight_size > 0) {
    weight_y = at_npu::native::NPUNativeFunctions::npu_broadcast(weight,
                                                                 input.sizes());
  }
  OpCommand cmd;
  string reduction = "none";
  cmd.Name("SigmoidFocalLoss")
      .Input(input)
      .Input(target_y)
      .Input(weight_y)
      .Output(output)
      .Attr("gamma", gamma)
      .Attr("alpha", alpha)
      .Attr("reduction", reduction)
=======
  at::Tensor target_y = at::reshape(target, input.sizes());
  target_y = at_npu::native::NPUNativeFunctions::npu_dtype_cast(
                                                 target_y, at::kInt);
  at::Tensor grad_up = at::ones_like(input);
  int64_t weight_size = weight.size(0);
  at::Tensor weight_y = at::ones_like(input);
  if(weight_size > 0) {
    weight_y = at_npu::native::NPUNativeFunctions::npu_broadcast(
                                                   weight, input.sizes());
  }
  OpCommand cmd;
  cmd.Name("SigmoidFocalLoss")
      .Input(input)
      .Input(target_y)
      .Input(grad_up)
      .Input(weight_y)
      .Output(grad_input)
      .Attr("gamma", gamma)
      .Attr("alpha", alpha)
      .Attr("reduction", "none")
>>>>>>> 26f35e0 (code format)
      .Run();
}

void sigmoid_focal_loss_forward_impl(Tensor input, Tensor target,
                                     Tensor weight, Tensor output,
                                     float gamma, float alpha);

<<<<<<< HEAD
void sigmoid_focal_loss_backward_npu(Tensor input, Tensor target, Tensor weight,
<<<<<<< HEAD
                                     Tensor grad_input, float gamma,
                                     float alpha) {
  int64_t n_class = input.size(1);
  at::Tensor target_y = at::ones_like(input);
  if (n_class == 1) {
    target_y = at::reshape(target, input.sizes());
  } else {
    target_y = at_npu::native::NPUNativeFunctions::one_hot(target, n_class);
    target_y = at::mul(target_y, -1.0);
    target_y = at::add(target_y, 1.0);
  }
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
  string reduction = "none";
=======
void sigmoid_focal_loss_backward_npu(Tensor input, Tensor target,
                                     Tensor weight, Tensor grad_input,
                                     float gamma, float alpha) {
  at::Tensor target_y = at::reshape(target, input.sizes());
  target_y = at_npu::native::NPUNativeFunctions::npu_dtype_cast(
                                                 target_y, at::kInt);
  at::Tensor grad_up = at::ones_like(input);
  int64_t weight_size = weight.size(0);
  at::Tensor weight_y = at::ones_like(input);
  if(weight_size > 0) {
      weight_y = at_npu::native::NPUNativeFunctions::npu_broadcast(
                                                     weight, input.sizes());
  }
  OpCommand cmd;
>>>>>>> 26f35e0 (code format)
  cmd.Name("SigmoidFocalLossGrad")
      .Input(input)
      .Input(target_y)
      .Input(grad_up)
      .Input(weight_y)
      .Output(grad_input)
      .Attr("gamma", gamma)
      .Attr("alpha", alpha)
<<<<<<< HEAD
      .Attr("reduction", reduction)
=======
      .Attr("reduction", "none")
>>>>>>> 26f35e0 (code format)
      .Run();
}

void sigmoid_focal_loss_backward_impl(Tensor input, Tensor target,
                                      Tensor weight, Tensor grad_input,
                                      float gamma, float alpha);

<<<<<<< HEAD
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
  at::Tensor op_output = at::ones_like(input);
  OpCommand cmd;
  string reduction = "none";
  cmd.Name("SoftmaxFocalLoss")
      .Input(input)
      .Input(target_y)
      .Input(weight_y)
      .Output(op_output)
      .Attr("gamma", gamma)
      .Attr("alpha", alpha)
      .Attr("reduction", reduction)
      .Run();
  int64_t n_batch = input.size(0);
  c10::SmallVector<int64_t, 2> offsets = {0, 0};
  c10::SmallVector<int64_t, 2> sizes = {n_batch, 1};
  at::IntArrayRef offset = at::IntArrayRef(offsets);
  at::IntArrayRef size = at::IntArrayRef(sizes);
  at_npu::native::NPUNativeFunctions::npu_slice_out(op_output, offset, size,
                                                    output);
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
  string reduction = "none";
=======
void softmax_focal_loss_forward_npu(Tensor input, Tensor target,
                                    Tensor weight, Tensor output,
                                    float gamma, float alpha) {
  int64_t n_class = input.size(1);
  at::Tensor target_y = at_npu::native::NPUNativeFunctions::one_hot(
                                                            target, n_class);
  target_y = at_npu::native::NPUNativeFunctions::npu_dtype_cast(
                                                 target_y, at::kInt);
  at::Tensor grad_up = at::ones_like(input);
  int64_t weight_size = weight.size(0);
  at::Tensor weight_y = at::ones_like(input);
  if(weight_size > 0) {
      weight_y = at_npu::native::NPUNativeFunctions::npu_broadcast(
                                                     weight, input.sizes());
  }
  OpCommand cmd;
  cmd.Name("SoftmaxFocalLoss")
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

void softmax_focal_loss_forward_impl(Tensor input, Tensor target,
                                     Tensor weight, Tensor grad_input,
                                     float gamma, float alpha);

void softmax_focal_loss_backward_npu(Tensor input, Tensor target,
                                     Tensor weight, Tensor buff,
                                     Tensor grad_input, float gamma,
                                     float alpha) {
  int64_t n_class = input.size(1);
  at::Tensor target_y = at_npu::native::NPUNativeFunctions::one_hot(
                                                            target, n_class);
  target_y = at_npu::native::NPUNativeFunctions::npu_dtype_cast(
                                               target_y, at::kInt);
  at::Tensor grad_up = at::ones_like(input);
  int64_t weight_size = weight.size(0);
  at::Tensor weight_y = at::ones_like(input);
  if(weight_size > 0) {
      weight_y = at_npu::native::NPUNativeFunctions::npu_broadcast(
                                                   weight, input.sizes());
  }

  OpCommand cmd;
>>>>>>> 26f35e0 (code format)
  cmd.Name("SoftmaxFocalLossGrad")
      .Input(input)
      .Input(target_y)
      .Input(grad_up)
      .Input(weight_y)
      .Output(grad_input)
      .Attr("gamma", gamma)
      .Attr("alpha", alpha)
<<<<<<< HEAD
      .Attr("reduction", reduction)
=======
      .Attr("reduction", "none")
>>>>>>> 26f35e0 (code format)
      .Run();
=======
                                     Tensor grad_input, float gamma, float alpha) {

    at::Tensor target_y = at::reshape(target, input.sizes());
    target_y = at_npu::native::NPUNativeFunctions::npu_dtype_cast(target_y, at::kInt);
    at::Tensor grad_up = at::ones_like(input);
    int64_t weight_size = weight.size(0);
    at::Tensor weight_y = at::ones_like(input);
    if(weight_size > 0) {
        weight_y = at_npu::native::NPUNativeFunctions::npu_broadcast(weight, input.sizes());
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

void sigmoid_focal_loss_backward_impl(Tensor input, Tensor target, Tensor weight,
                                      Tensor grad_input, float gamma, float alpha);

void softmax_focal_loss_forward_npu(Tensor input, Tensor target, Tensor weight,
                                    Tensor output, float gamma, float alpha) {

    int64_t n_class = input.size(1);
    at::Tensor target_y = at_npu::native::NPUNativeFunctions::one_hot(target, n_class);
    target_y = at_npu::native::NPUNativeFunctions::npu_dtype_cast(target_y, at::kInt);
    at::Tensor grad_up = at::ones_like(input);
    int64_t weight_size = weight.size(0);
    at::Tensor weight_y = at::ones_like(input);
    if(weight_size > 0) {
        weight_y = at_npu::native::NPUNativeFunctions::npu_broadcast(weight, input.sizes());
    }

    OpCommand cmd;
    cmd.Name("SoftmaxFocalLoss")
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

void softmax_focal_loss_forward_impl(Tensor input, Tensor target, Tensor weight,
                                     Tensor grad_input, float gamma, float alpha);

void softmax_focal_loss_backward_npu(Tensor input, Tensor target, Tensor weight, Tensor buff,
                                     Tensor grad_input, float gamma, float alpha) {

    int64_t n_class = input.size(1);
    at::Tensor target_y = at_npu::native::NPUNativeFunctions::one_hot(target, n_class);
    target_y = at_npu::native::NPUNativeFunctions::npu_dtype_cast(target_y, at::kInt);
    at::Tensor grad_up = at::ones_like(input);
    int64_t weight_size = weight.size(0);
    at::Tensor weight_y = at::ones_like(input);
    if(weight_size > 0) {
        weight_y = at_npu::native::NPUNativeFunctions::npu_broadcast(weight, input.sizes());
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
>>>>>>> 08f0a16 (clean code)
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
<<<<<<< HEAD
                  softmax_focal_loss_backward_npu);
=======
                  softmax_focal_loss_backward_npu);
>>>>>>> 26f35e0 (code format)
