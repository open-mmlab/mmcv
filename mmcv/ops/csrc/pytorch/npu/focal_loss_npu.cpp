#include "pytorch_npu_helper.hpp"
using namespace NPU_NAME_SPACE;
using namespace std;

void sigmoid_focal_loss_forward_npu(Tensor input, Tensor target, Tensor weight,
                                    Tensor output, float gamma, float alpha) {
  at::Tensor input_y = input;
  at::Tensor output_y = output;
  bool is_half = input.scalar_type() == at::kHalf;
  if (is_half) {
    input_y = input.to(at::kFloat);
    output_y = output.to(at::kFloat);
  }
  int64_t weight_size = weight.size(0);
  at::Tensor weight_y = at::ones_like(input_y);
  if (weight_size > 0) {
    weight_y = at::broadcast_to(weight, input.sizes());
    if (is_half) {
      weight_y = weight_y.to(at::kFloat);
    }
  }
  int64_t n_class = input.size(1);
  at::Tensor target_y = at::ones_like(input);
  if (n_class == 1) {
    target_y = at::reshape(target, input.sizes());
    target_y = at::mul(target_y, -1.0);
    target_y = at::add(target_y, 1.0);
  } else {
    target_y = at::one_hot(target, n_class);
    weight_y = at::mul(weight_y, target_y);
    weight_y = at::sum(weight_y, 1, true);
    weight_y = at::broadcast_to(weight_y, input.sizes());
  }
  target_y = target_y.to(at::kInt);
  OpCommand cmd;
  string reduction = "none";
  cmd.Name("SigmoidFocalLoss")
      .Input(input_y)
      .Input(target_y)
      .Input(weight_y)
      .Output(output_y)
      .Attr("gamma", gamma)
      .Attr("alpha", alpha)
      .Attr("reduction", reduction)
      .Run();
  if (is_half) {
    output_y = output_y.to(at::kHalf);
  }
  output.copy_(output_y);
}

void sigmoid_focal_loss_forward_impl(Tensor input, Tensor target, Tensor weight,
                                     Tensor output, float gamma, float alpha);

void sigmoid_focal_loss_backward_npu(Tensor input, Tensor target, Tensor weight,
                                     Tensor grad_input, float gamma,
                                     float alpha) {
  at::Tensor input_y = input;
  at::Tensor grad_input_y = grad_input;
  bool is_half = input.scalar_type() == at::kHalf;
  if (is_half) {
    input_y = input.to(at::kFloat);
    grad_input_y = grad_input.to(at::kFloat);
  }
  int64_t weight_size = weight.size(0);
  at::Tensor weight_y = at::ones_like(input_y);
  if (weight_size > 0) {
    weight_y = at::broadcast_to(weight, input.sizes());
    if (is_half) {
      weight_y = weight_y.to(at::kFloat);
    }
  }
  int64_t n_class = input.size(1);
  at::Tensor target_y = at::ones_like(input);
  if (n_class == 1) {
    target_y = at::reshape(target, input.sizes());
  } else {
    target_y = at::one_hot(target, n_class);
    weight_y = at::mul(weight_y, target_y);
    weight_y = at::sum(weight_y, 1, true);
    weight_y = at::broadcast_to(weight_y, input.sizes());
    target_y = at::mul(target_y, -1.0);
    target_y = at::add(target_y, 1.0);
  }
  target_y = target_y.to(at::kInt);
  at::Tensor grad_up = at::ones_like(input);
  OpCommand cmd;
  string reduction = "none";
  cmd.Name("SigmoidFocalLossGrad")
      .Input(input_y)
      .Input(target_y)
      .Input(grad_up)
      .Input(weight_y)
      .Output(grad_input_y)
      .Attr("gamma", gamma)
      .Attr("alpha", alpha)
      .Attr("reduction", reduction)
      .Run();
  if (is_half) {
    grad_input_y = grad_input_y.to(at::kHalf);
  }
  grad_input.copy_(grad_input_y);
}

void sigmoid_focal_loss_backward_impl(Tensor input, Tensor target,
                                      Tensor weight, Tensor grad_input,
                                      float gamma, float alpha);

void softmax_focal_loss_forward_npu(Tensor input, Tensor target, Tensor weight,
                                    Tensor output, float gamma, float alpha) {
  at::Tensor input_y = input;
  bool is_half = input.scalar_type() == at::kHalf;
  if (is_half) {
    input_y = input.to(at::kFloat);
  }
  int64_t n_class = input.size(1);
  at::Tensor target_y = at::one_hot(target, n_class);
  target_y = target_y.to(at::kInt);
  int64_t weight_size = weight.size(0);
  at::Tensor weight_y = at::ones_like(input_y);
  if (weight_size > 0) {
    weight_y = at::broadcast_to(weight, input.sizes());
    if (is_half) {
      weight_y = weight_y.to(at::kFloat);
    }
    weight_y = at::mul(weight_y, target_y);
    weight_y = at::sum(weight_y, 1, true);
    weight_y = at::broadcast_to(weight_y, input.sizes());
  }
  at::Tensor op_output = at::ones_like(input_y);
  OpCommand cmd;
  string reduction = "none";
  cmd.Name("SoftmaxFocalLoss")
      .Input(input_y)
      .Input(target_y)
      .Input(weight_y)
      .Output(op_output)
      .Attr("gamma", gamma)
      .Attr("alpha", alpha)
      .Attr("reduction", reduction)
      .Run();
  if (is_half) {
    op_output = op_output.to(at::kHalf);
  }
  int64_t n_batch = input.size(0);
  c10::SmallVector<int64_t, 2> offsets = {0, 0};
  c10::SmallVector<int64_t, 2> sizes = {n_batch, 1};
  at::IntArrayRef offset = at::IntArrayRef(offsets);
  at::IntArrayRef size = at::IntArrayRef(sizes);
  at::IntArrayRef size_array = at::IntArrayRef(sizes);
  c10::SmallVector<int64_t, 8> offsetVec;
  for (uint64_t i = 0; i < offset.size(); i++) {
    offsetVec.emplace_back(offset[i]);
  }
  c10::SmallVector<int64_t, 8> sizeVec;
  for (uint64_t i = 0; i < size_array.size(); i++) {
    sizeVec.emplace_back(size_array[i]);
  }
  OpCommand cmd2;
  cmd2.Name("Slice")
      .Input(op_output)
      .Input(offsetVec)
      .Input(sizeVec)
      .Output(output)
      .Run();
}

void softmax_focal_loss_forward_impl(Tensor input, Tensor target, Tensor weight,
                                     Tensor grad_input, float gamma,
                                     float alpha);

void softmax_focal_loss_backward_npu(Tensor input, Tensor target, Tensor weight,
                                     Tensor buff, Tensor grad_input,
                                     float gamma, float alpha) {
  at::Tensor input_y = input;
  at::Tensor grad_input_y = grad_input;
  bool is_half = input.scalar_type() == at::kHalf;
  if (is_half) {
    input_y = input.to(at::kFloat);
    grad_input_y = grad_input.to(at::kFloat);
  }
  int64_t n_class = input.size(1);
  at::Tensor target_y = at::one_hot(target, n_class);
  target_y = target_y.to(at::kInt);
  at::Tensor grad_up = at::ones_like(input);
  int64_t weight_size = weight.size(0);
  at::Tensor weight_y = at::ones_like(input_y);
  if (weight_size > 0) {
    weight_y = at::broadcast_to(weight, input.sizes());
    if (is_half) {
      weight_y = weight_y.to(at::kFloat);
    }
    weight_y = at::mul(weight_y, target_y);
    weight_y = at::sum(weight_y, 1, true);
    weight_y = at::broadcast_to(weight_y, input.sizes());
  }
  grad_input_y = grad_input_y.fill_(0);
  OpCommand cmd;
  string reduction = "none";
  cmd.Name("SoftmaxFocalLossGrad")
      .Input(input_y)
      .Input(target_y)
      .Input(grad_up)
      .Input(weight_y)
      .Output(grad_input_y)
      .Attr("gamma", gamma)
      .Attr("alpha", alpha)
      .Attr("reduction", reduction)
      .Run();
  if (is_half) {
    grad_input_y = grad_input_y.to(at::kHalf);
  }
  grad_input.copy_(grad_input_y);
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