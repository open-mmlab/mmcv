#include <iostream>
#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;


void sigmoid_focal_loss_forward_npu(Tensor input, Tensor target, Tensor weight, 
                                    Tensor output, float gamma, float alpha) {
    at::Scalar s = 1;
    OpPreparation::CheckOut(
        {input, target, weight},
        output,
        input);

    at::Tensor target_y = at_npu::native::NPUNativeFunctions::npu_reshape(target, input.sizes(), true);
    OpCommand cmd;
    cmd.Name("AxpyV2")
        .Input(input)
        .Input(target_y)
        .Input(s, input.scalar_type())
        .Output(output)
        .Run();
}

void sigmoid_focal_loss_forward_impl(Tensor input, Tensor target, Tensor weight, 
                                     Tensor output, float gamma, float alpha);

void sigmoid_focal_loss_backward_npu(Tensor input, Tensor target, Tensor weight, 
                                     Tensor grad_input, float gamma, float alpha) {
    OpPreparation::CheckOut(
        {input, target, weight},
        grad_input,
        input);

    at::Tensor target_y = at_npu::native::NPUNativeFunctions::npu_reshape(target, input.sizes(), true);
    at::Tensor dout = at::ones_like(input);
    int64_t weight_size = weight.size(0);
    at::Tensor weight_y = at::ones_like(input);
    if(weight_size > 0) {
        weight_y = aat_npu::native::NPUNativeFunctions::npu_broadcast(weight, input.sizes());
    }

    OpCommand cmd;
    cmd.Name("SigmoidFocalLossGrad")
        .Input(input)
        .Input(target_y)
        .Input(dout)
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
    at::Scalar s = 1;
    OpPreparation::CheckOut(
        {input, target, weight},
        output,
        input);

    int64_t  n_class = input.size(1);
    at::Tensor target_y = at_npu::native::NPUNativeFunctions::one_hot(target, n_class);
    target_y = at_npu::native::NPUNativeFunctions::npu_dtype_cast(target_y, input.scalar_type());

    OpCommand cmd;
    cmd.Name("AxpyV2")
        .Input(input)
        .Input(target_y)
        .Input(s, input.scalar_type())
        .Output(output)
        .Run();
}

void softmax_focal_loss_forward_impl(Tensor input, Tensor target, Tensor weight, 
                                     Tensor grad_input, float gamma, float alpha);

void softmax_focal_loss_backward_npu(Tensor input, Tensor target, Tensor weight, Tensor buff,
                                     Tensor grad_input, float gamma, float alpha) {
    OpPreparation::CheckOut(
        {input, target, weight},
        grad_input,
        input);

    int64_t  n_class = input.size(1);
    at::Tensor target_y = at_npu::native::NPUNativeFunctions::one_hot(target, n_class);
    target_y = at_npu::native::NPUNativeFunctions::npu_dtype_cast(target_y, input.scalar_type());
    at::Tensor dout = at::ones_like(input);
    int64_t weight_size = weight.size(0);
    at::Tensor weight_y = at::ones_like(input);
    if(weight_size > 0) {
        weight_y = aat_npu::native::NPUNativeFunctions::npu_broadcast(weight, input.sizes());
    }
    
    OpCommand cmd;
    cmd.Name("SoftmaxFocalLossGrad")
        .Input(input)
        .Input(target_y)
        .Input(dout)
        .Input(weight_y)
        .Output(grad_input)
        .Attr("gamma", gamma)
        .Attr("alpha", alpha)
        .Attr("reduction", "none")
        .Run();
}

void softmax_focal_loss_backward_impl(Tensor input, Tensor target, Tensor weight, Tensor buff,
                                      Tensor grad_input, float gamma, float alpha);


REGISTER_NPU_IMPL(sigmoid_focal_loss_forward_impl, sigmoid_focal_loss_forward_npu);

REGISTER_NPU_IMPL(sigmoid_focal_loss_backward_impl, sigmoid_focal_loss_backward_npu);

REGISTER_NPU_IMPL(softmax_focal_loss_forward_impl, softmax_focal_loss_forward_npu);

REGISTER_NPU_IMPL(softmax_focal_loss_backward_impl, softmax_focal_loss_backward_npu);


